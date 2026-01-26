"""Batch processing for large-scale RAG evaluation.

This module provides BatchEvaluator for efficiently evaluating large testsets
with memory management, progress tracking, and resume capabilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from ragnarok_ai.cache.base import CacheProtocol
    from ragnarok_ai.core.checkpoint import CheckpointData, CheckpointManager
    from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
    from ragnarok_ai.core.protocols import RAGProtocol
    from ragnarok_ai.core.types import Query, TestSet

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        batch_size: Number of queries per batch. Default 100.
        max_concurrency: Parallel queries within each batch. Default 10.
        timeout: Timeout in seconds per query. None for no timeout.
        max_retries: Maximum retries per query on failure.
        retry_delay: Delay in seconds between retries.
        checkpoint_dir: Directory for checkpoints. None to disable.
        checkpoint_interval: Save checkpoint every N batches.
        k: Number of top results for @K metrics.
    """

    batch_size: int = 100
    max_concurrency: int = 10
    timeout: float | None = None
    max_retries: int = 0
    retry_delay: float = 1.0
    checkpoint_dir: str | None = None
    checkpoint_interval: int = 1
    k: int = 10


@dataclass
class BatchProgress:
    """Progress information for batch processing.

    Attributes:
        batch_index: Current batch index (0-based).
        total_batches: Total number of batches.
        queries_completed: Total queries completed so far.
        queries_total: Total queries to process.
        current_batch_size: Size of the current batch.
    """

    batch_index: int
    total_batches: int
    queries_completed: int
    queries_total: int
    current_batch_size: int

    @property
    def percent(self) -> float:
        """Get progress as percentage."""
        if self.queries_total == 0:
            return 100.0
        return (self.queries_completed / self.queries_total) * 100


@dataclass
class BatchResult:
    """Result of a single batch evaluation.

    Attributes:
        batch_index: Index of this batch.
        results: List of QueryResult from this batch.
        errors: List of (query, exception) tuples for failed queries.
        latency_ms: Total time for this batch in milliseconds.
    """

    batch_index: int
    results: list[QueryResult] = field(default_factory=list)
    errors: list[tuple[Query, Exception]] = field(default_factory=list)
    latency_ms: float = 0.0


class BatchEvaluator:
    """Evaluate large testsets in batches for memory efficiency and resumability.

    BatchEvaluator splits a large testset into smaller batches, evaluates them
    sequentially (with parallel queries within each batch), and optionally
    saves checkpoints for crash recovery.

    Example:
        >>> from ragnarok_ai import BatchEvaluator, BatchConfig
        >>>
        >>> config = BatchConfig(
        ...     batch_size=100,
        ...     max_concurrency=20,
        ...     checkpoint_dir="./checkpoints",
        ... )
        >>> evaluator = BatchEvaluator(rag_pipeline, config=config)
        >>>
        >>> # Stream results batch by batch
        >>> async for batch in evaluator.evaluate_batches(testset):
        ...     print(f"Batch {batch.batch_index}: {len(batch.results)} results")
        >>>
        >>> # Or get aggregated result
        >>> result = await evaluator.evaluate(testset)
        >>> print(result.summary())
    """

    def __init__(
        self,
        rag_pipeline: RAGProtocol,
        config: BatchConfig | None = None,
        cache: CacheProtocol | None = None,
        pipeline_id: str | None = None,
    ) -> None:
        """Initialize BatchEvaluator.

        Args:
            rag_pipeline: RAG pipeline implementing RAGProtocol.
            config: Batch processing configuration.
            cache: Optional cache for evaluation results.
            pipeline_id: Optional identifier for cache key isolation.
        """
        self.rag = rag_pipeline
        self.config = config or BatchConfig()
        self.cache = cache
        self.pipeline_id = pipeline_id
        self._checkpoint_manager: CheckpointManager | None = None

        if self.config.checkpoint_dir:
            from ragnarok_ai.core.checkpoint import CheckpointManager

            self._checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

    async def evaluate(
        self,
        testset: TestSet,
        *,
        on_progress: Callable[[BatchProgress], Any] | None = None,
        resume_from: str | None = None,
    ) -> EvaluationResult:
        """Evaluate entire testset in batches, return aggregated result.

        Args:
            testset: Test set to evaluate.
            on_progress: Optional callback for progress updates.
            resume_from: Optional checkpoint ID to resume from.

        Returns:
            Aggregated EvaluationResult with all metrics and responses.
        """
        from ragnarok_ai.core.evaluate import EvaluationResult

        all_metrics: list = []
        all_responses: list[str] = []
        all_query_results: list[QueryResult] = []
        all_errors: list[tuple[Query, Exception]] = []
        total_latency = 0.0

        async for batch_result in self.evaluate_batches(testset, on_progress=on_progress, resume_from=resume_from):
            total_latency += batch_result.latency_ms
            all_errors.extend(batch_result.errors)

            for qr in batch_result.results:
                all_query_results.append(qr)
                if qr.error is None:
                    all_metrics.append(qr.metric)
                    all_responses.append(qr.answer)

        return EvaluationResult(
            testset=testset,
            metrics=all_metrics,
            responses=all_responses,
            query_results=all_query_results,
            total_latency_ms=total_latency,
            errors=all_errors,
        )

    async def evaluate_batches(
        self,
        testset: TestSet,
        *,
        on_progress: Callable[[BatchProgress], Any] | None = None,
        resume_from: str | None = None,
    ) -> AsyncIterator[BatchResult]:
        """Evaluate testset batch by batch, yielding results as they complete.

        Args:
            testset: Test set to evaluate.
            on_progress: Optional callback for progress updates.
            resume_from: Optional checkpoint ID to resume from.

        Yields:
            BatchResult for each completed batch.
        """
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import TestSet as TestSetType

        queries = testset.queries
        total = len(queries)
        batch_size = self.config.batch_size
        total_batches = (total + batch_size - 1) // batch_size

        # Resume from checkpoint if provided
        start_batch = 0
        checkpoint: CheckpointData | None = None

        if resume_from and self._checkpoint_manager:
            try:
                checkpoint = self._checkpoint_manager.load_by_id(resume_from)
                start_batch = checkpoint.completed_items // batch_size
                logger.info(f"Resuming from checkpoint {resume_from} at batch {start_batch}")
            except FileNotFoundError:
                logger.warning(f"Checkpoint {resume_from} not found, starting from beginning")

        # Create new checkpoint if checkpointing is enabled
        if self._checkpoint_manager and checkpoint is None:
            testset_name = getattr(testset, "name", None) or "batch_eval"
            checkpoint = self._checkpoint_manager.create(
                task_type="batch_evaluation",
                total_items=total,
                config={
                    "batch_size": batch_size,
                    "max_concurrency": self.config.max_concurrency,
                    "k": self.config.k,
                    "pipeline_id": self.pipeline_id,
                },
                metadata={"testset_name": testset_name},
            )
            logger.info(f"Created checkpoint {checkpoint.checkpoint_id}")

        completed_queries = start_batch * batch_size

        for batch_idx in range(start_batch, total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_queries = queries[start:end]

            # Create mini testset for this batch
            batch_testset = TestSetType(queries=batch_queries)

            # Evaluate batch
            batch_start = time.perf_counter()

            result = await evaluate(
                self.rag,
                batch_testset,
                k=self.config.k,
                max_concurrency=self.config.max_concurrency,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
                cache=self.cache,
                pipeline_id=self.pipeline_id,
                fail_fast=False,  # Continue on errors in batch mode
            )

            batch_latency = (time.perf_counter() - batch_start) * 1000
            completed_queries += len(batch_queries)

            # Build batch result
            batch_result = BatchResult(
                batch_index=batch_idx,
                results=result.query_results,
                errors=result.errors,
                latency_ms=batch_latency,
            )

            # Progress callback
            if on_progress:
                progress = BatchProgress(
                    batch_index=batch_idx,
                    total_batches=total_batches,
                    queries_completed=completed_queries,
                    queries_total=total,
                    current_batch_size=len(batch_queries),
                )
                callback_result = on_progress(progress)
                # Support async callbacks
                import asyncio

                if asyncio.iscoroutine(callback_result):
                    await callback_result

            # Update checkpoint
            if checkpoint and self._checkpoint_manager:
                # Convert results to dicts for checkpoint storage
                batch_results_dicts = [
                    {
                        "query_text": qr.query.text,
                        "answer": qr.answer,
                        "latency_ms": qr.latency_ms,
                        "error": str(qr.error) if qr.error else None,
                        "metrics": {
                            "precision": qr.metric.precision,
                            "recall": qr.metric.recall,
                            "mrr": qr.metric.mrr,
                            "ndcg": qr.metric.ndcg,
                        },
                    }
                    for qr in result.query_results
                ]

                if (batch_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint = self._checkpoint_manager.save_batch_progress(checkpoint, batch_results_dicts)
                    logger.debug(f"Checkpoint saved at batch {batch_idx}")

            yield batch_result

        # Clear checkpoint on successful completion
        if checkpoint and self._checkpoint_manager:
            self._checkpoint_manager.cleanup(checkpoint)
            logger.info(f"Evaluation complete, checkpoint {checkpoint.checkpoint_id} removed")

    def get_checkpoint_id(self) -> str | None:
        """Get the current checkpoint ID if checkpointing is enabled.

        Returns:
            Checkpoint ID or None if no active checkpoint.
        """
        # This would need to be tracked during evaluation
        # For now, users should use list_checkpoints() to find checkpoints
        return None

    def list_checkpoints(self) -> list[CheckpointData]:
        """List all batch evaluation checkpoints.

        Returns:
            List of checkpoint data for batch evaluations.
        """
        if not self._checkpoint_manager:
            return []
        return self._checkpoint_manager.list_checkpoints(task_type="batch_evaluation")
