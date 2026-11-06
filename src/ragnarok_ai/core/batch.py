"""Batch processing for large-scale RAG evaluation.

This module provides BatchEvaluator for efficiently evaluating large testsets
with memory management, progress tracking, and resume capabilities.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import CheckpointFormatError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from ragnarok_ai.cache.base import CacheProtocol
    from ragnarok_ai.core.checkpoint import CheckpointData, CheckpointManager
    from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
    from ragnarok_ai.core.protocols import RAGProtocol
    from ragnarok_ai.core.types import Query, TestSet
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

logger = logging.getLogger(__name__)

# Layout of the results stored in a batch-evaluation checkpoint.
# Version 1 (before 1.12.3) stored results by batch number, without the data
# needed to rebuild a QueryResult; such checkpoints cannot be resumed.
# Version 2 stores one entry per query, keyed by ``query_key``.
BATCH_CHECKPOINT_FORMAT = 2
BATCH_TASK_TYPE = "batch_evaluation"


class CheckpointedError(Exception):
    """Error restored from a checkpoint; only its message survived the round trip."""


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
        checkpoint_interval: Save checkpoint every N batches. The last batch
            is always saved, whatever the interval.
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
        restored: True when the results come from a checkpoint instead of
            having been computed in this run.
    """

    batch_index: int
    results: list[QueryResult] = field(default_factory=list)
    errors: list[tuple[Query, Exception]] = field(default_factory=list)
    latency_ms: float = 0.0
    restored: bool = False


def query_key(query: Query) -> str:
    """Stable identifier of a query inside a checkpoint.

    ``metadata["id"]`` when the testset provides one, otherwise a hash of the
    query text. Independent of the query's position and of the batch size.
    """
    explicit = query.metadata.get("id") if query.metadata else None
    if explicit is not None and str(explicit):
        return str(explicit)
    return hashlib.sha256(query.text.encode("utf-8")).hexdigest()[:16]


def query_keys(queries: Sequence[Query]) -> list[str]:
    """One unique key per query; duplicates get an occurrence suffix."""
    seen: dict[str, int] = {}
    keys: list[str] = []
    for query in queries:
        base = query_key(query)
        n = seen.get(base, 0)
        seen[base] = n + 1
        keys.append(base if n == 0 else f"{base}#{n}")
    return keys


def _query_result_to_dict(key: str, qr: QueryResult) -> dict[str, Any]:
    """Everything needed to rebuild ``qr`` from a checkpoint."""
    return {
        "query_key": key,
        "query_text": qr.query.text,
        "answer": qr.answer,
        "latency_ms": qr.latency_ms,
        "error": str(qr.error) if qr.error else None,
        "retrieved_doc_ids": list(qr.retrieved_doc_ids),
        "metrics": {
            "precision": qr.metric.precision,
            "recall": qr.metric.recall,
            "mrr": qr.metric.mrr,
            "ndcg": qr.metric.ndcg,
            "k": qr.metric.k,
        },
    }


def _query_result_from_dict(entry: dict[str, Any], query: Query) -> QueryResult:
    """Rebuild a QueryResult stored by :func:`_query_result_to_dict`."""
    from ragnarok_ai.core.evaluate import QueryResult
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

    error = entry.get("error")
    return QueryResult(
        query=query,
        metric=RetrievalMetrics(**entry["metrics"]),
        answer=entry["answer"],
        latency_ms=float(entry.get("latency_ms", 0.0)),
        error=CheckpointedError(error) if error else None,
        retrieved_doc_ids=list(entry.get("retrieved_doc_ids", [])),
    )


class BatchEvaluator:
    """Evaluate large testsets in batches for memory efficiency and resumability.

    BatchEvaluator splits a large testset into smaller batches, evaluates them
    sequentially (with parallel queries within each batch), and optionally
    saves checkpoints for crash recovery.

    Checkpoints are keyed by query, not by batch number: a resumed run
    restores the results of every query already evaluated (whatever the
    ``batch_size`` was) and evaluates only the remaining ones. Queries that
    failed before the interruption are evaluated again.

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
        >>>
        >>> # After a crash: restore what was done, evaluate the rest
        >>> result = await evaluator.evaluate(testset, resume_from=checkpoint_id)
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
        self._checkpoint_id: str | None = None

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
            resume_from: Optional checkpoint ID to resume from. Results of
                the queries already evaluated are restored from the
                checkpoint and included in the returned result.

        Returns:
            Aggregated EvaluationResult with all metrics and responses.
        """
        from ragnarok_ai.core.evaluate import EvaluationResult

        all_metrics: list[RetrievalMetrics] = []
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

        When resuming, the first yielded batch carries the restored results
        (``restored=True``); the following batches are computed.

        Args:
            testset: Test set to evaluate.
            on_progress: Optional callback for progress updates.
            resume_from: Optional checkpoint ID to resume from.

        Yields:
            BatchResult for each completed batch.

        Raises:
            CheckpointFormatError: If the checkpoint was written in a format
                that cannot be resumed (earlier than version 2).
        """
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import TestSet as TestSetType

        queries = testset.queries
        keys = query_keys(queries)
        total = len(queries)
        batch_size = self.config.batch_size

        checkpoint: CheckpointData | None = None
        stored: dict[str, dict[str, Any]] = {}

        if resume_from and self._checkpoint_manager:
            try:
                checkpoint = self._checkpoint_manager.load_by_id(resume_from)
            except FileNotFoundError:
                logger.warning(f"Checkpoint {resume_from} not found, starting from beginning")
            else:
                self._check_resumable(checkpoint)
                # Latest entry wins; failed queries are not restored, they are evaluated again
                for entry in checkpoint.results:
                    key = entry.get("query_key")
                    if key is None:
                        continue
                    if entry.get("error") is None:
                        stored[key] = entry
                    else:
                        stored.pop(key, None)
                unknown = set(stored) - set(keys)
                if unknown:
                    logger.warning(
                        f"Checkpoint {resume_from}: {len(unknown)} stored queries are not in this testset and are ignored"
                    )
                logger.info(f"Resuming from checkpoint {resume_from}: {len(stored)} queries restored")

        if self._checkpoint_manager and checkpoint is None:
            testset_name = getattr(testset, "name", None) or "batch_eval"
            checkpoint = self._checkpoint_manager.create(
                task_type=BATCH_TASK_TYPE,
                total_items=total,
                config={
                    "batch_size": batch_size,
                    "max_concurrency": self.config.max_concurrency,
                    "k": self.config.k,
                    "pipeline_id": self.pipeline_id,
                },
                metadata={"testset_name": testset_name},
                format_version=BATCH_CHECKPOINT_FORMAT,
            )
            logger.info(f"Created checkpoint {checkpoint.checkpoint_id}")

        self._checkpoint_id = checkpoint.checkpoint_id if checkpoint else None

        restored_results = [
            _query_result_from_dict(stored[key], query)
            for query, key in zip(queries, keys, strict=True)
            if key in stored
        ]
        pending = [(query, key) for query, key in zip(queries, keys, strict=True) if key not in stored]
        pending_batches = (len(pending) + batch_size - 1) // batch_size
        batch_offset = 1 if restored_results else 0
        total_batches = pending_batches + batch_offset
        completed_queries = len(restored_results)

        if restored_results:
            yield BatchResult(batch_index=0, results=restored_results, restored=True)

        unsaved: list[dict[str, Any]] = []

        for pending_idx in range(pending_batches):
            batch_idx = pending_idx + batch_offset
            start = pending_idx * batch_size
            chunk = pending[start : start + batch_size]
            batch_queries = [query for query, _ in chunk]

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

            # Update checkpoint: every result is kept until the next save, so
            # an interval above 1 delays the save but never drops a batch
            if checkpoint and self._checkpoint_manager:
                # Keys of this chunk grouped by base key; duplicates are consumed in order
                keys_by_base: dict[str, list[str]] = {}
                for query, key in chunk:
                    keys_by_base.setdefault(query_key(query), []).append(key)
                for qr in result.query_results:
                    unsaved.append(_query_result_to_dict(keys_by_base[query_key(qr.query)].pop(0), qr))

                is_last = pending_idx == pending_batches - 1
                if is_last or (pending_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint = self._checkpoint_manager.save_batch_progress(checkpoint, unsaved)
                    unsaved = []
                    logger.debug(f"Checkpoint saved at batch {batch_idx}")

            yield batch_result

        # Clear checkpoint on successful completion
        if checkpoint and self._checkpoint_manager:
            self._checkpoint_manager.cleanup(checkpoint)
            logger.info(f"Evaluation complete, checkpoint {checkpoint.checkpoint_id} removed")

    @staticmethod
    def _check_resumable(checkpoint: CheckpointData) -> None:
        """Refuse checkpoints that cannot be resumed, with the reason."""
        if checkpoint.task_type != BATCH_TASK_TYPE:
            msg = f"Checkpoint {checkpoint.checkpoint_id} is a {checkpoint.task_type!r} checkpoint, not a batch evaluation"
            raise CheckpointFormatError(msg)
        if checkpoint.format_version < BATCH_CHECKPOINT_FORMAT:
            msg = (
                f"Checkpoint {checkpoint.checkpoint_id} uses format {checkpoint.format_version}, written by an "
                f"earlier version of ragnarok-ai; format {BATCH_CHECKPOINT_FORMAT} is required to resume. "
                "Delete it and run the evaluation again."
            )
            raise CheckpointFormatError(msg)

    def get_checkpoint_id(self) -> str | None:
        """Get the ID of the checkpoint used by the last evaluation.

        Set when ``evaluate()`` or ``evaluate_batches()`` creates or resumes a
        checkpoint; it stays available after an interruption so the run can
        be resumed with ``resume_from``.

        Returns:
            Checkpoint ID or None if checkpointing is disabled or no
            evaluation has started.
        """
        return self._checkpoint_id

    def list_checkpoints(self) -> list[CheckpointData]:
        """List all batch evaluation checkpoints.

        Returns:
            List of checkpoint data for batch evaluations.
        """
        if not self._checkpoint_manager:
            return []
        return self._checkpoint_manager.list_checkpoints(task_type=BATCH_TASK_TYPE)
