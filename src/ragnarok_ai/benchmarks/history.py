"""High-level API for benchmark tracking.

This module provides BenchmarkHistory, the main interface for recording
and querying benchmark results.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ragnarok_ai.benchmarks.models import BenchmarkRecord
from ragnarok_ai.benchmarks.storage import JSONFileStore, StorageProtocol

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import EvaluationResult
    from ragnarok_ai.core.types import TestSet
    from ragnarok_ai.regression import RegressionResult, RegressionThresholds


class BenchmarkHistory:
    """High-level API for benchmark tracking.

    Provides methods for recording evaluation results, querying history,
    managing baselines, and comparing against baselines for regression detection.

    Example:
        >>> history = BenchmarkHistory()
        >>> record = await history.record(result, "my-config", testset)
        >>> await history.set_baseline(record.id)
        >>> regression = await history.compare_to_baseline("my-config", new_result, testset)
    """

    def __init__(self, store: StorageProtocol | None = None) -> None:
        """Initialize with storage backend.

        Args:
            store: Storage backend (default: JSONFileStore).
        """
        self._store: StorageProtocol = store or JSONFileStore()

    def _compute_metrics(self, result: EvaluationResult) -> dict[str, float]:
        """Compute metrics from evaluation result including latency.

        Args:
            result: Evaluation result.

        Returns:
            Dict with precision, recall, mrr, ndcg, and latency_ms.
        """
        metrics = result.summary().copy()

        # Add average latency
        if result.query_results:
            avg_latency = sum(qr.latency_ms for qr in result.query_results) / len(result.query_results)
            metrics["latency_ms"] = avg_latency
        elif result.total_latency_ms > 0 and result.metrics:
            metrics["latency_ms"] = result.total_latency_ms / len(result.metrics)

        return metrics

    def _metrics_to_evaluation(
        self,
        record: BenchmarkRecord,
        testset: TestSet,
    ) -> EvaluationResult:
        """Create a minimal EvaluationResult from stored metrics.

        This is used to create a baseline for RegressionDetector.

        Args:
            record: Benchmark record with stored metrics.
            testset: TestSet for context.

        Returns:
            Minimal EvaluationResult with metrics.
        """
        from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
        from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

        # Extract metrics
        metrics = record.metrics
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        mrr = metrics.get("mrr", 0.0)
        ndcg = metrics.get("ndcg", 0.0)
        latency_ms = metrics.get("latency_ms", 0.0)
        num_queries = int(metrics.get("num_queries", len(testset.queries)))

        # Create RetrievalMetrics for each query (same values for simplicity)
        retrieval_metrics = RetrievalMetrics(
            precision=precision,
            recall=recall,
            mrr=mrr,
            ndcg=ndcg,
            k=10,
        )

        # Create query results with latency
        query_results: list[QueryResult] = []
        for query in testset.queries[:num_queries]:
            query_results.append(
                QueryResult(
                    query=query,
                    metric=retrieval_metrics,
                    answer="",
                    latency_ms=latency_ms,
                )
            )

        return EvaluationResult(
            testset=testset,
            metrics=[retrieval_metrics] * num_queries,
            responses=[""] * num_queries,
            query_results=query_results,
            total_latency_ms=latency_ms * num_queries,
        )

    async def record(
        self,
        result: EvaluationResult,
        config_name: str,
        testset: TestSet,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """Store evaluation result as benchmark record.

        Args:
            result: Evaluation result to record.
            config_name: Name of the configuration.
            testset: TestSet used for evaluation.
            metadata: Optional custom metadata.

        Returns:
            The created benchmark record.
        """
        record = BenchmarkRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            config_name=config_name,
            testset_hash=testset.hash_short,
            testset_name=testset.name or "unnamed",
            metrics=self._compute_metrics(result),
            metadata=metadata or {},
            is_baseline=False,
        )

        await self._store.save(record)
        return record

    async def get_latest(self, config_name: str) -> BenchmarkRecord | None:
        """Get the most recent record for a config.

        Args:
            config_name: The configuration name.

        Returns:
            The most recent record, or None if no records exist.
        """
        records = await self._store.list(config_name=config_name, limit=1)
        return records[0] if records else None

    async def get_history(
        self,
        config_name: str,
        limit: int = 100,
    ) -> list[BenchmarkRecord]:
        """Get historical records for a config.

        Args:
            config_name: The configuration name.
            limit: Maximum number of records to return.

        Returns:
            List of records, sorted by timestamp descending.
        """
        return await self._store.list(config_name=config_name, limit=limit)

    async def get_baseline(self, config_name: str) -> BenchmarkRecord | None:
        """Get the baseline record for a config.

        Args:
            config_name: The configuration name.

        Returns:
            The baseline record if set, None otherwise.
        """
        return await self._store.get_baseline(config_name)

    async def set_baseline(self, record_id: str) -> None:
        """Mark a record as baseline.

        Args:
            record_id: The record ID to mark as baseline.

        Raises:
            ValueError: If record not found.
        """
        await self._store.set_baseline(record_id)

    async def compare_to_baseline(
        self,
        config_name: str,
        current: EvaluationResult,
        testset: TestSet,
        thresholds: RegressionThresholds | None = None,
    ) -> RegressionResult | None:
        """Compare current result to stored baseline.

        Args:
            config_name: The configuration name.
            current: Current evaluation result to compare.
            testset: TestSet used for evaluation (must match baseline).
            thresholds: Thresholds for regression detection.

        Returns:
            RegressionResult if baseline exists, None otherwise.

        Raises:
            ValueError: If testset hash doesn't match baseline.
        """
        from ragnarok_ai.regression import RegressionDetector

        baseline_record = await self.get_baseline(config_name)
        if baseline_record is None:
            return None

        # Validate testset hash
        current_hash = testset.hash_short
        if baseline_record.testset_hash != current_hash:
            raise ValueError(f"Testset hash mismatch: baseline={baseline_record.testset_hash}, current={current_hash}")

        # Create EvaluationResult from stored metrics
        baseline_evaluation = self._metrics_to_evaluation(baseline_record, testset)

        # Use RegressionDetector
        detector = RegressionDetector(
            baseline=baseline_evaluation,
            thresholds=thresholds,
        )

        return detector.detect(current)
