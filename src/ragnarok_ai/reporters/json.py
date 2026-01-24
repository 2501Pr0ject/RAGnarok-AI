"""JSON reporter for ragnarok-ai.

This module provides JSON output for evaluation results,
suitable for CI/CD pipelines and machine processing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.reporters.console import DEFAULT_THRESHOLDS, Threshold

if TYPE_CHECKING:
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics


class JSONReporter:
    """Reporter that outputs evaluation results as JSON.

    Provides structured JSON output with metrics, timestamps,
    and status indicators based on configurable thresholds.

    Attributes:
        thresholds: Threshold configuration for status indicators.
        indent: JSON indentation level (None for compact).

    Example:
        >>> reporter = JSONReporter()
        >>> json_str = reporter.report(metrics)
        >>> print(json_str)
        {
          "timestamp": "2024-01-15T10:30:00Z",
          "metrics": {
            "precision": {"value": 0.82, "status": "pass"},
            ...
          }
        }
    """

    def __init__(
        self,
        thresholds: dict[str, Threshold] | None = None,
        indent: int | None = 2,
    ) -> None:
        """Initialize JSONReporter.

        Args:
            thresholds: Custom thresholds for status indicators.
            indent: JSON indentation level. Defaults to 2. Use None for compact output.
        """
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.indent = indent

    def _get_status(self, metric_name: str, value: float) -> str:
        """Get status string for a metric value.

        Args:
            metric_name: Name of the metric.
            value: The metric value.

        Returns:
            Status string: "pass", "warn", or "fail".
        """
        threshold = self.thresholds.get(metric_name, Threshold())

        # Special case: hallucination (lower is better)
        if metric_name == "hallucination":
            if value <= threshold.good:
                return "pass"
            if value <= threshold.warning:
                return "warn"
            return "fail"

        # Normal case: higher is better
        if value >= threshold.good:
            return "pass"
        if value >= threshold.warning:
            return "warn"
        return "fail"

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _metrics_to_dict(
        self,
        metrics: RetrievalMetrics,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert RetrievalMetrics to a dictionary.

        Args:
            metrics: The retrieval metrics to convert.
            metadata: Optional metadata to include.

        Returns:
            Dictionary representation of the metrics.
        """
        k = metrics.k

        return {
            "timestamp": self._get_timestamp(),
            "k": k,
            "metrics": {
                "precision": {
                    "value": metrics.precision,
                    "status": self._get_status("precision", metrics.precision),
                },
                "recall": {
                    "value": metrics.recall,
                    "status": self._get_status("recall", metrics.recall),
                },
                "mrr": {
                    "value": metrics.mrr,
                    "status": self._get_status("mrr", metrics.mrr),
                },
                "ndcg": {
                    "value": metrics.ndcg,
                    "status": self._get_status("ndcg", metrics.ndcg),
                },
            },
            "metadata": metadata or {},
        }

    def report(
        self,
        metrics: RetrievalMetrics,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate JSON report for retrieval metrics.

        Args:
            metrics: The retrieval metrics to report.
            metadata: Optional metadata to include in the report.

        Returns:
            JSON string representation of the metrics.

        Example:
            >>> json_str = reporter.report(metrics, metadata={"query": "test"})
        """
        data = self._metrics_to_dict(metrics, metadata)
        return json.dumps(data, indent=self.indent)

    def report_to_file(
        self,
        metrics: RetrievalMetrics,
        path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write JSON report to a file.

        Args:
            metrics: The retrieval metrics to report.
            path: Path to the output file.
            metadata: Optional metadata to include in the report.

        Example:
            >>> reporter.report_to_file(metrics, Path("results.json"))
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._metrics_to_dict(metrics, metadata)
        path.write_text(json.dumps(data, indent=self.indent))

    def report_summary(
        self,
        results: list[RetrievalMetrics],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate JSON summary report for multiple evaluations.

        Args:
            results: List of retrieval metrics from multiple queries.
            metadata: Optional metadata to include in the report.

        Returns:
            JSON string with aggregated metrics and individual results.

        Example:
            >>> json_str = reporter.report_summary(metrics_list)
        """
        if not results:
            return json.dumps(
                {
                    "timestamp": self._get_timestamp(),
                    "summary": None,
                    "results": [],
                    "metadata": metadata or {},
                },
                indent=self.indent,
            )

        n = len(results)
        k = results[0].k

        # Calculate averages
        avg_precision = sum(m.precision for m in results) / n
        avg_recall = sum(m.recall for m in results) / n
        avg_mrr = sum(m.mrr for m in results) / n
        avg_ndcg = sum(m.ndcg for m in results) / n

        # Calculate min/max for each metric
        summary = {
            "num_queries": n,
            "k": k,
            "averages": {
                "precision": {
                    "value": avg_precision,
                    "status": self._get_status("precision", avg_precision),
                },
                "recall": {
                    "value": avg_recall,
                    "status": self._get_status("recall", avg_recall),
                },
                "mrr": {
                    "value": avg_mrr,
                    "status": self._get_status("mrr", avg_mrr),
                },
                "ndcg": {
                    "value": avg_ndcg,
                    "status": self._get_status("ndcg", avg_ndcg),
                },
            },
            "ranges": {
                "precision": {
                    "min": min(m.precision for m in results),
                    "max": max(m.precision for m in results),
                },
                "recall": {
                    "min": min(m.recall for m in results),
                    "max": max(m.recall for m in results),
                },
                "mrr": {
                    "min": min(m.mrr for m in results),
                    "max": max(m.mrr for m in results),
                },
                "ndcg": {
                    "min": min(m.ndcg for m in results),
                    "max": max(m.ndcg for m in results),
                },
            },
        }

        # Individual results
        individual_results = [
            {
                "precision": m.precision,
                "recall": m.recall,
                "mrr": m.mrr,
                "ndcg": m.ndcg,
            }
            for m in results
        ]

        data = {
            "timestamp": self._get_timestamp(),
            "summary": summary,
            "results": individual_results,
            "metadata": metadata or {},
        }

        return json.dumps(data, indent=self.indent)

    def report_summary_to_file(
        self,
        results: list[RetrievalMetrics],
        path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write JSON summary report to a file.

        Args:
            results: List of retrieval metrics from multiple queries.
            path: Path to the output file.
            metadata: Optional metadata to include in the report.

        Example:
            >>> reporter.report_summary_to_file(results, Path("summary.json"))
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        json_str = self.report_summary(results, metadata)
        path.write_text(json_str)
