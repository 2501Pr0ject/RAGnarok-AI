"""Regression detector for RAG evaluation results.

This module provides the RegressionDetector class for detecting
quality regressions between baseline and current evaluation results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.regression.models import (
    RegressionAlert,
    RegressionResult,
    RegressionThresholds,
)

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import EvaluationResult


# Metrics where lower is better (increase = regression)
LOWER_IS_BETTER: set[str] = {"latency_ms", "avg_latency_ms", "total_latency_ms"}


class RegressionDetector:
    """Detect regressions between baseline and current evaluation results.

    The detector compares evaluation metrics and identifies regressions
    based on configurable thresholds.

    Attributes:
        baseline: Baseline evaluation result to compare against.
        thresholds: Thresholds for regression detection.

    Example:
        >>> detector = RegressionDetector(
        ...     baseline=baseline_result,
        ...     thresholds=RegressionThresholds(precision_drop=0.03),
        ... )
        >>> result = detector.detect(current_result)
        >>> if result.has_regressions:
        ...     for alert in result.alerts:
        ...         print(alert.message)
    """

    def __init__(
        self,
        baseline: EvaluationResult,
        thresholds: RegressionThresholds | None = None,
    ) -> None:
        """Initialize detector with baseline.

        Args:
            baseline: Baseline evaluation result to compare against.
            thresholds: Thresholds for regression detection. Defaults to RegressionThresholds().
        """
        self.baseline = baseline
        self.thresholds = thresholds or RegressionThresholds()
        self._baseline_summary = self._compute_summary(baseline)

    def _compute_summary(self, result: EvaluationResult) -> dict[str, float]:
        """Compute summary metrics including latency.

        Args:
            result: Evaluation result to summarize.

        Returns:
            Dict with precision, recall, mrr, ndcg, and latency_ms.
        """
        summary = result.summary().copy()

        # Add average latency
        if result.query_results:
            avg_latency = sum(qr.latency_ms for qr in result.query_results) / len(result.query_results)
            summary["latency_ms"] = avg_latency
        elif result.total_latency_ms > 0 and result.metrics:
            summary["latency_ms"] = result.total_latency_ms / len(result.metrics)

        return summary

    def _get_threshold(self, metric: str) -> float:
        """Get threshold for a specific metric.

        Mapping:
            - precision → precision_drop
            - recall → recall_drop
            - mrr → mrr_drop
            - ndcg → ndcg_drop
            - latency_ms, avg_latency_ms → latency_increase

        Args:
            metric: Metric name.

        Returns:
            Threshold value (as fraction, e.g., 0.05 for 5%).
        """
        mapping = {
            "precision": self.thresholds.precision_drop,
            "recall": self.thresholds.recall_drop,
            "mrr": self.thresholds.mrr_drop,
            "ndcg": self.thresholds.ndcg_drop,
            "latency_ms": self.thresholds.latency_increase,
            "avg_latency_ms": self.thresholds.latency_increase,
            "total_latency_ms": self.thresholds.latency_increase,
        }
        return mapping.get(metric, 0.05)  # Default 5%

    def detect(self, current: EvaluationResult) -> RegressionResult:
        """Detect regressions vs baseline.

        Args:
            current: Current evaluation result to check for regressions.

        Returns:
            RegressionResult with any detected alerts.

        Raises:
            ValueError: If evaluations have different query counts (incompatible).

        Example:
            >>> result = detector.detect(current_evaluation)
            >>> if result.has_critical:
            ...     raise RuntimeError("Critical regression detected!")
        """
        # Validate compatibility
        if len(current.metrics) != len(self.baseline.metrics):
            raise ValueError(
                f"Incompatible evaluations: baseline has {len(self.baseline.metrics)} queries, "
                f"current has {len(current.metrics)} queries"
            )

        current_summary = self._compute_summary(current)
        alerts: list[RegressionAlert] = []

        for metric, baseline_value in self._baseline_summary.items():
            # Skip num_queries - not a performance metric
            if metric == "num_queries":
                continue

            current_value = current_summary.get(metric, 0.0)
            threshold = self._get_threshold(metric)

            # Skip if baseline is zero (avoid division by zero)
            if baseline_value == 0:
                continue

            # Calculate percentage change
            change_percent = ((current_value - baseline_value) / baseline_value) * 100

            # Determine if this is a regression
            if metric in LOWER_IS_BETTER:
                # For latency: increase is bad
                is_regression = change_percent > threshold * 100
                critical_threshold = threshold * self.thresholds.critical_multiplier * 100
                is_critical = change_percent > critical_threshold
            else:
                # For quality metrics: drop is bad (negative change)
                is_regression = change_percent < -threshold * 100
                critical_threshold = threshold * self.thresholds.critical_multiplier * 100
                is_critical = change_percent < -critical_threshold

            if is_regression:
                alerts.append(
                    RegressionAlert(
                        metric=metric,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        change_percent=change_percent,
                        threshold_percent=threshold * 100,
                        severity="critical" if is_critical else "warning",
                    )
                )

        return RegressionResult(
            alerts=alerts,
            baseline_summary=self._baseline_summary,
            current_summary=current_summary,
        )
