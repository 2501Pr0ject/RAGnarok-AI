"""Models for regression detection.

This module provides dataclasses for regression detection thresholds,
alerts, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class RegressionThresholds:
    """Thresholds for regression detection.

    Quality metrics (precision, recall, mrr, ndcg): alert if drop exceeds threshold.
    Latency metrics: alert if increase exceeds threshold.

    Attributes:
        precision_drop: Threshold for precision regression (default 5%).
        recall_drop: Threshold for recall regression (default 5%).
        mrr_drop: Threshold for MRR regression (default 5%).
        ndcg_drop: Threshold for NDCG regression (default 5%).
        latency_increase: Threshold for latency regression (default 20%).
        critical_multiplier: Multiplier for critical severity (default 2x).

    Example:
        >>> thresholds = RegressionThresholds(precision_drop=0.03)  # Stricter 3%
        >>> thresholds.precision_drop
        0.03
    """

    precision_drop: float = 0.05
    recall_drop: float = 0.05
    mrr_drop: float = 0.05
    ndcg_drop: float = 0.05
    latency_increase: float = 0.20
    critical_multiplier: float = 2.0


@dataclass
class RegressionAlert:
    """Alert for a detected regression.

    Attributes:
        metric: Name of the metric that regressed.
        baseline_value: Value from baseline evaluation.
        current_value: Value from current evaluation.
        change_percent: Percentage change (negative = drop, positive = increase).
        threshold_percent: Threshold that was exceeded (as percentage).
        severity: "warning" if exceeds threshold, "critical" if exceeds 2x threshold.

    Example:
        >>> alert = RegressionAlert(
        ...     metric="precision",
        ...     baseline_value=0.85,
        ...     current_value=0.78,
        ...     change_percent=-8.24,
        ...     threshold_percent=5.0,
        ...     severity="warning",
        ... )
        >>> alert.message
        'precision dropped by 8.2% (threshold: 5.0%)'
    """

    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    threshold_percent: float
    severity: Literal["warning", "critical"]

    @property
    def message(self) -> str:
        """Human-readable alert message.

        Returns:
            Formatted message describing the regression.
        """
        if self.metric in {"latency_ms", "avg_latency_ms", "total_latency_ms"}:
            direction = "increased"
            change = self.change_percent
        else:
            direction = "dropped"
            change = abs(self.change_percent)

        return f"{self.metric} {direction} by {change:.1f}% (threshold: {self.threshold_percent:.1f}%)"


@dataclass
class RegressionResult:
    """Result of regression detection.

    Attributes:
        alerts: List of regression alerts detected.
        baseline_summary: Summary metrics from baseline evaluation.
        current_summary: Summary metrics from current evaluation.
        timestamp: When the detection was performed.

    Example:
        >>> result = detector.detect(current_evaluation)
        >>> if result.has_critical:
        ...     print("Critical regressions detected!")
        >>> for alert in result.alerts:
        ...     print(alert.message)
    """

    alerts: list[RegressionAlert]
    baseline_summary: dict[str, float]
    current_summary: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_regressions(self) -> bool:
        """Check if any regressions were detected.

        Returns:
            True if there are any alerts.
        """
        return len(self.alerts) > 0

    @property
    def has_critical(self) -> bool:
        """Check if any critical regressions were detected.

        Returns:
            True if any alert has critical severity.
        """
        return any(alert.severity == "critical" for alert in self.alerts)

    @property
    def warning_count(self) -> int:
        """Count of warning-level regressions."""
        return sum(1 for alert in self.alerts if alert.severity == "warning")

    @property
    def critical_count(self) -> int:
        """Count of critical-level regressions."""
        return sum(1 for alert in self.alerts if alert.severity == "critical")

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Multi-line summary string.
        """
        if not self.alerts:
            return "No regressions detected."

        lines = [
            f"Regression Detection Summary ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})",
            f"  Critical: {self.critical_count}, Warnings: {self.warning_count}",
            "",
            "Alerts:",
        ]

        for alert in self.alerts:
            severity_marker = "[CRITICAL]" if alert.severity == "critical" else "[WARNING]"
            lines.append(f"  {severity_marker} {alert.message}")

        return "\n".join(lines)
