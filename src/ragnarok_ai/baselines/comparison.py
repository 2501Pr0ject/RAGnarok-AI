"""Baseline comparison for ragnarok-ai.

This module provides tools for comparing evaluation results against baselines.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ragnarok_ai.baselines.results import BaselineResult, get_baseline_result


class MetricComparison(BaseModel):
    """Comparison of a single metric against baseline.

    Attributes:
        metric_name: Name of the metric.
        your_value: Your measured value.
        baseline_value: Expected baseline value.
        difference: Absolute difference (your - baseline).
        percent_change: Percentage change from baseline.
        is_better: Whether your value is better than baseline.
    """

    metric_name: str = Field(..., description="Metric name")
    your_value: float = Field(..., description="Your measured value")
    baseline_value: float = Field(..., description="Baseline expected value")
    difference: float = Field(..., description="Absolute difference")
    percent_change: float = Field(..., description="Percentage change")
    is_better: bool = Field(..., description="Whether your value is better")

    @property
    def status(self) -> str:
        """Get status indicator.

        Returns:
            Status string (better, worse, equal).
        """
        if abs(self.percent_change) < 1.0:
            return "equal"
        return "better" if self.is_better else "worse"

    @property
    def formatted_change(self) -> str:
        """Get formatted change string.

        Returns:
            Formatted string like '+5.2%' or '-3.1%'.
        """
        sign = "+" if self.percent_change >= 0 else ""
        return f"{sign}{self.percent_change:.1f}%"


class BaselineComparison(BaseModel):
    """Comparison of evaluation results against a baseline.

    Attributes:
        baseline_name: Name of the baseline compared against.
        metrics: Individual metric comparisons.
        overall_score: Overall comparison score (-1 to 1).
        summary_text: Human-readable summary.
    """

    baseline_name: str = Field(..., description="Baseline name")
    metrics: list[MetricComparison] = Field(default_factory=list, description="Metric comparisons")
    overall_score: float = Field(default=0.0, description="Overall score")
    summary_text: str = Field(default="", description="Summary text")

    @property
    def better_count(self) -> int:
        """Count of metrics that are better than baseline."""
        return sum(1 for m in self.metrics if m.is_better)

    @property
    def worse_count(self) -> int:
        """Count of metrics that are worse than baseline."""
        return sum(1 for m in self.metrics if not m.is_better and abs(m.percent_change) >= 1.0)

    @property
    def equal_count(self) -> int:
        """Count of metrics that are equal to baseline."""
        return sum(1 for m in self.metrics if abs(m.percent_change) < 1.0)

    def summary(self) -> str:
        """Get a formatted summary of the comparison.

        Returns:
            Multi-line summary string.
        """
        lines = [
            f"Comparison vs '{self.baseline_name}' baseline:",
            f"  Overall: {self.summary_text}",
            f"  Better: {self.better_count}, Worse: {self.worse_count}, Equal: {self.equal_count}",
            "",
            "Metrics:",
        ]

        for m in self.metrics:
            status_icon = "✓" if m.is_better else ("✗" if m.status == "worse" else "=")
            lines.append(f"  {status_icon} {m.metric_name}: {m.your_value:.3f} ({m.formatted_change} vs {m.baseline_value:.3f})")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Comparison as dictionary.
        """
        return {
            "baseline_name": self.baseline_name,
            "overall_score": self.overall_score,
            "summary": self.summary_text,
            "better_count": self.better_count,
            "worse_count": self.worse_count,
            "equal_count": self.equal_count,
            "metrics": [
                {
                    "name": m.metric_name,
                    "your_value": m.your_value,
                    "baseline_value": m.baseline_value,
                    "change": m.formatted_change,
                    "status": m.status,
                }
                for m in self.metrics
            ],
        }


def compare(
    your_results: dict[str, float],
    baseline: str | BaselineResult,
    higher_is_better: dict[str, bool] | None = None,
) -> BaselineComparison:
    """Compare your results against a baseline.

    Args:
        your_results: Dictionary of metric names to values.
        baseline: Baseline name or BaselineResult object.
        higher_is_better: Dictionary specifying if higher is better for each metric.
                         Defaults to True for most metrics, False for latency.

    Returns:
        BaselineComparison with detailed comparison.

    Example:
        >>> comparison = compare(
        ...     your_results={"precision": 0.78, "recall": 0.65, "mrr": 0.74},
        ...     baseline="balanced",
        ... )
        >>> print(comparison.summary())
    """
    # Get baseline result
    baseline_result = get_baseline_result(baseline) if isinstance(baseline, str) else baseline

    # Default higher_is_better mapping
    default_hib = {
        "precision": True,
        "recall": True,
        "mrr": True,
        "ndcg": True,
        "faithfulness": True,
        "relevance": True,
        "latency": False,
        "latency_ms": False,
    }
    hib = {**default_hib, **(higher_is_better or {})}

    # Compare each metric
    metric_comparisons: list[MetricComparison] = []

    for metric_name, your_value in your_results.items():
        baseline_value = baseline_result.get_metric(metric_name)

        if baseline_value is None:
            continue

        difference = your_value - baseline_value

        # Calculate percent change
        if baseline_value != 0:
            percent_change = (difference / baseline_value) * 100
        else:
            percent_change = 100.0 if difference > 0 else (-100.0 if difference < 0 else 0.0)

        # Determine if better
        metric_hib = hib.get(metric_name.lower(), True)
        is_better = (difference > 0) if metric_hib else (difference < 0)

        metric_comparisons.append(
            MetricComparison(
                metric_name=metric_name,
                your_value=your_value,
                baseline_value=baseline_value,
                difference=difference,
                percent_change=percent_change,
                is_better=is_better,
            )
        )

    # Calculate overall score (-1 to 1)
    if metric_comparisons:
        # Weight by absolute percent change, normalized
        total_weight = sum(abs(m.percent_change) for m in metric_comparisons)
        if total_weight > 0:
            overall_score = sum(
                (1 if m.is_better else -1) * abs(m.percent_change) for m in metric_comparisons
            ) / total_weight
        else:
            overall_score = 0.0
    else:
        overall_score = 0.0

    # Generate summary text
    if overall_score > 0.2:
        summary_text = "Your results are better than the baseline"
    elif overall_score < -0.2:
        summary_text = "Your results are below the baseline"
    else:
        summary_text = "Your results are comparable to the baseline"

    return BaselineComparison(
        baseline_name=baseline_result.baseline_name,
        metrics=metric_comparisons,
        overall_score=overall_score,
        summary_text=summary_text,
    )
