"""Data models for diff reports.

This module provides dataclasses for representing differences
between two evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datetime import datetime

# Standard metrics to track
STANDARD_METRICS = ("precision", "recall", "mrr", "ndcg")


@dataclass
class QueryDiff:
    """Diff for a single query between two runs.

    Attributes:
        query_text: The query text.
        query_id: Optional query identifier.
        baseline_metrics: Metrics from baseline run.
        current_metrics: Metrics from current run.
        status: Classification of the change.
    """

    query_text: str
    query_id: str | None
    baseline_metrics: dict[str, float]
    current_metrics: dict[str, float]
    status: Literal["improved", "degraded", "unchanged"]

    @property
    def precision_change(self) -> float:
        """Absolute change in precision."""
        return self.current_metrics.get("precision", 0.0) - self.baseline_metrics.get("precision", 0.0)

    @property
    def recall_change(self) -> float:
        """Absolute change in recall."""
        return self.current_metrics.get("recall", 0.0) - self.baseline_metrics.get("recall", 0.0)

    @property
    def mrr_change(self) -> float:
        """Absolute change in MRR."""
        return self.current_metrics.get("mrr", 0.0) - self.baseline_metrics.get("mrr", 0.0)

    @property
    def ndcg_change(self) -> float:
        """Absolute change in NDCG."""
        return self.current_metrics.get("ndcg", 0.0) - self.baseline_metrics.get("ndcg", 0.0)

    def _format_percent_change(self, baseline: float, current: float) -> str:
        """Format percent change, handling zero baseline."""
        if baseline == 0:
            if current == 0:
                return "0%"
            return "N/A"
        change = ((current - baseline) / baseline) * 100
        sign = "+" if change > 0 else ""
        return f"{sign}{change:.1f}%"

    def summary(self) -> str:
        """One-line summary of the query diff.

        Returns:
            Summary like 'Query "X": recall -20%, precision +5%'
        """
        changes = []
        for metric in STANDARD_METRICS:
            baseline = self.baseline_metrics.get(metric, 0.0)
            current = self.current_metrics.get(metric, 0.0)
            if baseline != current:
                pct = self._format_percent_change(baseline, current)
                changes.append(f"{metric} {pct}")

        query_preview = self.query_text[:30] + "..." if len(self.query_text) > 30 else self.query_text
        if changes:
            return f'Query "{query_preview}": {", ".join(changes)}'
        return f'Query "{query_preview}": unchanged'

    def biggest_change(self) -> tuple[str, float, float, float]:
        """Find the metric with the biggest absolute change.

        Returns:
            Tuple of (metric_name, baseline_value, current_value, change).
        """
        biggest = ("precision", 0.0, 0.0, 0.0)
        max_abs_change = 0.0

        for metric in STANDARD_METRICS:
            baseline = self.baseline_metrics.get(metric, 0.0)
            current = self.current_metrics.get(metric, 0.0)
            change = current - baseline
            if abs(change) > max_abs_change:
                max_abs_change = abs(change)
                biggest = (metric, baseline, current, change)

        return biggest


@dataclass
class DiffReport:
    """Diff between two evaluation runs.

    Attributes:
        baseline_id: Optional BenchmarkRecord ID for baseline.
        current_id: Optional BenchmarkRecord ID for current.
        baseline_name: Name for the baseline in reports.
        current_name: Name for the current run in reports.
        timestamp: When the diff was generated.
        testset_name: Name of the testset used.
        metrics_diff: Aggregate metric changes.
        improved: Queries that improved.
        degraded: Queries that degraded.
        unchanged: Queries that stayed the same.
        change_threshold: Threshold used for classification.
    """

    baseline_id: str | None
    current_id: str | None
    baseline_name: str
    current_name: str
    timestamp: datetime
    testset_name: str
    metrics_diff: dict[str, float]
    improved: list[QueryDiff]
    degraded: list[QueryDiff]
    unchanged: list[QueryDiff]
    change_threshold: float = 0.01

    # Store aggregate metrics for display
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    current_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_queries(self) -> int:
        """Total number of queries compared."""
        return len(self.improved) + len(self.degraded) + len(self.unchanged)

    @property
    def improvement_rate(self) -> float:
        """Percent of queries that improved (0.0-1.0)."""
        if self.total_queries == 0:
            return 0.0
        return len(self.improved) / self.total_queries

    @property
    def degradation_rate(self) -> float:
        """Percent of queries that degraded (0.0-1.0)."""
        if self.total_queries == 0:
            return 0.0
        return len(self.degraded) / self.total_queries

    def summary(self) -> str:
        """Quick summary of the diff.

        Returns:
            Summary like '5 improved, 3 degraded, 42 unchanged'
        """
        return f"{len(self.improved)} improved, {len(self.degraded)} degraded, {len(self.unchanged)} unchanged"

    def to_markdown(self) -> str:
        """Generate full markdown report.

        Returns:
            Markdown-formatted diff report.
        """
        lines = [
            "## RAGnarok Diff Report",
            "",
            f"**Baseline:** {self.baseline_name}",
            f"**Current:** {self.current_name}",
            f"**Testset:** {self.testset_name}",
            f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "### Summary",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
        ]

        # Add metric rows
        for metric in STANDARD_METRICS:
            baseline = self.baseline_metrics.get(metric, 0.0)
            current = self.current_metrics.get(metric, 0.0)
            diff = self.metrics_diff.get(metric, 0.0)

            # Format percent change
            if baseline == 0:
                pct_str = "N/A" if current != 0 else "0%"
            else:
                pct = (diff / baseline) * 100
                sign = "+" if pct > 0 else ""
                pct_str = f"{sign}{pct:.1f}%"
                # Add warning emoji for negative changes
                if pct < -5:
                    pct_str += " ⚠️"

            lines.append(f"| {metric.capitalize()} | {baseline:.2f} | {current:.2f} | {pct_str} |")

        # Query changes summary
        lines.extend(
            [
                "",
                "### Query Changes",
                "",
                f"- **Improved:** {len(self.improved)} queries ({self.improvement_rate * 100:.1f}%)",
                f"- **Degraded:** {len(self.degraded)} queries ({self.degradation_rate * 100:.1f}%)",
                f"- **Unchanged:** {len(self.unchanged)} queries ({(1 - self.improvement_rate - self.degradation_rate) * 100:.1f}%)",
            ]
        )

        # Top degraded queries (max 5)
        if self.degraded:
            lines.extend(
                [
                    "",
                    "### Top Degraded Queries",
                    "",
                    "| Query | Metric | Before | After | Change |",
                    "|-------|--------|--------|-------|--------|",
                ]
            )
            # Sort by biggest change (most negative first)
            sorted_degraded = sorted(
                self.degraded,
                key=lambda q: q.biggest_change()[3],
            )[:5]
            for qd in sorted_degraded:
                metric, before, after, _ = qd.biggest_change()
                query_preview = qd.query_text[:25] + "..." if len(qd.query_text) > 25 else qd.query_text
                if before == 0:
                    pct_str = "N/A"
                else:
                    pct = ((after - before) / before) * 100
                    pct_str = f"{pct:.0f}%"
                lines.append(f'| "{query_preview}" | {metric} | {before:.2f} | {after:.2f} | {pct_str} |')

        # Top improved queries (max 5)
        if self.improved:
            lines.extend(
                [
                    "",
                    "### Top Improved Queries",
                    "",
                    "| Query | Metric | Before | After | Change |",
                    "|-------|--------|--------|-------|--------|",
                ]
            )
            # Sort by biggest change (most positive first)
            sorted_improved = sorted(
                self.improved,
                key=lambda q: q.biggest_change()[3],
                reverse=True,
            )[:5]
            for qd in sorted_improved:
                metric, before, after, _ = qd.biggest_change()
                query_preview = qd.query_text[:25] + "..." if len(qd.query_text) > 25 else qd.query_text
                if before == 0:
                    pct_str = "N/A" if after != 0 else "0%"
                else:
                    pct = ((after - before) / before) * 100
                    pct_str = f"+{pct:.0f}%"
                lines.append(f'| "{query_preview}" | {metric} | {before:.2f} | {after:.2f} | {pct_str} |')

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation of the diff report.
        """
        return {
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "baseline_name": self.baseline_name,
            "current_name": self.current_name,
            "timestamp": self.timestamp.isoformat(),
            "testset_name": self.testset_name,
            "metrics_diff": self.metrics_diff,
            "baseline_metrics": self.baseline_metrics,
            "current_metrics": self.current_metrics,
            "change_threshold": self.change_threshold,
            "summary": {
                "total_queries": self.total_queries,
                "improved": len(self.improved),
                "degraded": len(self.degraded),
                "unchanged": len(self.unchanged),
                "improvement_rate": self.improvement_rate,
                "degradation_rate": self.degradation_rate,
            },
            "improved": [
                {
                    "query_text": q.query_text,
                    "query_id": q.query_id,
                    "baseline_metrics": q.baseline_metrics,
                    "current_metrics": q.current_metrics,
                    "status": q.status,
                }
                for q in self.improved
            ],
            "degraded": [
                {
                    "query_text": q.query_text,
                    "query_id": q.query_id,
                    "baseline_metrics": q.baseline_metrics,
                    "current_metrics": q.current_metrics,
                    "status": q.status,
                }
                for q in self.degraded
            ],
            "unchanged": [
                {
                    "query_text": q.query_text,
                    "query_id": q.query_id,
                    "baseline_metrics": q.baseline_metrics,
                    "current_metrics": q.current_metrics,
                    "status": q.status,
                }
                for q in self.unchanged
            ],
        }
