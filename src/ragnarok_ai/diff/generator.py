"""Diff generation for evaluation results.

This module provides the compute_diff function for comparing
two evaluation runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ragnarok_ai.diff.models import STANDARD_METRICS, DiffReport, QueryDiff

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import EvaluationResult


def _extract_query_metrics(
    result: EvaluationResult,
    index: int,
) -> dict[str, float]:
    """Extract metrics for a single query.

    Args:
        result: The evaluation result.
        index: Query index.

    Returns:
        Dict of metric name to value.
    """
    if index >= len(result.metrics):
        return {}

    metric = result.metrics[index]
    return {
        "precision": metric.precision,
        "recall": metric.recall,
        "mrr": metric.mrr,
        "ndcg": metric.ndcg,
    }


def _classify_query(
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
    threshold: float,
) -> Literal["improved", "degraded", "unchanged"]:
    """Classify query based on aggregate metric change.

    Strategy:
    - Compute average change across precision, recall, mrr, ndcg
    - If avg_change > threshold: improved
    - If avg_change < -threshold: degraded
    - Else: unchanged

    Args:
        baseline_metrics: Metrics from baseline run.
        current_metrics: Metrics from current run.
        threshold: Minimum change to be significant.

    Returns:
        Classification status.
    """
    changes = []
    for metric in STANDARD_METRICS:
        baseline = baseline_metrics.get(metric, 0.0)
        current = current_metrics.get(metric, 0.0)
        changes.append(current - baseline)

    if not changes:
        return "unchanged"

    avg_change = sum(changes) / len(changes)

    if avg_change > threshold:
        return "improved"
    if avg_change < -threshold:
        return "degraded"
    return "unchanged"


def _build_query_index(result: EvaluationResult) -> dict[str, int]:
    """Build index mapping query text to position.

    Args:
        result: Evaluation result.

    Returns:
        Dict mapping query text to index.
    """
    index = {}
    for i, query in enumerate(result.testset.queries):
        # Use query text as key
        if query.text not in index:
            index[query.text] = i
    return index


def compute_diff(
    baseline: EvaluationResult,
    current: EvaluationResult,
    *,
    baseline_name: str = "baseline",
    current_name: str = "current",
    baseline_id: str | None = None,
    current_id: str | None = None,
    change_threshold: float = 0.01,
) -> DiffReport:
    """Compute diff between two evaluation results.

    Args:
        baseline: Baseline evaluation result.
        current: Current evaluation result to compare.
        baseline_name: Name for the baseline in reports.
        current_name: Name for the current run in reports.
        baseline_id: Optional BenchmarkRecord ID.
        current_id: Optional BenchmarkRecord ID.
        change_threshold: Minimum change to be considered significant (default 1%).

    Returns:
        DiffReport with per-query breakdown.

    Raises:
        ValueError: If query counts differ.
    """
    # Validate query counts match
    baseline_count = len(baseline.testset.queries)
    current_count = len(current.testset.queries)

    if baseline_count != current_count:
        raise ValueError(
            f"Query count mismatch: baseline has {baseline_count} queries, current has {current_count} queries"
        )

    # Build query text index for matching
    baseline_index = _build_query_index(baseline)

    # Compute per-query diffs
    improved: list[QueryDiff] = []
    degraded: list[QueryDiff] = []
    unchanged: list[QueryDiff] = []

    for i, query in enumerate(current.testset.queries):
        # Try to match by query text, fallback to index
        baseline_idx = baseline_index.get(query.text, i)

        baseline_metrics = _extract_query_metrics(baseline, baseline_idx)
        current_metrics = _extract_query_metrics(current, i)

        status = _classify_query(baseline_metrics, current_metrics, change_threshold)

        query_diff = QueryDiff(
            query_text=query.text,
            query_id=query.metadata.get("id") if query.metadata else None,
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            status=status,
        )

        if status == "improved":
            improved.append(query_diff)
        elif status == "degraded":
            degraded.append(query_diff)
        else:
            unchanged.append(query_diff)

    # Compute aggregate metrics
    baseline_summary = baseline.summary()
    current_summary = current.summary()

    metrics_diff = {}
    for metric in STANDARD_METRICS:
        baseline_val = baseline_summary.get(metric, 0.0)
        current_val = current_summary.get(metric, 0.0)
        metrics_diff[metric] = current_val - baseline_val

    # Get testset name
    testset_name = current.testset.name or baseline.testset.name or "unnamed"

    return DiffReport(
        baseline_id=baseline_id,
        current_id=current_id,
        baseline_name=baseline_name,
        current_name=current_name,
        timestamp=datetime.now(),
        testset_name=testset_name,
        metrics_diff=metrics_diff,
        improved=improved,
        degraded=degraded,
        unchanged=unchanged,
        change_threshold=change_threshold,
        baseline_metrics={m: baseline_summary.get(m, 0.0) for m in STANDARD_METRICS},
        current_metrics={m: current_summary.get(m, 0.0) for m in STANDARD_METRICS},
    )
