"""Production drift detection against a recorded baseline.

Typical workflow:

    >>> from ragnarok_ai.drift import DriftDetector, build_baseline
    >>> from ragnarok_ai.monitor.store import MonitorStore
    >>>
    >>> store = MonitorStore()
    >>> # 1. Record a baseline from a known-good period
    >>> baseline = build_baseline(store.get_traces(since=last_week, until=yesterday))
    >>> baseline.save("drift-baseline.json")
    >>>
    >>> # 2. Periodically compare recent traffic against it
    >>> detector = DriftDetector(DriftBaseline.load("drift-baseline.json"))
    >>> report = detector.detect(store.get_traces(since=one_hour_ago))
    >>> if report.has_drift:
    ...     await detector.alert(report, alert_manager)
"""

from __future__ import annotations

import math
from statistics import mean, quantiles
from typing import TYPE_CHECKING

from ragnarok_ai.drift.models import (
    DistributionSnapshot,
    DriftBaseline,
    DriftFinding,
    DriftReport,
    DriftThresholds,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragnarok_ai.alerts.manager import AlertManager
    from ragnarok_ai.alerts.protocols import AlertResult, AlertSeverity
    from ragnarok_ai.monitor.models import TraceEvent

# Numeric trace fields tracked for distribution drift
TRACKED_FIELDS: tuple[str, ...] = (
    "total_latency_ms",
    "query_length",
    "answer_length",
    "retrieval_count",
)

# Number of PSI buckets (decile-based, the common convention)
_N_BUCKETS = 10

# Floor for bucket proportions so the PSI log term is always defined
_PSI_EPSILON = 1e-4


def _field_values(traces: Sequence[TraceEvent], fld: str) -> list[float]:
    """Extract non-null numeric values of *fld* from traces."""
    return [float(v) for trace in traces if (v := getattr(trace, fld)) is not None]


def _percentile_95(values: Sequence[float]) -> float:
    """95th percentile with small-sample fallback."""
    ordered = sorted(values)
    if len(ordered) >= 20:
        return quantiles(ordered, n=100)[94]
    return ordered[-1]


def _snapshot(values: Sequence[float]) -> DistributionSnapshot:
    """Bucketize *values* into a decile-based distribution snapshot."""
    # Deduplicated decile edges: heavy ties (e.g. constant fields) may
    # collapse to fewer buckets, which PSI handles fine.
    edges = sorted(set(quantiles(values, n=_N_BUCKETS))) if len(values) >= _N_BUCKETS else sorted(set(values))[:-1]
    return DistributionSnapshot(
        count=len(values),
        mean=mean(values),
        bucket_edges=list(edges),
        bucket_proportions=_proportions(values, edges),
    )


def _proportions(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    """Proportion of *values* per bucket defined by *edges*."""
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[_bucket_index(v, edges)] += 1
    total = len(values)
    return [c / total for c in counts]


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    """Index of the bucket containing *value* (buckets are (a, b])."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def _psi(baseline_proportions: Sequence[float], current_proportions: Sequence[float]) -> float:
    """Population Stability Index between two bucket distributions."""
    total = 0.0
    for base_p, cur_p in zip(baseline_proportions, current_proportions, strict=True):
        b = max(base_p, _PSI_EPSILON)
        c = max(cur_p, _PSI_EPSILON)
        total += (c - b) * math.log(c / b)
    return total


def build_baseline(
    traces: Sequence[TraceEvent],
    *,
    fields: Sequence[str] = TRACKED_FIELDS,
) -> DriftBaseline:
    """Build a drift baseline from a reference window of traces.

    Args:
        traces: Traces from a known-good period (e.g. via
            ``MonitorStore.get_traces``). Must be non-empty.
        fields: Numeric trace fields to track for distribution drift.
            Fields with no values in the window are skipped.

    Returns:
        A self-contained, serializable ``DriftBaseline``.

    Raises:
        ValueError: If *traces* is empty.
    """
    if not traces:
        msg = "Cannot build a drift baseline from zero traces"
        raise ValueError(msg)

    latencies = _field_values(traces, "total_latency_ms")
    distributions = {}
    for fld in fields:
        values = _field_values(traces, fld)
        if values:
            distributions[fld] = _snapshot(values)

    timestamps = [trace.timestamp for trace in traces]
    return DriftBaseline(
        window_start=min(timestamps),
        window_end=max(timestamps),
        trace_count=len(traces),
        success_rate=sum(1 for t in traces if t.success) / len(traces),
        latency_p95=_percentile_95(latencies),
        distributions=distributions,
    )


class DriftDetector:
    """Detect drift between current production traffic and a baseline.

    Runs two families of checks:

    - **Distribution drift** (PSI) on every field the baseline tracks.
    - **Metric drift** on success rate (absolute drop) and p95 latency
      (relative increase).

    Attributes:
        baseline: The reference snapshot to compare against.
        thresholds: Detection thresholds.

    Example:
        >>> detector = DriftDetector(baseline)
        >>> report = detector.detect(store.get_traces(since=one_hour_ago))
        >>> report.has_drift
        False
    """

    def __init__(self, baseline: DriftBaseline, thresholds: DriftThresholds | None = None) -> None:
        """Initialize the detector.

        Args:
            baseline: Baseline built with ``build_baseline`` or loaded
                from a file.
            thresholds: Optional custom thresholds.
        """
        self.baseline = baseline
        self.thresholds = thresholds or DriftThresholds()

    def detect(self, traces: Sequence[TraceEvent]) -> DriftReport:
        """Compare a window of traces against the baseline.

        Args:
            traces: The current window (e.g. the last hour of traffic).

        Returns:
            A ``DriftReport``. When the window holds fewer than
            ``thresholds.min_samples`` traces, the report is flagged
            ``insufficient_data`` and carries no findings.
        """
        current_summary = self._summarize(traces)
        baseline_summary = {
            "trace_count": float(self.baseline.trace_count),
            "success_rate": self.baseline.success_rate,
            "latency_p95": self.baseline.latency_p95,
        }
        timestamps = [trace.timestamp for trace in traces]
        window_start = min(timestamps) if timestamps else None
        window_end = max(timestamps) if timestamps else None

        if len(traces) < self.thresholds.min_samples:
            return DriftReport(
                has_drift=False,
                findings=[],
                insufficient_data=True,
                baseline_summary=baseline_summary,
                current_summary=current_summary,
                window_start=window_start,
                window_end=window_end,
            )

        findings = [
            *self._check_metrics(traces),
            *self._check_distributions(traces),
        ]
        return DriftReport(
            has_drift=bool(findings),
            findings=findings,
            insufficient_data=False,
            baseline_summary=baseline_summary,
            current_summary=current_summary,
            window_start=window_start,
            window_end=window_end,
        )

    async def alert(self, report: DriftReport, manager: AlertManager) -> list[AlertResult]:
        """Dispatch a report's findings through an ``AlertManager``.

        Args:
            report: A report produced by ``detect``.
            manager: The alert manager with configured adapters.

        Returns:
            Flat list of per-adapter send results (empty if no findings).
        """
        results: list[AlertResult] = []
        for alert in report.to_alerts():
            results.extend(await manager.send(alert))
        return results

    async def check_and_alert(self, traces: Sequence[TraceEvent], manager: AlertManager) -> DriftReport:
        """Detect drift and dispatch any findings in one call."""
        report = self.detect(traces)
        if report.has_drift:
            await self.alert(report, manager)
        return report

    # ── Checks ───────────────────────────────────────────────────────────

    def _check_metrics(self, traces: Sequence[TraceEvent]) -> list[DriftFinding]:
        """Threshold checks on scalar health metrics."""
        findings: list[DriftFinding] = []
        t = self.thresholds

        # Success rate: absolute drop in percentage points
        current_rate = sum(1 for trace in traces if trace.success) / len(traces)
        drop = self.baseline.success_rate - current_rate
        if drop >= t.success_rate_drop:
            findings.append(
                DriftFinding(
                    kind="metric",
                    metric="success_rate",
                    baseline_value=self.baseline.success_rate,
                    current_value=current_rate,
                    score=drop,
                    threshold=t.success_rate_drop,
                    severity=self._metric_severity(drop, t.success_rate_drop),
                    message=(
                        f"Success rate dropped from {self.baseline.success_rate:.1%} "
                        f"to {current_rate:.1%} (-{drop:.1%})"
                    ),
                )
            )

        # p95 latency: relative increase
        latencies = _field_values(traces, "total_latency_ms")
        if latencies and self.baseline.latency_p95 > 0:
            current_p95 = _percentile_95(latencies)
            increase = (current_p95 - self.baseline.latency_p95) / self.baseline.latency_p95
            if increase >= t.latency_increase:
                findings.append(
                    DriftFinding(
                        kind="metric",
                        metric="latency_p95",
                        baseline_value=self.baseline.latency_p95,
                        current_value=current_p95,
                        score=increase,
                        threshold=t.latency_increase,
                        severity=self._metric_severity(increase, t.latency_increase),
                        message=(
                            f"p95 latency increased from {self.baseline.latency_p95:.0f}ms "
                            f"to {current_p95:.0f}ms (+{increase:.0%})"
                        ),
                    )
                )

        return findings

    def _check_distributions(self, traces: Sequence[TraceEvent]) -> list[DriftFinding]:
        """PSI checks on every distribution the baseline tracks."""
        findings: list[DriftFinding] = []
        t = self.thresholds

        for fld, snapshot in self.baseline.distributions.items():
            values = _field_values(traces, fld)
            if not values:
                continue
            psi = _psi(snapshot.bucket_proportions, _proportions(values, snapshot.bucket_edges))
            if psi < t.psi_warning:
                continue
            severity = self._psi_severity(psi)
            current_mean = mean(values)
            findings.append(
                DriftFinding(
                    kind="distribution",
                    metric=fld,
                    baseline_value=snapshot.mean,
                    current_value=current_mean,
                    score=psi,
                    threshold=t.psi_warning,
                    severity=severity,
                    message=(
                        f"Distribution of {fld} shifted (PSI={psi:.3f}, mean {snapshot.mean:.1f} → {current_mean:.1f})"
                    ),
                )
            )

        return findings

    # ── Helpers ──────────────────────────────────────────────────────────

    def _metric_severity(self, change: float, threshold: float) -> AlertSeverity:
        from ragnarok_ai.alerts.protocols import AlertSeverity

        if change >= threshold * self.thresholds.critical_multiplier:
            return AlertSeverity.CRITICAL
        return AlertSeverity.WARNING

    def _psi_severity(self, psi: float) -> AlertSeverity:
        from ragnarok_ai.alerts.protocols import AlertSeverity

        if psi >= self.thresholds.psi_critical:
            return AlertSeverity.CRITICAL
        return AlertSeverity.WARNING

    @staticmethod
    def _summarize(traces: Sequence[TraceEvent]) -> dict[str, float]:
        """Scalar summary of a trace window."""
        if not traces:
            return {"trace_count": 0.0}
        latencies = _field_values(traces, "total_latency_ms")
        summary = {
            "trace_count": float(len(traces)),
            "success_rate": sum(1 for t in traces if t.success) / len(traces),
        }
        if latencies:
            summary["latency_p95"] = _percentile_95(latencies)
        return summary
