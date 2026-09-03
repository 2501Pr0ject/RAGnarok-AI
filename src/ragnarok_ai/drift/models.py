"""Data models for production drift detection.

Drift detection compares a window of production traces against a recorded
baseline. Two families of checks are supported:

- **Distribution drift** via the Population Stability Index (PSI) over
  bucketized numeric fields (latency, query length, answer length...).
- **Metric drift** via threshold checks on scalar health metrics
  (success rate, p95 latency).

The baseline is a self-contained, JSON-serializable snapshot: it stores
bucket edges and proportions, not raw traces, so it can be committed to a
repo or shipped alongside a deployment without leaking production data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ragnarok_ai.alerts.protocols import Alert, AlertSeverity


class DistributionSnapshot(BaseModel):
    """Bucketized summary of one numeric field in the baseline window.

    Attributes:
        count: Number of non-null observations.
        mean: Mean of the observations.
        bucket_edges: Interior bucket boundaries (len = n_buckets - 1).
            Buckets are (-inf, e1], (e1, e2], ..., (e_last, +inf).
        bucket_proportions: Proportion of baseline observations per bucket
            (len = n_buckets, sums to ~1.0).
    """

    count: int
    mean: float
    bucket_edges: list[float]
    bucket_proportions: list[float]

    model_config = {"frozen": True}


class DriftBaseline(BaseModel):
    """Reference snapshot of production behavior over a time window.

    Built with ``build_baseline()`` from monitor traces, then persisted
    with ``save()`` and reloaded with ``load()``.

    Attributes:
        created_at: When the baseline was built.
        window_start: Start of the reference window.
        window_end: End of the reference window.
        trace_count: Number of traces in the window.
        success_rate: Fraction of successful requests (0.0-1.0).
        latency_p95: 95th percentile of total latency (ms).
        distributions: Per-field bucketized distributions for PSI.
    """

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    window_start: datetime
    window_end: datetime
    trace_count: int
    success_rate: float
    latency_p95: float
    distributions: dict[str, DistributionSnapshot]

    model_config = {"frozen": True}

    def save(self, path: str | Path) -> None:
        """Write the baseline to a JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DriftBaseline:
        """Read a baseline from a JSON file."""
        return cls.model_validate_json(Path(path).read_text())


@dataclass
class DriftThresholds:
    """Thresholds for drift detection.

    PSI interpretation follows the common industry convention:
    below 0.1 no significant change, 0.1-0.25 moderate shift,
    above 0.25 major shift.

    Attributes:
        psi_warning: PSI above this is a warning-level distribution shift.
        psi_critical: PSI above this is a critical distribution shift.
        success_rate_drop: Absolute drop in success rate that warns
            (e.g. 0.05 = five percentage points).
        latency_increase: Relative p95 latency increase that warns
            (e.g. 0.25 = +25%).
        critical_multiplier: Metric changes beyond threshold x multiplier
            are critical instead of warning.
        min_samples: Minimum traces in the current window for a
            statistically meaningful comparison; below it, detection is
            skipped and the report is flagged ``insufficient_data``.
    """

    psi_warning: float = 0.1
    psi_critical: float = 0.25
    success_rate_drop: float = 0.05
    latency_increase: float = 0.25
    critical_multiplier: float = 2.0
    min_samples: int = 100


@dataclass(frozen=True)
class DriftFinding:
    """One detected drift signal.

    Attributes:
        kind: "distribution" (PSI) or "metric" (threshold check).
        metric: Field or metric name (e.g. "total_latency_ms").
        baseline_value: Reference value (distribution mean for PSI).
        current_value: Current-window value.
        score: PSI value for distribution findings; relative or absolute
            change for metric findings.
        threshold: The threshold the score exceeded.
        severity: warning or critical.
        message: Human-readable description.
    """

    kind: Literal["distribution", "metric"]
    metric: str
    baseline_value: float
    current_value: float
    score: float
    threshold: float
    severity: AlertSeverity
    message: str


@dataclass
class DriftReport:
    """Result of comparing a current window against a baseline.

    Attributes:
        has_drift: True if at least one finding was produced.
        findings: Detected drift signals, empty when stable.
        insufficient_data: True when the current window had fewer than
            ``min_samples`` traces and detection was skipped.
        baseline_summary: Scalar summary of the baseline window.
        current_summary: Scalar summary of the current window.
        window_start: Start of the analyzed window.
        window_end: End of the analyzed window.
        timestamp: When the report was produced.
    """

    has_drift: bool
    findings: list[DriftFinding]
    insufficient_data: bool
    baseline_summary: dict[str, float]
    current_summary: dict[str, float]
    window_start: datetime | None = None
    window_end: datetime | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_alerts(self) -> list[Alert]:
        """Convert findings into alerts (source="drift") for dispatch."""
        return [
            Alert(
                title=f"Drift detected: {finding.metric}",
                message=finding.message,
                severity=finding.severity,
                source="drift",
                metadata={
                    "kind": finding.kind,
                    "metric": finding.metric,
                    "baseline_value": finding.baseline_value,
                    "current_value": finding.current_value,
                    "score": finding.score,
                    "threshold": finding.threshold,
                },
            )
            for finding in self.findings
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "has_drift": self.has_drift,
            "insufficient_data": self.insufficient_data,
            "timestamp": self.timestamp.isoformat(),
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "baseline_summary": self.baseline_summary,
            "current_summary": self.current_summary,
            "findings": [
                {
                    "kind": f.kind,
                    "metric": f.metric,
                    "baseline_value": f.baseline_value,
                    "current_value": f.current_value,
                    "score": f.score,
                    "threshold": f.threshold,
                    "severity": f.severity.value,
                    "message": f.message,
                }
                for f in self.findings
            ],
        }
