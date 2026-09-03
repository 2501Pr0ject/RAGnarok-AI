"""Tests for production drift detection."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from ragnarok_ai.alerts.manager import AlertManager
from ragnarok_ai.alerts.protocols import Alert, AlertResult, AlertSeverity
from ragnarok_ai.drift.detector import DriftDetector, _psi, build_baseline
from ragnarok_ai.drift.models import DriftBaseline, DriftThresholds
from ragnarok_ai.monitor.models import TraceEvent


def make_traces(
    n: int,
    *,
    latency_mean: float = 200.0,
    latency_spread: float = 40.0,
    query_length: int = 50,
    success_rate: float = 1.0,
    seed: int = 42,
    start: datetime | None = None,
) -> list[TraceEvent]:
    """Generate synthetic traces with a controllable distribution."""
    rng = random.Random(seed)
    start = start or datetime(2026, 9, 1, tzinfo=timezone.utc)
    traces = []
    for i in range(n):
        traces.append(
            TraceEvent(
                query_hash=f"hash{i}",
                query_length=query_length + rng.randint(-10, 10),
                total_latency_ms=max(1.0, rng.gauss(latency_mean, latency_spread)),
                retrieval_count=5,
                answer_length=200 + rng.randint(-50, 50),
                success=rng.random() < success_rate,
                timestamp=start + timedelta(seconds=i),
            )
        )
    return traces


class TestBuildBaseline:
    """Test suite for build_baseline."""

    def test_baseline_captures_window_stats(self) -> None:
        traces = make_traces(500)
        baseline = build_baseline(traces)

        assert baseline.trace_count == 500
        assert baseline.success_rate == 1.0
        assert 100 < baseline.latency_p95 < 400
        assert baseline.window_start == traces[0].timestamp
        assert baseline.window_end == traces[-1].timestamp

    def test_baseline_tracks_distributions(self) -> None:
        baseline = build_baseline(make_traces(500))

        assert set(baseline.distributions) == {
            "total_latency_ms",
            "query_length",
            "answer_length",
            "retrieval_count",
        }
        latency = baseline.distributions["total_latency_ms"]
        assert latency.count == 500
        assert len(latency.bucket_proportions) == len(latency.bucket_edges) + 1
        assert sum(latency.bucket_proportions) == pytest.approx(1.0)

    def test_empty_traces_raise(self) -> None:
        with pytest.raises(ValueError, match="zero traces"):
            build_baseline([])

    def test_roundtrip_serialization(self, tmp_path) -> None:
        baseline = build_baseline(make_traces(200))
        path = tmp_path / "baseline.json"

        baseline.save(path)
        loaded = DriftBaseline.load(path)

        assert loaded == baseline


class TestPsi:
    """Test suite for the PSI computation."""

    def test_identical_distributions_score_zero(self) -> None:
        proportions = [0.1] * 10

        assert _psi(proportions, proportions) == pytest.approx(0.0)

    def test_shifted_distribution_scores_high(self) -> None:
        baseline = [0.5, 0.5, 0.0, 0.0]
        current = [0.0, 0.0, 0.5, 0.5]

        assert _psi(baseline, current) > 0.25

    def test_empty_bucket_does_not_crash(self) -> None:
        assert _psi([1.0, 0.0], [0.0, 1.0]) > 0


class TestDriftDetector:
    """Test suite for DriftDetector.detect."""

    def test_stable_traffic_reports_no_drift(self) -> None:
        baseline = build_baseline(make_traces(1000, seed=1))
        detector = DriftDetector(baseline)

        report = detector.detect(make_traces(500, seed=2))

        assert report.has_drift is False
        assert report.findings == []
        assert report.insufficient_data is False

    def test_latency_distribution_shift_is_detected(self) -> None:
        baseline = build_baseline(make_traces(1000, latency_mean=200, seed=1))
        detector = DriftDetector(baseline)

        report = detector.detect(make_traces(500, latency_mean=600, seed=2))

        assert report.has_drift is True
        metrics = {f.metric for f in report.findings}
        assert "total_latency_ms" in metrics  # PSI
        assert "latency_p95" in metrics  # threshold
        psi_finding = next(f for f in report.findings if f.metric == "total_latency_ms")
        assert psi_finding.kind == "distribution"
        assert psi_finding.severity == AlertSeverity.CRITICAL
        assert psi_finding.score > 0.25

    def test_query_length_shift_is_detected(self) -> None:
        baseline = build_baseline(make_traces(1000, query_length=50, seed=1))
        detector = DriftDetector(baseline)

        report = detector.detect(make_traces(500, query_length=300, seed=2))

        assert any(f.metric == "query_length" and f.kind == "distribution" for f in report.findings)

    def test_success_rate_drop_is_detected(self) -> None:
        baseline = build_baseline(make_traces(1000, seed=1))
        detector = DriftDetector(baseline)

        report = detector.detect(make_traces(500, success_rate=0.8, seed=2))

        finding = next(f for f in report.findings if f.metric == "success_rate")
        assert finding.kind == "metric"
        assert finding.severity == AlertSeverity.CRITICAL  # ~20pt drop >= 5pt x 2.0
        assert finding.baseline_value == 1.0
        assert finding.current_value < 0.9

    def test_moderate_drop_is_warning(self) -> None:
        baseline = build_baseline(make_traces(1000, seed=1))
        detector = DriftDetector(
            baseline,
            DriftThresholds(success_rate_drop=0.05, critical_multiplier=4.0),
        )

        report = detector.detect(make_traces(500, success_rate=0.93, seed=2))

        finding = next(f for f in report.findings if f.metric == "success_rate")
        assert finding.severity == AlertSeverity.WARNING

    def test_small_window_is_flagged_insufficient(self) -> None:
        baseline = build_baseline(make_traces(1000, seed=1))
        detector = DriftDetector(baseline)

        # Wildly drifted but tiny window: no findings, explicit flag
        report = detector.detect(make_traces(10, latency_mean=5000, seed=2))

        assert report.insufficient_data is True
        assert report.has_drift is False
        assert report.findings == []

    def test_min_samples_is_configurable(self) -> None:
        baseline = build_baseline(make_traces(1000, latency_mean=200, seed=1))
        detector = DriftDetector(baseline, DriftThresholds(min_samples=10))

        report = detector.detect(make_traces(50, latency_mean=600, seed=2))

        assert report.insufficient_data is False
        assert report.has_drift is True

    def test_report_serializes(self) -> None:
        baseline = build_baseline(make_traces(1000, seed=1))
        report = DriftDetector(baseline).detect(make_traces(500, latency_mean=600, seed=2))

        data = report.to_dict()

        assert data["has_drift"] is True
        assert data["findings"]
        assert all(f["severity"] in {"warning", "critical"} for f in data["findings"])


class TestAlerting:
    """Test suite for the alerting integration."""

    def test_findings_convert_to_drift_alerts(self) -> None:
        baseline = build_baseline(make_traces(1000, seed=1))
        report = DriftDetector(baseline).detect(make_traces(500, latency_mean=600, seed=2))

        alerts = report.to_alerts()

        assert len(alerts) == len(report.findings)
        assert all(a.source == "drift" for a in alerts)
        assert all("metric" in a.metadata for a in alerts)

    @pytest.mark.asyncio
    async def test_check_and_alert_dispatches(self) -> None:
        class RecordingAdapter:
            def __init__(self) -> None:
                self.sent: list[Alert] = []

            @property
            def name(self) -> str:
                return "recording"

            async def send(self, alert: Alert) -> AlertResult:
                self.sent.append(alert)
                return AlertResult(success=True, adapter=self.name)

        adapter = RecordingAdapter()
        manager = AlertManager()
        manager.add_adapter(adapter)

        baseline = build_baseline(make_traces(1000, seed=1))
        detector = DriftDetector(baseline)

        report = await detector.check_and_alert(make_traces(500, latency_mean=600, seed=2), manager)

        assert report.has_drift is True
        assert len(adapter.sent) == len(report.findings)

    @pytest.mark.asyncio
    async def test_stable_traffic_sends_nothing(self) -> None:
        class RecordingAdapter:
            def __init__(self) -> None:
                self.sent: list[Alert] = []

            @property
            def name(self) -> str:
                return "recording"

            async def send(self, alert: Alert) -> AlertResult:
                self.sent.append(alert)
                return AlertResult(success=True, adapter=self.name)

        adapter = RecordingAdapter()
        manager = AlertManager()
        manager.add_adapter(adapter)

        baseline = build_baseline(make_traces(1000, seed=1))
        report = await DriftDetector(baseline).check_and_alert(make_traces(500, seed=2), manager)

        assert report.has_drift is False
        assert adapter.sent == []
