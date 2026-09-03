"""Tests for live A/B testing."""

from __future__ import annotations

import random

import pytest

from ragnarok_ai.ab.analyzer import ABAnalyzer, _two_proportion_p, _welch_p
from ragnarok_ai.ab.models import ABTestConfig, Experiment
from ragnarok_ai.monitor.models import TraceEvent


def make_variant_traces(
    experiment: str,
    variant: str,
    n: int,
    *,
    latency_mean: float = 200.0,
    latency_spread: float = 30.0,
    success_rate: float = 1.0,
    seed: int = 42,
) -> list[TraceEvent]:
    """Generate synthetic traces tagged for an experiment variant."""
    rng = random.Random(seed)
    return [
        TraceEvent(
            query_hash=f"{variant}{i}",
            query_length=50,
            total_latency_ms=max(1.0, rng.gauss(latency_mean, latency_spread)),
            success=rng.random() < success_rate,
            metadata={"experiment": experiment, "variant": variant},
        )
        for i in range(n)
    ]


class TestExperiment:
    """Test suite for Experiment assignment."""

    def test_assignment_is_deterministic(self) -> None:
        exp = Experiment(name="test")

        assert all(exp.assign("user-1") == exp.assign("user-1") for _ in range(20))

    def test_split_is_roughly_even(self) -> None:
        exp = Experiment(name="test", variants=["control", "candidate"])

        assigned = [exp.assign(f"user-{i}") for i in range(2000)]
        control_share = assigned.count("control") / len(assigned)

        assert 0.45 < control_share < 0.55

    def test_weights_shape_the_split(self) -> None:
        exp = Experiment(name="test", variants=["a", "b"], weights=[0.9, 0.1])

        assigned = [exp.assign(f"user-{i}") for i in range(2000)]

        assert assigned.count("a") / len(assigned) > 0.85

    def test_salt_reshuffles_assignments(self) -> None:
        base = Experiment(name="test")
        salted = Experiment(name="test", salt="v2")

        keys = [f"user-{i}" for i in range(500)]
        moved = sum(1 for k in keys if base.assign(k) != salted.assign(k))

        assert moved > 100  # a substantial fraction reassigned

    def test_validation_rejects_bad_configs(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            Experiment(name="test", variants=["only"])
        with pytest.raises(ValueError, match="unique"):
            Experiment(name="test", variants=["a", "a"])
        with pytest.raises(ValueError, match="same length"):
            Experiment(name="test", variants=["a", "b"], weights=[1.0])
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            Experiment(name="test", variants=["a", "b"], weights=[0.9, 0.9])

    def test_tag_records_assignment_in_trace_metadata(self) -> None:
        from ragnarok_ai.monitor.client import MonitorClient

        client = MonitorClient(sample_rate=1.0)
        exp = Experiment(name="reranker-test", variants=["control", "reranker"])

        # force=True samples the trace without needing a running daemon
        # (traces are buffered, nothing is sent below the buffer size)
        with client.trace("what is CHF?", force=True) as trace:
            variant = exp.tag(trace, "user-42")

        assert trace.metadata == {"experiment": "reranker-test", "variant": variant}
        assert variant in ("control", "reranker")


class TestStatisticalTests:
    """Test suite for the significance tests."""

    def test_identical_proportions_are_not_significant(self) -> None:
        assert _two_proportion_p(90, 100, 90, 100) == pytest.approx(1.0)

    def test_clear_proportion_gap_is_significant(self) -> None:
        assert _two_proportion_p(99, 100, 70, 100) < 0.001

    def test_degenerate_proportions_do_not_crash(self) -> None:
        assert _two_proportion_p(100, 100, 100, 100) == 1.0

    def test_identical_samples_are_not_significant(self) -> None:
        values = [float(i) for i in range(100)]

        assert _welch_p(values, values) == pytest.approx(1.0)

    def test_shifted_means_are_significant(self) -> None:
        rng = random.Random(7)
        a = [rng.gauss(200, 20) for _ in range(200)]
        b = [rng.gauss(300, 20) for _ in range(200)]

        assert _welch_p(a, b) < 0.001


class TestABAnalyzer:
    """Test suite for ABAnalyzer."""

    @pytest.fixture
    def exp(self) -> Experiment:
        return Experiment(name="reranker-test", variants=["control", "reranker"])

    def test_split_ignores_foreign_traces(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 5),
            *make_variant_traces("reranker-test", "reranker", 3),
            *make_variant_traces("other-experiment", "control", 4),
            TraceEvent(query_hash="untagged", query_length=1, total_latency_ms=1.0),
        ]

        groups = ABAnalyzer(exp).split(traces)

        assert len(groups["control"]) == 5
        assert len(groups["reranker"]) == 3

    def test_equivalent_variants_produce_no_winner(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 300, seed=1),
            *make_variant_traces("reranker-test", "reranker", 300, seed=2),
        ]

        report = ABAnalyzer(exp).analyze(traces)

        assert report.winner is None
        assert report.insufficient_data is False
        assert all(not v.significant for v in report.verdicts)

    def test_faster_variant_wins(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 300, latency_mean=300, seed=1),
            *make_variant_traces("reranker-test", "reranker", 300, latency_mean=200, seed=2),
        ]

        report = ABAnalyzer(exp).analyze(traces)

        assert report.winner == "reranker"
        latency = next(v for v in report.verdicts if v.metric == "latency_mean")
        assert latency.significant is True
        assert latency.winner == "reranker"
        assert latency.p_value < 0.001

    def test_more_reliable_variant_wins(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 400, success_rate=0.80, seed=1),
            *make_variant_traces("reranker-test", "reranker", 400, success_rate=0.97, seed=2),
        ]

        report = ABAnalyzer(exp).analyze(traces)

        assert report.winner == "reranker"
        success = next(v for v in report.verdicts if v.metric == "success_rate")
        assert success.significant is True
        assert success.winner == "reranker"

    def test_mixed_outcome_has_no_overall_winner(self, exp: Experiment) -> None:
        # reranker: much better success rate but much slower
        traces = [
            *make_variant_traces("reranker-test", "control", 400, latency_mean=150, success_rate=0.80, seed=1),
            *make_variant_traces("reranker-test", "reranker", 400, latency_mean=400, success_rate=0.97, seed=2),
        ]

        report = ABAnalyzer(exp).analyze(traces)

        assert report.winner is None
        assert all(v.significant for v in report.verdicts)  # both metrics differ...
        winners = {v.winner for v in report.verdicts}
        assert winners == {"control", "reranker"}  # ...in opposite directions

    def test_small_samples_are_flagged(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 10, latency_mean=100, seed=1),
            *make_variant_traces("reranker-test", "reranker", 10, latency_mean=900, seed=2),
        ]

        report = ABAnalyzer(exp).analyze(traces)

        assert report.insufficient_data is True
        assert report.verdicts == []
        assert report.winner is None

    def test_min_samples_is_configurable(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 30, latency_mean=100, seed=1),
            *make_variant_traces("reranker-test", "reranker", 30, latency_mean=900, seed=2),
        ]

        report = ABAnalyzer(exp, ABTestConfig(min_samples=20)).analyze(traces)

        assert report.insufficient_data is False
        assert report.winner == "control"

    def test_unknown_variant_raises(self, exp: Experiment) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            ABAnalyzer(exp).analyze([], variant_a="nope")

    def test_report_serializes(self, exp: Experiment) -> None:
        traces = [
            *make_variant_traces("reranker-test", "control", 300, latency_mean=300, seed=1),
            *make_variant_traces("reranker-test", "reranker", 300, latency_mean=200, seed=2),
        ]

        data = ABAnalyzer(exp).analyze(traces).to_dict()

        assert data["experiment"] == "reranker-test"
        assert data["winner"] == "reranker"
        assert set(data["stats"]) == {"control", "reranker"}
        assert {v["metric"] for v in data["verdicts"]} == {"success_rate", "latency_mean"}
