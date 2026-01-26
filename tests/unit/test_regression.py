"""Unit tests for the regression detection module."""

from __future__ import annotations

from datetime import datetime

import pytest

from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
from ragnarok_ai.regression import (
    LOWER_IS_BETTER,
    RegressionAlert,
    RegressionDetector,
    RegressionResult,
    RegressionThresholds,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_testset() -> TestSet:
    """Create a sample testset for testing."""
    return TestSet(
        queries=[
            Query(text="What is RAG?", ground_truth_docs=["doc1", "doc2"]),
            Query(text="How does it work?", ground_truth_docs=["doc3"]),
        ],
        name="test_set",
    )


@pytest.fixture
def baseline_evaluation(sample_testset: TestSet) -> EvaluationResult:
    """Create a baseline evaluation result."""
    return EvaluationResult(
        testset=sample_testset,
        metrics=[
            RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
            RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
        ],
        responses=["Answer 1", "Answer 2"],
        query_results=[
            QueryResult(
                query=sample_testset.queries[0],
                metric=RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
                answer="Answer 1",
                latency_ms=100.0,
            ),
            QueryResult(
                query=sample_testset.queries[1],
                metric=RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
                answer="Answer 2",
                latency_ms=100.0,
            ),
        ],
        total_latency_ms=200.0,
    )


def create_evaluation(
    sample_testset: TestSet,
    precision: float = 0.80,
    recall: float = 0.70,
    mrr: float = 0.90,
    ndcg: float = 0.85,
    latency_ms: float = 100.0,
) -> EvaluationResult:
    """Helper to create evaluation results with specific metrics."""
    metrics = RetrievalMetrics(precision=precision, recall=recall, mrr=mrr, ndcg=ndcg, k=10)
    return EvaluationResult(
        testset=sample_testset,
        metrics=[metrics, metrics],
        responses=["Answer 1", "Answer 2"],
        query_results=[
            QueryResult(
                query=sample_testset.queries[0],
                metric=metrics,
                answer="Answer 1",
                latency_ms=latency_ms,
            ),
            QueryResult(
                query=sample_testset.queries[1],
                metric=metrics,
                answer="Answer 2",
                latency_ms=latency_ms,
            ),
        ],
        total_latency_ms=latency_ms * 2,
    )


# ============================================================================
# RegressionThresholds Tests
# ============================================================================


class TestRegressionThresholds:
    """Tests for RegressionThresholds dataclass."""

    def test_default_values(self) -> None:
        """Default thresholds are set correctly."""
        thresholds = RegressionThresholds()

        assert thresholds.precision_drop == 0.05
        assert thresholds.recall_drop == 0.05
        assert thresholds.mrr_drop == 0.05
        assert thresholds.ndcg_drop == 0.05
        assert thresholds.latency_increase == 0.20
        assert thresholds.critical_multiplier == 2.0

    def test_custom_values(self) -> None:
        """Custom thresholds can be set."""
        thresholds = RegressionThresholds(
            precision_drop=0.03,
            recall_drop=0.10,
            latency_increase=0.15,
            critical_multiplier=3.0,
        )

        assert thresholds.precision_drop == 0.03
        assert thresholds.recall_drop == 0.10
        assert thresholds.latency_increase == 0.15
        assert thresholds.critical_multiplier == 3.0


# ============================================================================
# RegressionAlert Tests
# ============================================================================


class TestRegressionAlert:
    """Tests for RegressionAlert dataclass."""

    def test_message_quality_metric(self) -> None:
        """Message for quality metric shows 'dropped'."""
        alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.78,
            change_percent=-8.24,
            threshold_percent=5.0,
            severity="warning",
        )

        assert "precision" in alert.message
        assert "dropped" in alert.message
        assert "8.2%" in alert.message
        assert "5.0%" in alert.message

    def test_message_latency_metric(self) -> None:
        """Message for latency metric shows 'increased'."""
        alert = RegressionAlert(
            metric="latency_ms",
            baseline_value=100.0,
            current_value=130.0,
            change_percent=30.0,
            threshold_percent=20.0,
            severity="warning",
        )

        assert "latency_ms" in alert.message
        assert "increased" in alert.message
        assert "30.0%" in alert.message
        assert "20.0%" in alert.message

    def test_severity_values(self) -> None:
        """Severity can be warning or critical."""
        warning_alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.78,
            change_percent=-8.24,
            threshold_percent=5.0,
            severity="warning",
        )
        critical_alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.70,
            change_percent=-17.6,
            threshold_percent=5.0,
            severity="critical",
        )

        assert warning_alert.severity == "warning"
        assert critical_alert.severity == "critical"


# ============================================================================
# RegressionResult Tests
# ============================================================================


class TestRegressionResult:
    """Tests for RegressionResult dataclass."""

    def test_has_regressions_empty(self) -> None:
        """has_regressions is False when no alerts."""
        result = RegressionResult(
            alerts=[],
            baseline_summary={"precision": 0.85},
            current_summary={"precision": 0.85},
        )

        assert not result.has_regressions

    def test_has_regressions_with_alerts(self) -> None:
        """has_regressions is True when alerts exist."""
        alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.78,
            change_percent=-8.24,
            threshold_percent=5.0,
            severity="warning",
        )
        result = RegressionResult(
            alerts=[alert],
            baseline_summary={"precision": 0.85},
            current_summary={"precision": 0.78},
        )

        assert result.has_regressions

    def test_has_critical_false(self) -> None:
        """has_critical is False when only warnings."""
        alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.78,
            change_percent=-8.24,
            threshold_percent=5.0,
            severity="warning",
        )
        result = RegressionResult(
            alerts=[alert],
            baseline_summary={"precision": 0.85},
            current_summary={"precision": 0.78},
        )

        assert not result.has_critical

    def test_has_critical_true(self) -> None:
        """has_critical is True when critical alert exists."""
        alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.70,
            change_percent=-17.6,
            threshold_percent=5.0,
            severity="critical",
        )
        result = RegressionResult(
            alerts=[alert],
            baseline_summary={"precision": 0.85},
            current_summary={"precision": 0.70},
        )

        assert result.has_critical

    def test_warning_and_critical_count(self) -> None:
        """warning_count and critical_count are correct."""
        alerts = [
            RegressionAlert(
                metric="precision",
                baseline_value=0.85,
                current_value=0.78,
                change_percent=-8.24,
                threshold_percent=5.0,
                severity="warning",
            ),
            RegressionAlert(
                metric="recall",
                baseline_value=0.80,
                current_value=0.70,
                change_percent=-12.5,
                threshold_percent=5.0,
                severity="critical",
            ),
            RegressionAlert(
                metric="ndcg",
                baseline_value=0.90,
                current_value=0.80,
                change_percent=-11.1,
                threshold_percent=5.0,
                severity="warning",
            ),
        ]
        result = RegressionResult(
            alerts=alerts,
            baseline_summary={},
            current_summary={},
        )

        assert result.warning_count == 2
        assert result.critical_count == 1

    def test_summary_no_regressions(self) -> None:
        """summary() shows no regressions message."""
        result = RegressionResult(
            alerts=[],
            baseline_summary={"precision": 0.85},
            current_summary={"precision": 0.85},
        )

        assert "No regressions detected" in result.summary()

    def test_summary_with_regressions(self) -> None:
        """summary() shows alerts properly formatted."""
        alert = RegressionAlert(
            metric="precision",
            baseline_value=0.85,
            current_value=0.78,
            change_percent=-8.24,
            threshold_percent=5.0,
            severity="warning",
        )
        result = RegressionResult(
            alerts=[alert],
            baseline_summary={"precision": 0.85},
            current_summary={"precision": 0.78},
        )

        summary = result.summary()
        assert "Regression Detection Summary" in summary
        assert "[WARNING]" in summary
        assert "precision" in summary

    def test_timestamp_set_automatically(self) -> None:
        """timestamp is set to current time on creation."""
        before = datetime.now()
        result = RegressionResult(
            alerts=[],
            baseline_summary={},
            current_summary={},
        )
        after = datetime.now()

        assert before <= result.timestamp <= after


# ============================================================================
# RegressionDetector Tests
# ============================================================================


class TestRegressionDetector:
    """Tests for RegressionDetector class."""

    def test_detect_no_regression(self, sample_testset: TestSet, baseline_evaluation: EvaluationResult) -> None:
        """No regression when metrics are the same."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # Same metrics as baseline
        current = create_evaluation(sample_testset)
        result = detector.detect(current)

        assert not result.has_regressions
        assert len(result.alerts) == 0

    def test_detect_precision_regression_warning(
        self, sample_testset: TestSet, baseline_evaluation: EvaluationResult
    ) -> None:
        """Detect precision drop as warning."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # 7.5% precision drop (exceeds 5% threshold but not 10% critical)
        # Baseline precision is 0.80, so 0.74 = 7.5% drop
        current = create_evaluation(sample_testset, precision=0.74)
        result = detector.detect(current)

        assert result.has_regressions
        precision_alerts = [a for a in result.alerts if a.metric == "precision"]
        assert len(precision_alerts) == 1
        assert precision_alerts[0].severity == "warning"

    def test_detect_precision_regression_critical(
        self, sample_testset: TestSet, baseline_evaluation: EvaluationResult
    ) -> None:
        """Detect severe precision drop as critical."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # 15% precision drop (exceeds 10% critical threshold = 5% * 2)
        current = create_evaluation(sample_testset, precision=0.68)
        result = detector.detect(current)

        assert result.has_critical
        precision_alerts = [a for a in result.alerts if a.metric == "precision"]
        assert len(precision_alerts) == 1
        assert precision_alerts[0].severity == "critical"

    def test_detect_recall_regression(self, sample_testset: TestSet, baseline_evaluation: EvaluationResult) -> None:
        """Detect recall drop."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # 15% recall drop
        current = create_evaluation(sample_testset, recall=0.595)
        result = detector.detect(current)

        recall_alerts = [a for a in result.alerts if a.metric == "recall"]
        assert len(recall_alerts) == 1

    def test_detect_latency_regression(self, sample_testset: TestSet, baseline_evaluation: EvaluationResult) -> None:
        """Detect latency increase."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # 30% latency increase (exceeds 20% threshold)
        current = create_evaluation(sample_testset, latency_ms=130.0)
        result = detector.detect(current)

        latency_alerts = [a for a in result.alerts if a.metric == "latency_ms"]
        assert len(latency_alerts) == 1
        assert latency_alerts[0].change_percent > 0  # Positive = increase

    def test_detect_multiple_regressions(self, sample_testset: TestSet, baseline_evaluation: EvaluationResult) -> None:
        """Detect multiple regressions at once."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # Multiple regressions
        current = create_evaluation(
            sample_testset,
            precision=0.70,  # -12.5%
            recall=0.60,  # -14.3%
            latency_ms=150.0,  # +50%
        )
        result = detector.detect(current)

        assert len(result.alerts) >= 3

    def test_incompatible_query_count_error(self, baseline_evaluation: EvaluationResult) -> None:
        """Raise ValueError when query counts differ."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # Create evaluation with different query count
        different_testset = TestSet(
            queries=[Query(text="Single query", ground_truth_docs=["doc1"])],
            name="different",
        )
        current = EvaluationResult(
            testset=different_testset,
            metrics=[RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10)],
            responses=["Answer"],
        )

        with pytest.raises(ValueError, match="Incompatible evaluations"):
            detector.detect(current)

    def test_custom_thresholds(self, sample_testset: TestSet, baseline_evaluation: EvaluationResult) -> None:
        """Custom thresholds are used."""
        # Stricter threshold (3%)
        detector = RegressionDetector(
            baseline=baseline_evaluation,
            thresholds=RegressionThresholds(precision_drop=0.03),
        )

        # 5% drop - would be ok with default, but exceeds 3%
        current = create_evaluation(sample_testset, precision=0.76)
        result = detector.detect(current)

        precision_alerts = [a for a in result.alerts if a.metric == "precision"]
        assert len(precision_alerts) == 1

    def test_zero_baseline_skipped(self, sample_testset: TestSet) -> None:
        """Metrics with zero baseline are skipped."""
        # Create baseline with zero precision
        baseline = EvaluationResult(
            testset=sample_testset,
            metrics=[
                RetrievalMetrics(precision=0.0, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
                RetrievalMetrics(precision=0.0, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
            ],
            responses=["Answer 1", "Answer 2"],
        )
        detector = RegressionDetector(baseline=baseline)

        current = create_evaluation(sample_testset, precision=0.50)
        result = detector.detect(current)

        # Should not crash or have precision alert
        precision_alerts = [a for a in result.alerts if a.metric == "precision"]
        assert len(precision_alerts) == 0

    def test_latency_from_query_results(self, baseline_evaluation: EvaluationResult) -> None:
        """Latency is computed from query_results."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # Baseline has 100ms latency from query_results
        assert detector._baseline_summary.get("latency_ms") == 100.0

    def test_latency_from_total_latency_ms(self, sample_testset: TestSet) -> None:
        """Latency is computed from total_latency_ms when no query_results."""
        baseline = EvaluationResult(
            testset=sample_testset,
            metrics=[
                RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
                RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10),
            ],
            responses=["Answer 1", "Answer 2"],
            total_latency_ms=200.0,
            query_results=[],  # Empty query_results
        )
        detector = RegressionDetector(baseline=baseline)

        # Should compute from total_latency_ms / num_metrics
        assert detector._baseline_summary.get("latency_ms") == 100.0

    def test_improvement_not_flagged(self, sample_testset: TestSet, baseline_evaluation: EvaluationResult) -> None:
        """Improvements are not flagged as regressions."""
        detector = RegressionDetector(baseline=baseline_evaluation)

        # Better metrics
        current = create_evaluation(
            sample_testset,
            precision=0.90,  # +12.5%
            recall=0.80,  # +14.3%
            latency_ms=80.0,  # -20% (faster)
        )
        result = detector.detect(current)

        assert not result.has_regressions


# ============================================================================
# LOWER_IS_BETTER Tests
# ============================================================================


class TestLowerIsBetter:
    """Tests for LOWER_IS_BETTER constant."""

    def test_latency_metrics_included(self) -> None:
        """Latency metrics are in LOWER_IS_BETTER."""
        assert "latency_ms" in LOWER_IS_BETTER
        assert "avg_latency_ms" in LOWER_IS_BETTER
        assert "total_latency_ms" in LOWER_IS_BETTER

    def test_quality_metrics_not_included(self) -> None:
        """Quality metrics are NOT in LOWER_IS_BETTER."""
        assert "precision" not in LOWER_IS_BETTER
        assert "recall" not in LOWER_IS_BETTER
        assert "mrr" not in LOWER_IS_BETTER
        assert "ndcg" not in LOWER_IS_BETTER
