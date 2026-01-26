"""Unit tests for the diff module."""

from __future__ import annotations

from datetime import datetime

import pytest

from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.diff import DiffReport, QueryDiff, compute_diff
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_testset() -> TestSet:
    """Create a sample testset for testing."""
    return TestSet(
        queries=[
            Query(text="What is authentication?", ground_truth_docs=["doc1"]),
            Query(text="How to configure rate limits?", ground_truth_docs=["doc2"]),
            Query(text="Pricing information?", ground_truth_docs=["doc3"]),
        ],
        name="test-queries",
    )


@pytest.fixture
def baseline_metrics() -> list[RetrievalMetrics]:
    """Baseline metrics for 3 queries."""
    return [
        RetrievalMetrics(precision=0.8, recall=1.0, mrr=0.9, ndcg=0.85, k=10),
        RetrievalMetrics(precision=0.6, recall=0.8, mrr=0.7, ndcg=0.75, k=10),
        RetrievalMetrics(precision=0.5, recall=0.6, mrr=0.5, ndcg=0.55, k=10),
    ]


@pytest.fixture
def improved_metrics() -> list[RetrievalMetrics]:
    """Improved metrics for 3 queries."""
    return [
        RetrievalMetrics(precision=0.9, recall=1.0, mrr=0.95, ndcg=0.90, k=10),
        RetrievalMetrics(precision=0.8, recall=0.9, mrr=0.85, ndcg=0.85, k=10),
        RetrievalMetrics(precision=0.7, recall=0.8, mrr=0.7, ndcg=0.75, k=10),
    ]


@pytest.fixture
def degraded_metrics() -> list[RetrievalMetrics]:
    """Degraded metrics for 3 queries."""
    return [
        RetrievalMetrics(precision=0.6, recall=0.5, mrr=0.6, ndcg=0.55, k=10),
        RetrievalMetrics(precision=0.4, recall=0.5, mrr=0.4, ndcg=0.45, k=10),
        RetrievalMetrics(precision=0.3, recall=0.3, mrr=0.3, ndcg=0.35, k=10),
    ]


@pytest.fixture
def baseline_result(sample_testset: TestSet, baseline_metrics: list[RetrievalMetrics]) -> EvaluationResult:
    """Create baseline evaluation result."""
    query_results = [
        QueryResult(query=q, metric=m, answer="", latency_ms=100.0)
        for q, m in zip(sample_testset.queries, baseline_metrics, strict=False)
    ]
    return EvaluationResult(
        testset=sample_testset,
        metrics=baseline_metrics,
        responses=[""] * 3,
        query_results=query_results,
    )


@pytest.fixture
def improved_result(sample_testset: TestSet, improved_metrics: list[RetrievalMetrics]) -> EvaluationResult:
    """Create improved evaluation result."""
    query_results = [
        QueryResult(query=q, metric=m, answer="", latency_ms=100.0)
        for q, m in zip(sample_testset.queries, improved_metrics, strict=False)
    ]
    return EvaluationResult(
        testset=sample_testset,
        metrics=improved_metrics,
        responses=[""] * 3,
        query_results=query_results,
    )


@pytest.fixture
def degraded_result(sample_testset: TestSet, degraded_metrics: list[RetrievalMetrics]) -> EvaluationResult:
    """Create degraded evaluation result."""
    query_results = [
        QueryResult(query=q, metric=m, answer="", latency_ms=100.0)
        for q, m in zip(sample_testset.queries, degraded_metrics, strict=False)
    ]
    return EvaluationResult(
        testset=sample_testset,
        metrics=degraded_metrics,
        responses=[""] * 3,
        query_results=query_results,
    )


# ============================================================================
# QueryDiff Tests
# ============================================================================


class TestQueryDiff:
    """Tests for QueryDiff dataclass."""

    def test_precision_change(self) -> None:
        """precision_change computes correct difference."""
        diff = QueryDiff(
            query_text="test query",
            query_id=None,
            baseline_metrics={"precision": 0.8, "recall": 0.9},
            current_metrics={"precision": 0.9, "recall": 0.85},
            status="improved",
        )
        assert diff.precision_change == pytest.approx(0.1)

    def test_recall_change(self) -> None:
        """recall_change computes correct difference."""
        diff = QueryDiff(
            query_text="test query",
            query_id=None,
            baseline_metrics={"precision": 0.8, "recall": 0.9},
            current_metrics={"precision": 0.9, "recall": 0.85},
            status="improved",
        )
        assert diff.recall_change == pytest.approx(-0.05)

    def test_mrr_change(self) -> None:
        """mrr_change computes correct difference."""
        diff = QueryDiff(
            query_text="test query",
            query_id=None,
            baseline_metrics={"mrr": 0.5},
            current_metrics={"mrr": 0.8},
            status="improved",
        )
        assert diff.mrr_change == pytest.approx(0.3)

    def test_ndcg_change(self) -> None:
        """ndcg_change computes correct difference."""
        diff = QueryDiff(
            query_text="test query",
            query_id=None,
            baseline_metrics={"ndcg": 0.7},
            current_metrics={"ndcg": 0.6},
            status="degraded",
        )
        assert diff.ndcg_change == pytest.approx(-0.1)

    def test_summary_improved(self) -> None:
        """summary() shows positive changes for improved query."""
        diff = QueryDiff(
            query_text="How to authenticate?",
            query_id=None,
            baseline_metrics={"precision": 0.5, "recall": 0.6},
            current_metrics={"precision": 0.8, "recall": 0.9},
            status="improved",
        )
        summary = diff.summary()
        assert "How to authenticate?" in summary
        assert "precision" in summary
        assert "+" in summary

    def test_summary_degraded(self) -> None:
        """summary() shows negative changes for degraded query."""
        diff = QueryDiff(
            query_text="Rate limits?",
            query_id=None,
            baseline_metrics={"precision": 0.8, "recall": 0.9},
            current_metrics={"precision": 0.5, "recall": 0.6},
            status="degraded",
        )
        summary = diff.summary()
        assert "Rate limits?" in summary
        assert "-" in summary

    def test_summary_unchanged(self) -> None:
        """summary() indicates unchanged for same metrics."""
        diff = QueryDiff(
            query_text="Test query",
            query_id=None,
            baseline_metrics={"precision": 0.8, "recall": 0.9},
            current_metrics={"precision": 0.8, "recall": 0.9},
            status="unchanged",
        )
        summary = diff.summary()
        assert "unchanged" in summary

    def test_summary_truncates_long_query(self) -> None:
        """summary() truncates long query text."""
        diff = QueryDiff(
            query_text="This is a very long query that should be truncated in the summary output",
            query_id=None,
            baseline_metrics={"precision": 0.8},
            current_metrics={"precision": 0.8},
            status="unchanged",
        )
        summary = diff.summary()
        assert "..." in summary
        assert len(summary) < 200

    def test_biggest_change(self) -> None:
        """biggest_change() returns metric with largest absolute change."""
        diff = QueryDiff(
            query_text="test",
            query_id=None,
            baseline_metrics={"precision": 0.8, "recall": 0.9, "mrr": 0.5, "ndcg": 0.7},
            current_metrics={"precision": 0.85, "recall": 0.5, "mrr": 0.55, "ndcg": 0.75},
            status="degraded",
        )
        metric, before, after, change = diff.biggest_change()
        # recall has biggest change: 0.5 - 0.9 = -0.4
        assert metric == "recall"
        assert before == pytest.approx(0.9)
        assert after == pytest.approx(0.5)
        assert change == pytest.approx(-0.4)

    def test_zero_baseline_percent_change(self) -> None:
        """summary() handles zero baseline gracefully."""
        diff = QueryDiff(
            query_text="test",
            query_id=None,
            baseline_metrics={"precision": 0.0, "recall": 0.5},
            current_metrics={"precision": 0.5, "recall": 0.5},
            status="improved",
        )
        summary = diff.summary()
        # Should not raise, and should show N/A for zero baseline
        assert "N/A" in summary or "precision" in summary


# ============================================================================
# DiffReport Tests
# ============================================================================


class TestDiffReport:
    """Tests for DiffReport dataclass."""

    @pytest.fixture
    def sample_diff_report(self) -> DiffReport:
        """Create a sample diff report."""
        improved = [
            QueryDiff(
                query_text="Query A",
                query_id="1",
                baseline_metrics={"precision": 0.5, "recall": 0.6, "mrr": 0.5, "ndcg": 0.5},
                current_metrics={"precision": 0.8, "recall": 0.9, "mrr": 0.8, "ndcg": 0.8},
                status="improved",
            ),
        ]
        degraded = [
            QueryDiff(
                query_text="Query B",
                query_id="2",
                baseline_metrics={"precision": 0.9, "recall": 0.9, "mrr": 0.9, "ndcg": 0.9},
                current_metrics={"precision": 0.5, "recall": 0.5, "mrr": 0.5, "ndcg": 0.5},
                status="degraded",
            ),
        ]
        unchanged = [
            QueryDiff(
                query_text="Query C",
                query_id="3",
                baseline_metrics={"precision": 0.7, "recall": 0.7, "mrr": 0.7, "ndcg": 0.7},
                current_metrics={"precision": 0.7, "recall": 0.7, "mrr": 0.7, "ndcg": 0.7},
                status="unchanged",
            ),
        ]
        return DiffReport(
            baseline_id="base-123",
            current_id="curr-456",
            baseline_name="v1.0",
            current_name="v1.1",
            timestamp=datetime(2026, 1, 26, 12, 0, 0),
            testset_name="test-queries",
            metrics_diff={"precision": 0.05, "recall": -0.02, "mrr": 0.03, "ndcg": 0.01},
            improved=improved,
            degraded=degraded,
            unchanged=unchanged,
            baseline_metrics={"precision": 0.7, "recall": 0.73, "mrr": 0.7, "ndcg": 0.7},
            current_metrics={"precision": 0.75, "recall": 0.71, "mrr": 0.73, "ndcg": 0.71},
        )

    def test_total_queries(self, sample_diff_report: DiffReport) -> None:
        """total_queries returns sum of all categories."""
        assert sample_diff_report.total_queries == 3

    def test_improvement_rate(self, sample_diff_report: DiffReport) -> None:
        """improvement_rate calculates correct percentage."""
        # 1 improved out of 3 total
        assert sample_diff_report.improvement_rate == pytest.approx(1 / 3)

    def test_degradation_rate(self, sample_diff_report: DiffReport) -> None:
        """degradation_rate calculates correct percentage."""
        # 1 degraded out of 3 total
        assert sample_diff_report.degradation_rate == pytest.approx(1 / 3)

    def test_summary(self, sample_diff_report: DiffReport) -> None:
        """summary() returns formatted string."""
        summary = sample_diff_report.summary()
        assert "1 improved" in summary
        assert "1 degraded" in summary
        assert "1 unchanged" in summary

    def test_to_markdown_header(self, sample_diff_report: DiffReport) -> None:
        """to_markdown() includes header information."""
        md = sample_diff_report.to_markdown()
        assert "## RAGnarok Diff Report" in md
        assert "v1.0" in md
        assert "v1.1" in md
        assert "test-queries" in md

    def test_to_markdown_tables(self, sample_diff_report: DiffReport) -> None:
        """to_markdown() includes metric tables."""
        md = sample_diff_report.to_markdown()
        assert "| Metric | Baseline | Current | Change |" in md
        assert "Precision" in md
        assert "Recall" in md

    def test_to_markdown_query_sections(self, sample_diff_report: DiffReport) -> None:
        """to_markdown() includes query breakdown sections."""
        md = sample_diff_report.to_markdown()
        assert "### Query Changes" in md
        assert "### Top Degraded Queries" in md
        assert "### Top Improved Queries" in md

    def test_to_dict(self, sample_diff_report: DiffReport) -> None:
        """to_dict() returns JSON-serializable dict."""
        d = sample_diff_report.to_dict()
        assert d["baseline_id"] == "base-123"
        assert d["current_id"] == "curr-456"
        assert d["baseline_name"] == "v1.0"
        assert d["current_name"] == "v1.1"
        assert d["testset_name"] == "test-queries"
        assert "metrics_diff" in d
        assert "summary" in d
        assert d["summary"]["total_queries"] == 3
        assert len(d["improved"]) == 1
        assert len(d["degraded"]) == 1
        assert len(d["unchanged"]) == 1

    def test_empty_report(self) -> None:
        """DiffReport handles empty query lists."""
        report = DiffReport(
            baseline_id=None,
            current_id=None,
            baseline_name="base",
            current_name="current",
            timestamp=datetime.now(),
            testset_name="empty",
            metrics_diff={},
            improved=[],
            degraded=[],
            unchanged=[],
        )
        assert report.total_queries == 0
        assert report.improvement_rate == 0.0
        assert report.degradation_rate == 0.0
        assert "0 improved" in report.summary()


# ============================================================================
# compute_diff Tests
# ============================================================================


class TestComputeDiff:
    """Tests for compute_diff function."""

    def test_basic_diff(self, baseline_result: EvaluationResult, improved_result: EvaluationResult) -> None:
        """compute_diff returns DiffReport for valid inputs."""
        diff = compute_diff(
            baseline_result,
            improved_result,
            baseline_name="baseline",
            current_name="improved",
        )
        assert isinstance(diff, DiffReport)
        assert diff.total_queries == 3
        assert diff.baseline_name == "baseline"
        assert diff.current_name == "improved"

    def test_all_improved(self, baseline_result: EvaluationResult, improved_result: EvaluationResult) -> None:
        """compute_diff detects all improved queries."""
        diff = compute_diff(baseline_result, improved_result)
        # All queries should be improved since improved_metrics > baseline_metrics
        assert len(diff.improved) == 3
        assert len(diff.degraded) == 0
        assert len(diff.unchanged) == 0

    def test_all_degraded(self, baseline_result: EvaluationResult, degraded_result: EvaluationResult) -> None:
        """compute_diff detects all degraded queries."""
        diff = compute_diff(baseline_result, degraded_result)
        # All queries should be degraded
        assert len(diff.degraded) == 3
        assert len(diff.improved) == 0
        assert len(diff.unchanged) == 0

    def test_mixed_changes(self, sample_testset: TestSet, baseline_metrics: list[RetrievalMetrics]) -> None:
        """compute_diff handles mixed improvements/degradations."""
        # Create mixed result: first improved, second degraded, third unchanged
        mixed_metrics = [
            RetrievalMetrics(precision=0.95, recall=1.0, mrr=0.95, ndcg=0.95, k=10),  # improved
            RetrievalMetrics(precision=0.3, recall=0.4, mrr=0.3, ndcg=0.35, k=10),  # degraded
            RetrievalMetrics(precision=0.5, recall=0.6, mrr=0.5, ndcg=0.55, k=10),  # unchanged
        ]
        baseline = EvaluationResult(
            testset=sample_testset,
            metrics=baseline_metrics,
            responses=[""] * 3,
        )
        mixed = EvaluationResult(
            testset=sample_testset,
            metrics=mixed_metrics,
            responses=[""] * 3,
        )

        diff = compute_diff(baseline, mixed)
        assert len(diff.improved) == 1
        assert len(diff.degraded) == 1
        assert len(diff.unchanged) == 1

    def test_query_count_mismatch_raises(self, baseline_result: EvaluationResult) -> None:
        """compute_diff raises ValueError for mismatched query counts."""
        # Create result with different query count
        short_testset = TestSet(
            queries=[Query(text="Only one query", ground_truth_docs=["doc1"])],
            name="short",
        )
        short_result = EvaluationResult(
            testset=short_testset,
            metrics=[RetrievalMetrics(precision=0.8, recall=0.9, mrr=0.8, ndcg=0.8, k=10)],
            responses=[""],
        )

        with pytest.raises(ValueError, match="Query count mismatch"):
            compute_diff(baseline_result, short_result)

    def test_custom_threshold(self, baseline_result: EvaluationResult, sample_testset: TestSet) -> None:
        """compute_diff respects custom change_threshold."""
        # Create result with tiny improvements (below default threshold)
        tiny_improved_metrics = [
            RetrievalMetrics(precision=0.801, recall=1.0, mrr=0.901, ndcg=0.851, k=10),
            RetrievalMetrics(precision=0.601, recall=0.801, mrr=0.701, ndcg=0.751, k=10),
            RetrievalMetrics(precision=0.501, recall=0.601, mrr=0.501, ndcg=0.551, k=10),
        ]
        tiny_result = EvaluationResult(
            testset=sample_testset,
            metrics=tiny_improved_metrics,
            responses=[""] * 3,
        )

        # With default threshold (0.01), these should be unchanged
        diff_default = compute_diff(baseline_result, tiny_result)
        assert len(diff_default.unchanged) == 3

        # With lower threshold (0.0001), these should be improved
        diff_sensitive = compute_diff(baseline_result, tiny_result, change_threshold=0.0001)
        assert len(diff_sensitive.improved) == 3

    def test_zero_baseline_handled(self, sample_testset: TestSet) -> None:
        """compute_diff handles zero baseline metrics gracefully."""
        zero_metrics = [
            RetrievalMetrics(precision=0.0, recall=0.0, mrr=0.0, ndcg=0.0, k=10),
            RetrievalMetrics(precision=0.0, recall=0.0, mrr=0.0, ndcg=0.0, k=10),
            RetrievalMetrics(precision=0.0, recall=0.0, mrr=0.0, ndcg=0.0, k=10),
        ]
        improved_metrics = [
            RetrievalMetrics(precision=0.5, recall=0.5, mrr=0.5, ndcg=0.5, k=10),
            RetrievalMetrics(precision=0.5, recall=0.5, mrr=0.5, ndcg=0.5, k=10),
            RetrievalMetrics(precision=0.5, recall=0.5, mrr=0.5, ndcg=0.5, k=10),
        ]
        zero_result = EvaluationResult(
            testset=sample_testset,
            metrics=zero_metrics,
            responses=[""] * 3,
        )
        improved_result = EvaluationResult(
            testset=sample_testset,
            metrics=improved_metrics,
            responses=[""] * 3,
        )

        # Should not raise
        diff = compute_diff(zero_result, improved_result)
        assert diff.total_queries == 3
        # All improved from zero
        assert len(diff.improved) == 3

    def test_query_matching_by_text(self) -> None:
        """compute_diff matches queries by text, not index."""
        # Create testsets with same queries but different order
        testset1 = TestSet(
            queries=[
                Query(text="Query A", ground_truth_docs=["doc1"]),
                Query(text="Query B", ground_truth_docs=["doc2"]),
            ],
            name="test",
        )
        testset2 = TestSet(
            queries=[
                Query(text="Query B", ground_truth_docs=["doc2"]),  # Swapped order
                Query(text="Query A", ground_truth_docs=["doc1"]),
            ],
            name="test",
        )

        metrics1 = [
            RetrievalMetrics(precision=0.8, recall=0.8, mrr=0.8, ndcg=0.8, k=10),  # Query A
            RetrievalMetrics(precision=0.5, recall=0.5, mrr=0.5, ndcg=0.5, k=10),  # Query B
        ]
        metrics2 = [
            RetrievalMetrics(precision=0.5, recall=0.5, mrr=0.5, ndcg=0.5, k=10),  # Query B (same)
            RetrievalMetrics(precision=0.9, recall=0.9, mrr=0.9, ndcg=0.9, k=10),  # Query A (improved)
        ]

        result1 = EvaluationResult(testset=testset1, metrics=metrics1, responses=[""] * 2)
        result2 = EvaluationResult(testset=testset2, metrics=metrics2, responses=[""] * 2)

        diff = compute_diff(result1, result2)

        # Query A should be detected as improved (0.8 -> 0.9)
        # Query B should be unchanged (0.5 -> 0.5)
        assert len(diff.improved) == 1
        assert len(diff.unchanged) == 1
        assert diff.improved[0].query_text == "Query A"

    def test_metrics_diff_computed(self, baseline_result: EvaluationResult, improved_result: EvaluationResult) -> None:
        """compute_diff computes aggregate metrics_diff correctly."""
        diff = compute_diff(baseline_result, improved_result)

        # Check that metrics_diff contains the 4 standard metrics
        assert "precision" in diff.metrics_diff
        assert "recall" in diff.metrics_diff
        assert "mrr" in diff.metrics_diff
        assert "ndcg" in diff.metrics_diff

        # All should be positive (improved)
        for metric in ["precision", "recall", "mrr", "ndcg"]:
            assert diff.metrics_diff[metric] >= 0

    def test_optional_ids(self, baseline_result: EvaluationResult, improved_result: EvaluationResult) -> None:
        """compute_diff accepts optional baseline_id and current_id."""
        diff = compute_diff(
            baseline_result,
            improved_result,
            baseline_id="record-123",
            current_id="record-456",
        )
        assert diff.baseline_id == "record-123"
        assert diff.current_id == "record-456"

    def test_testset_name_from_current(
        self, baseline_result: EvaluationResult, improved_result: EvaluationResult
    ) -> None:
        """compute_diff uses testset name from current result."""
        diff = compute_diff(baseline_result, improved_result)
        assert diff.testset_name == "test-queries"
