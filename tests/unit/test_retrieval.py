"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from ragnarok_ai.core.types import Document, Query, RetrievalResult
from ragnarok_ai.evaluators.retrieval import (
    RetrievalMetrics,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self) -> None:
        """All retrieved docs are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c", "d"]

        result = precision_at_k(retrieved, relevant, k=3)

        assert result == 1.0

    def test_zero_precision(self) -> None:
        """No retrieved docs are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = ["d", "e", "f"]

        result = precision_at_k(retrieved, relevant, k=3)

        assert result == 0.0

    def test_partial_precision(self) -> None:
        """Some retrieved docs are relevant."""
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "c", "e"]

        result = precision_at_k(retrieved, relevant, k=4)

        assert result == 0.5

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list returns 0."""
        result = precision_at_k([], ["a", "b"], k=3)

        assert result == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant list returns 0."""
        result = precision_at_k(["a", "b", "c"], [], k=3)

        assert result == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """K larger than retrieved list uses all retrieved."""
        retrieved = ["a", "b"]
        relevant = ["a", "b", "c"]

        result = precision_at_k(retrieved, relevant, k=10)

        assert result == 1.0

    def test_k_zero(self) -> None:
        """K=0 returns 0."""
        result = precision_at_k(["a", "b"], ["a"], k=0)

        assert result == 0.0

    def test_k_negative(self) -> None:
        """Negative K returns 0."""
        result = precision_at_k(["a", "b"], ["a"], k=-1)

        assert result == 0.0

    @pytest.mark.parametrize(
        ("retrieved", "relevant", "k", "expected"),
        [
            (["a", "b", "c"], ["a", "b"], 3, 2 / 3),
            (["a", "b", "c"], ["a"], 2, 0.5),
            (["a", "b", "c", "d"], ["b", "d"], 4, 0.5),
            (["a", "b", "c"], ["a", "b", "c"], 2, 1.0),
            (["a", "b", "c"], ["d", "e", "f"], 3, 0.0),
        ],
    )
    def test_parametrized_precision(
        self, retrieved: list[str], relevant: list[str], k: int, expected: float
    ) -> None:
        """Parametrized precision tests."""
        result = precision_at_k(retrieved, relevant, k=k)

        assert result == pytest.approx(expected)


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """All relevant docs are retrieved."""
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "c"]

        result = recall_at_k(retrieved, relevant, k=4)

        assert result == 1.0

    def test_zero_recall(self) -> None:
        """No relevant docs are retrieved."""
        retrieved = ["a", "b", "c"]
        relevant = ["d", "e"]

        result = recall_at_k(retrieved, relevant, k=3)

        assert result == 0.0

    def test_partial_recall(self) -> None:
        """Some relevant docs are retrieved."""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "d", "e"]

        result = recall_at_k(retrieved, relevant, k=3)

        assert result == pytest.approx(1 / 3)

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list returns 0."""
        result = recall_at_k([], ["a", "b"], k=3)

        assert result == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant list returns 0 (avoid division by zero)."""
        result = recall_at_k(["a", "b", "c"], [], k=3)

        assert result == 0.0

    def test_k_limits_results(self) -> None:
        """K limits which retrieved docs are considered."""
        retrieved = ["a", "b", "c", "d"]
        relevant = ["c", "d"]

        result = recall_at_k(retrieved, relevant, k=2)

        assert result == 0.0  # c and d are not in top 2

    @pytest.mark.parametrize(
        ("retrieved", "relevant", "k", "expected"),
        [
            (["a", "b", "c"], ["a", "b", "c", "d"], 3, 0.75),
            (["a", "b", "c"], ["a", "b"], 3, 1.0),
            (["a", "b", "c", "d"], ["b", "d", "e", "f"], 4, 0.5),
        ],
    )
    def test_parametrized_recall(
        self, retrieved: list[str], relevant: list[str], k: int, expected: float
    ) -> None:
        """Parametrized recall tests."""
        result = recall_at_k(retrieved, relevant, k=k)

        assert result == pytest.approx(expected)


class TestMRR:
    """Tests for mrr (Mean Reciprocal Rank) function."""

    def test_first_result_relevant(self) -> None:
        """First result is relevant, MRR = 1.0."""
        result = mrr(["a", "b", "c"], ["a", "d"])

        assert result == 1.0

    def test_second_result_relevant(self) -> None:
        """Second result is first relevant, MRR = 0.5."""
        result = mrr(["a", "b", "c"], ["b", "d"])

        assert result == 0.5

    def test_third_result_relevant(self) -> None:
        """Third result is first relevant, MRR = 1/3."""
        result = mrr(["a", "b", "c"], ["c", "d"])

        assert result == pytest.approx(1 / 3)

    def test_no_relevant_found(self) -> None:
        """No relevant docs in retrieved, MRR = 0.0."""
        result = mrr(["a", "b", "c"], ["d", "e"])

        assert result == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list returns 0."""
        result = mrr([], ["a", "b"])

        assert result == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant list returns 0."""
        result = mrr(["a", "b", "c"], [])

        assert result == 0.0

    def test_multiple_relevant_uses_first(self) -> None:
        """Multiple relevant docs, uses position of first."""
        result = mrr(["a", "b", "c", "d"], ["b", "d"])

        assert result == 0.5  # b is at position 2

    @pytest.mark.parametrize(
        ("retrieved", "relevant", "expected"),
        [
            (["a"], ["a"], 1.0),
            (["a", "b", "c", "d", "e"], ["e"], 0.2),
            (["a", "b", "c"], ["a", "b", "c"], 1.0),
        ],
    )
    def test_parametrized_mrr(
        self, retrieved: list[str], relevant: list[str], expected: float
    ) -> None:
        """Parametrized MRR tests."""
        result = mrr(retrieved, relevant)

        assert result == pytest.approx(expected)


class TestNDCGAtK:
    """Tests for ndcg_at_k function."""

    def test_perfect_ranking(self) -> None:
        """All relevant docs at top positions, NDCG = 1.0."""
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "b"]

        result = ndcg_at_k(retrieved, relevant, k=4)

        assert result == 1.0

    def test_worst_ranking(self) -> None:
        """No relevant docs retrieved, NDCG = 0.0."""
        retrieved = ["a", "b", "c"]
        relevant = ["d", "e"]

        result = ndcg_at_k(retrieved, relevant, k=3)

        assert result == 0.0

    def test_partial_ranking(self) -> None:
        """Relevant docs not at top, NDCG < 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "c"]  # c is at position 3, not 2

        result = ndcg_at_k(retrieved, relevant, k=3)

        # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
        assert 0.0 < result < 1.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list returns 0."""
        result = ndcg_at_k([], ["a", "b"], k=3)

        assert result == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant list returns 0."""
        result = ndcg_at_k(["a", "b", "c"], [], k=3)

        assert result == 0.0

    def test_k_zero(self) -> None:
        """K=0 returns 0."""
        result = ndcg_at_k(["a", "b"], ["a"], k=0)

        assert result == 0.0

    def test_single_relevant_at_first(self) -> None:
        """Single relevant doc at first position."""
        result = ndcg_at_k(["a", "b", "c"], ["a"], k=3)

        assert result == 1.0

    def test_single_relevant_at_last(self) -> None:
        """Single relevant doc at last position."""
        retrieved = ["a", "b", "c"]
        relevant = ["c"]

        result = ndcg_at_k(retrieved, relevant, k=3)

        # DCG = 1/log2(4) = 0.5
        # IDCG = 1/log2(2) = 1.0
        assert result == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("retrieved", "relevant", "k", "expected_approx"),
        [
            (["a", "b", "c"], ["a", "b", "c"], 3, 1.0),
            (["a", "b", "c"], ["a"], 3, 1.0),
            (["c", "b", "a"], ["a"], 3, 0.5),  # a at position 3
        ],
    )
    def test_parametrized_ndcg(
        self, retrieved: list[str], relevant: list[str], k: int, expected_approx: float
    ) -> None:
        """Parametrized NDCG tests."""
        result = ndcg_at_k(retrieved, relevant, k=k)

        assert result == pytest.approx(expected_approx)


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics model."""

    def test_valid_metrics(self) -> None:
        """Valid metrics are accepted."""
        metrics = RetrievalMetrics(
            precision=0.8,
            recall=0.6,
            mrr=1.0,
            ndcg=0.92,
            k=10,
        )

        assert metrics.precision == 0.8
        assert metrics.recall == 0.6
        assert metrics.mrr == 1.0
        assert metrics.ndcg == 0.92
        assert metrics.k == 10

    def test_boundary_values(self) -> None:
        """Boundary values (0.0 and 1.0) are accepted."""
        metrics = RetrievalMetrics(
            precision=0.0,
            recall=1.0,
            mrr=0.0,
            ndcg=1.0,
            k=1,
        )

        assert metrics.precision == 0.0
        assert metrics.recall == 1.0

    def test_invalid_precision_too_high(self) -> None:
        """Precision > 1.0 raises error."""
        with pytest.raises(ValueError):
            RetrievalMetrics(precision=1.5, recall=0.5, mrr=0.5, ndcg=0.5, k=10)

    def test_invalid_k_zero(self) -> None:
        """K=0 raises error."""
        with pytest.raises(ValueError):
            RetrievalMetrics(precision=0.5, recall=0.5, mrr=0.5, ndcg=0.5, k=0)


class TestEvaluateRetrieval:
    """Tests for evaluate_retrieval function."""

    @pytest.fixture
    def sample_docs(self) -> list[Document]:
        """Sample documents for testing."""
        return [
            Document(id="d1", content="First document"),
            Document(id="d2", content="Second document"),
            Document(id="d3", content="Third document"),
            Document(id="d4", content="Fourth document"),
        ]

    def test_basic_evaluation(self, sample_docs: list[Document]) -> None:
        """Basic evaluation with ground truth."""
        query = Query(text="test query", ground_truth_docs=["d1", "d3"])
        result = RetrievalResult(query=query, retrieved_docs=sample_docs[:3])

        metrics = evaluate_retrieval(result, k=3)

        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.k == 3
        assert metrics.precision == pytest.approx(2 / 3)
        assert metrics.recall == 1.0
        assert metrics.mrr == 1.0

    def test_no_ground_truth(self, sample_docs: list[Document]) -> None:
        """No ground truth docs results in zero metrics."""
        query = Query(text="test query", ground_truth_docs=[])
        result = RetrievalResult(query=query, retrieved_docs=sample_docs[:3])

        metrics = evaluate_retrieval(result, k=3)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg == 0.0

    def test_no_retrieved_docs(self) -> None:
        """No retrieved docs results in zero metrics."""
        query = Query(text="test query", ground_truth_docs=["d1", "d2"])
        result = RetrievalResult(query=query, retrieved_docs=[])

        metrics = evaluate_retrieval(result, k=3)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg == 0.0

    def test_perfect_retrieval(self, sample_docs: list[Document]) -> None:
        """Perfect retrieval with all relevant docs at top."""
        query = Query(text="test query", ground_truth_docs=["d1", "d2"])
        result = RetrievalResult(query=query, retrieved_docs=sample_docs[:2])

        metrics = evaluate_retrieval(result, k=2)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg == 1.0

    def test_invalid_k_raises_error(self, sample_docs: list[Document]) -> None:
        """K < 1 raises ValueError."""
        query = Query(text="test query", ground_truth_docs=["d1"])
        result = RetrievalResult(query=query, retrieved_docs=sample_docs)

        with pytest.raises(ValueError, match="k must be at least 1"):
            evaluate_retrieval(result, k=0)

    def test_default_k_value(self, sample_docs: list[Document]) -> None:
        """Default K value is 10."""
        query = Query(text="test query", ground_truth_docs=["d1"])
        result = RetrievalResult(query=query, retrieved_docs=sample_docs)

        metrics = evaluate_retrieval(result)

        assert metrics.k == 10
