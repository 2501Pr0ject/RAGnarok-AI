"""Tests for notebook display module."""

import pytest

from ragnarok_ai.core.evaluate import EvaluationResult
from ragnarok_ai.cost.tracker import CostSummary, ProviderUsage
from ragnarok_ai.notebook.display import (
    _format_number,
    _in_notebook,
    _progress_bar,
    display,
    display_comparison,
    display_cost,
    display_metrics,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_number_zero(self):
        """Test formatting zero."""
        assert _format_number(0) == "0"

    def test_format_number_small(self):
        """Test formatting small numbers."""
        result = _format_number(0.001)
        assert "0.001" in result

    def test_format_number_normal(self):
        """Test formatting normal numbers."""
        assert _format_number(0.85) == "0.85"
        assert _format_number(0.857, decimals=3) == "0.857"

    def test_progress_bar_contains_html(self):
        """Test progress bar returns HTML."""
        html = _progress_bar(0.75)
        assert "<div" in html
        assert "75%" in html or "75.0%" in html

    def test_progress_bar_colors(self):
        """Test progress bar uses different colors."""
        low = _progress_bar(0.25)
        mid = _progress_bar(0.60)
        high = _progress_bar(0.90)

        # Low should be red
        assert "#e74c3c" in low
        # Mid should be orange
        assert "#f39c12" in mid
        # High should be green
        assert "#27ae60" in high


class TestInNotebook:
    """Tests for notebook detection."""

    def test_in_notebook_returns_bool(self):
        """Test that _in_notebook returns a boolean."""
        result = _in_notebook()
        assert isinstance(result, bool)

    def test_in_notebook_outside_notebook(self):
        """Test detection outside notebook environment."""
        # In test environment, we're not in a notebook
        assert _in_notebook() is False


class TestDisplayMetrics:
    """Tests for display_metrics function."""

    @pytest.fixture
    def mock_result(self):
        """Create a mock EvaluationResult."""
        from ragnarok_ai.core.types import TestSet
        from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

        metrics = [
            RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10),
            RetrievalMetrics(precision=0.7, recall=0.8, mrr=0.8, ndcg=0.75, k=10),
        ]

        testset = TestSet(queries=[])

        return EvaluationResult(
            testset=testset,
            metrics=metrics,
            responses=["answer1", "answer2"],
            total_latency_ms=1500.0,
        )

    def test_display_metrics_generates_html(self, mock_result, capsys):
        """Test that display_metrics works without errors."""
        # Should not raise
        display_metrics(mock_result)
        captured = capsys.readouterr()
        # Outside notebook, should print fallback message
        assert "HTML" in captured.out or captured.out == ""


class TestDisplayCost:
    """Tests for display_cost function."""

    @pytest.fixture
    def result_with_cost(self):
        """Create result with cost tracking."""
        from ragnarok_ai.core.types import TestSet

        testset = TestSet(queries=[])

        cost = CostSummary(
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost=0.01,
            by_provider={
                "openai:gpt-4o": ProviderUsage(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=1000,
                    output_tokens=500,
                    cost=0.01,
                    call_count=5,
                ),
            },
        )

        return EvaluationResult(
            testset=testset,
            metrics=[],
            responses=[],
            cost=cost,
        )

    @pytest.fixture
    def result_without_cost(self):
        """Create result without cost tracking."""
        from ragnarok_ai.core.types import TestSet

        testset = TestSet(queries=[])

        return EvaluationResult(
            testset=testset,
            metrics=[],
            responses=[],
            cost=None,
        )

    def test_display_cost_with_tracking(self, result_with_cost, capsys):
        """Test display_cost with cost tracking enabled."""
        display_cost(result_with_cost)
        captured = capsys.readouterr()
        assert "HTML" in captured.out or captured.out == ""

    def test_display_cost_without_tracking(self, result_without_cost, capsys):
        """Test display_cost without cost tracking."""
        display_cost(result_without_cost)
        captured = capsys.readouterr()
        assert "HTML" in captured.out or captured.out == ""


class TestDisplay:
    """Tests for main display function."""

    @pytest.fixture
    def full_result(self):
        """Create a full EvaluationResult."""
        from ragnarok_ai.core.types import TestSet
        from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

        metrics = [
            RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10),
        ]

        testset = TestSet(queries=[])

        cost = CostSummary(
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_cost=0.05,
            by_provider={
                "openai:gpt-4o": ProviderUsage(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=3000,
                    output_tokens=1000,
                    cost=0.05,
                    call_count=10,
                ),
                "ollama:llama2": ProviderUsage(
                    provider="ollama",
                    model="llama2",
                    input_tokens=2000,
                    output_tokens=1000,
                    cost=0.0,
                    call_count=5,
                ),
            },
        )

        return EvaluationResult(
            testset=testset,
            metrics=metrics,
            responses=["answer"],
            total_latency_ms=2500.0,
            cost=cost,
        )

    def test_display_full_result(self, full_result, capsys):
        """Test display with full result."""
        display(full_result)
        captured = capsys.readouterr()
        # Should work without errors
        assert "HTML" in captured.out or captured.out == ""


class TestDisplayComparison:
    """Tests for display_comparison function."""

    @pytest.fixture
    def comparison_results(self):
        """Create multiple results for comparison."""
        from ragnarok_ai.core.types import TestSet
        from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

        testset = TestSet(queries=[])

        result1 = EvaluationResult(
            testset=testset,
            metrics=[RetrievalMetrics(precision=0.7, recall=0.6, mrr=0.8, ndcg=0.7, k=10)],
            responses=["a"],
            total_latency_ms=1000.0,
        )

        result2 = EvaluationResult(
            testset=testset,
            metrics=[RetrievalMetrics(precision=0.9, recall=0.8, mrr=0.95, ndcg=0.9, k=10)],
            responses=["b"],
            total_latency_ms=1500.0,
        )

        return [("Baseline", result1), ("Improved", result2)]

    def test_display_comparison(self, comparison_results, capsys):
        """Test display_comparison works."""
        display_comparison(comparison_results)
        captured = capsys.readouterr()
        assert "HTML" in captured.out or captured.out == ""

    def test_display_comparison_empty(self, capsys):
        """Test display_comparison with empty list."""
        display_comparison([])
        captured = capsys.readouterr()
        assert "HTML" in captured.out or captured.out == ""
