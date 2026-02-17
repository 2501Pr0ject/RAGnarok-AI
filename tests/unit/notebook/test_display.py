"""Tests for notebook display functions."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from ragnarok_ai.notebook.display import (
    _display_cost_summary,
    _format_number,
    _in_notebook,
    _progress_bar,
    display,
    display_comparison,
    display_cost,
    display_metrics,
)


@dataclass
class MockProviderUsage:
    """Mock provider usage for testing."""

    provider: str
    model: str = "test-model"
    total_tokens: int = 1000
    cost: float = 0.01
    is_local: bool = False


@dataclass
class MockCostSummary:
    """Mock cost summary for testing."""

    total_cost: float = 0.05
    total_tokens: int = 5000
    by_provider: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.by_provider:
            self.by_provider = {
                "openai": MockProviderUsage("openai"),
                "anthropic": MockProviderUsage("anthropic", cost=0.02),
            }


@dataclass
class MockEvaluationResult:
    """Mock evaluation result for testing."""

    total_latency_ms: float = 1500.0
    cost: MockCostSummary | None = None
    errors: list = field(default_factory=list)
    _summary: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return self._summary


class TestInNotebook:
    """Tests for _in_notebook detection."""

    def test_not_in_notebook_terminal(self) -> None:
        """Test detection in terminal IPython."""
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "TerminalInteractiveShell"

        # Create a mock IPython module with get_ipython function
        mock_ipython = MagicMock()
        mock_ipython.get_ipython = MagicMock(return_value=mock_shell)

        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            result = _in_notebook()
            assert result is False

    def test_in_notebook_zmq(self) -> None:
        """Test detection in Jupyter notebook."""
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = MagicMock(return_value=mock_shell)

        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            result = _in_notebook()
            assert result is True

    def test_not_in_notebook_none_shell(self) -> None:
        """Test detection when get_ipython returns None."""
        mock_ipython = MagicMock()
        mock_ipython.get_ipython = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            result = _in_notebook()
            assert result is False

    def test_not_in_notebook_unknown_shell(self) -> None:
        """Test detection with unknown shell type."""
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "UnknownShell"

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = MagicMock(return_value=mock_shell)

        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            result = _in_notebook()
            assert result is False

    def test_not_in_notebook_import_error(self) -> None:
        """Test detection when IPython not installed."""
        with patch.dict(sys.modules, {"IPython": None}):
            # This will cause ImportError inside _in_notebook
            result = _in_notebook()
            # Should return False when IPython not available
            assert result is False


class TestFormatNumber:
    """Tests for _format_number function."""

    def test_format_zero(self) -> None:
        """Test formatting zero."""
        assert _format_number(0) == "0"

    def test_format_small_number(self) -> None:
        """Test formatting very small numbers."""
        result = _format_number(0.001)
        assert "0.0010" in result

    def test_format_normal_number(self) -> None:
        """Test formatting normal numbers."""
        result = _format_number(0.85)
        assert result == "0.85"

    def test_format_integer(self) -> None:
        """Test formatting integer value."""
        result = _format_number(1.0)
        assert result == "1.00"

    def test_format_custom_decimals(self) -> None:
        """Test formatting with custom decimal places."""
        result = _format_number(0.12345, decimals=3)
        assert result == "0.123"


class TestProgressBar:
    """Tests for _progress_bar function."""

    def test_progress_bar_full(self) -> None:
        """Test full progress bar."""
        result = _progress_bar(1.0)
        assert "█" in result
        assert result.count("█") == 20

    def test_progress_bar_empty(self) -> None:
        """Test empty progress bar."""
        result = _progress_bar(0.0)
        assert "░" in result
        assert result.count("░") == 20

    def test_progress_bar_half(self) -> None:
        """Test half-filled progress bar."""
        result = _progress_bar(0.5)
        assert "█" in result
        assert "░" in result
        assert result.count("█") == 10

    def test_progress_bar_low_value_red(self) -> None:
        """Test progress bar color for low values."""
        result = _progress_bar(0.3)
        assert "#f85149" in result

    def test_progress_bar_medium_value_yellow(self) -> None:
        """Test progress bar color for medium values."""
        result = _progress_bar(0.6)
        assert "#d29922" in result

    def test_progress_bar_high_value_green(self) -> None:
        """Test progress bar color for high values."""
        result = _progress_bar(0.9)
        assert "#3fb950" in result

    def test_progress_bar_clamped_over_max(self) -> None:
        """Test progress bar with value over max."""
        result = _progress_bar(1.5)
        assert result.count("█") == 20

    def test_progress_bar_custom_max(self) -> None:
        """Test progress bar with custom max value."""
        result = _progress_bar(50, max_value=100)
        assert "█" in result
        assert "░" in result
        assert result.count("█") == 10


class TestDisplayHtml:
    """Tests for _display_html function."""

    def test_display_html_not_in_notebook(self, capsys) -> None:
        """Test HTML display fallback outside notebook."""
        import sys

        # Get the actual module from sys.modules
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        from ragnarok_ai.notebook.display import _display_html

        # Patch at module level
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False
        try:
            _display_html("<p>Test</p>")
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_html_in_notebook(self) -> None:
        """Test HTML display in notebook environment."""
        import sys
        from unittest.mock import MagicMock

        display_module = sys.modules["ragnarok_ai.notebook.display"]
        from ragnarok_ai.notebook.display import _display_html

        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: True

        # Mock IPython display module
        mock_display_func = MagicMock()
        mock_html_class = MagicMock()
        mock_ipython_display = MagicMock()
        mock_ipython_display.display = mock_display_func
        mock_ipython_display.HTML = mock_html_class

        try:
            with patch.dict("sys.modules", {"IPython.display": mock_ipython_display}):
                _display_html("<p>Test</p>")
                # Verify display was called
                mock_display_func.assert_called_once()
        finally:
            display_module._in_notebook = original_in_notebook


class TestDisplayMetrics:
    """Tests for display_metrics function."""

    def test_display_metrics_with_data(self, capsys) -> None:
        """Test displaying metrics with data."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(
            _summary={
                "precision": 0.8,
                "recall": 0.7,
                "mrr": 0.9,
                "ndcg": 0.85,
                "num_queries": 10,
            }
        )

        try:
            display_metrics(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_metrics_empty_summary(self, capsys) -> None:
        """Test displaying metrics with empty summary."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(_summary={})

        try:
            display_metrics(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out


class TestDisplayCost:
    """Tests for display_cost function."""

    def test_display_cost_with_tracking(self, capsys) -> None:
        """Test displaying cost when tracking is enabled."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(cost=MockCostSummary())

        try:
            display_cost(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_cost_no_tracking(self, capsys) -> None:
        """Test displaying cost when tracking is disabled."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(cost=None)

        try:
            display_cost(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out


class TestDisplayCostSummary:
    """Tests for _display_cost_summary function."""

    def test_display_cost_summary(self, capsys) -> None:
        """Test displaying cost summary."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        cost = MockCostSummary()

        try:
            _display_cost_summary(cost)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_cost_summary_with_local_provider(self, capsys) -> None:
        """Test displaying cost summary with local provider."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        cost = MockCostSummary(
            by_provider={
                "ollama": MockProviderUsage("ollama", is_local=True, cost=0.0),
            }
        )

        try:
            _display_cost_summary(cost)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_cost_summary_zero_cost(self, capsys) -> None:
        """Test displaying cost summary with zero total cost."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        cost = MockCostSummary(
            total_cost=0.0,
            by_provider={
                "ollama": MockProviderUsage("ollama", is_local=True, cost=0.0),
            },
        )

        try:
            _display_cost_summary(cost)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out


class TestDisplay:
    """Tests for display function."""

    def test_display_full_result(self, capsys) -> None:
        """Test full display with all data."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(
            total_latency_ms=2500.0,
            cost=MockCostSummary(),
            _summary={
                "precision": 0.8,
                "recall": 0.7,
                "mrr": 0.9,
                "ndcg": 0.85,
                "num_queries": 10,
            },
        )

        try:
            display(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_without_cost(self, capsys) -> None:
        """Test display without cost tracking."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(
            _summary={
                "precision": 0.8,
                "recall": 0.7,
                "num_queries": 5,
            }
        )

        try:
            display(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_with_errors(self, capsys) -> None:
        """Test display with errors."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(
            errors=["Error 1", "Error 2"],
            _summary={"num_queries": 5},
        )

        try:
            display(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_zero_cost(self, capsys) -> None:
        """Test display with zero cost."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(
            cost=MockCostSummary(
                total_cost=0.0,
                by_provider={
                    "local": MockProviderUsage("local", is_local=True, cost=0.0),
                },
            ),
            _summary={"num_queries": 5},
        )

        try:
            display(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_empty_summary(self, capsys) -> None:
        """Test display with empty summary."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        result = MockEvaluationResult(_summary={})

        try:
            display(result)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out


class TestDisplayComparison:
    """Tests for display_comparison function."""

    def test_display_comparison_multiple_results(self, capsys) -> None:
        """Test comparing multiple results."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        results = [
            (
                "Baseline",
                MockEvaluationResult(
                    total_latency_ms=1000.0,
                    cost=MockCostSummary(total_cost=0.05),
                    _summary={"precision": 0.7, "recall": 0.6, "mrr": 0.8, "ndcg": 0.75},
                ),
            ),
            (
                "New Model",
                MockEvaluationResult(
                    total_latency_ms=800.0,
                    cost=MockCostSummary(total_cost=0.03),
                    _summary={"precision": 0.85, "recall": 0.75, "mrr": 0.9, "ndcg": 0.88},
                ),
            ),
        ]

        try:
            display_comparison(results)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_comparison_empty_results(self, capsys) -> None:
        """Test comparing empty results."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        try:
            display_comparison([])
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_comparison_single_result(self, capsys) -> None:
        """Test comparing single result."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        results = [
            (
                "Only Result",
                MockEvaluationResult(
                    total_latency_ms=1000.0,
                    _summary={"precision": 0.8, "recall": 0.7, "mrr": 0.85, "ndcg": 0.8},
                ),
            ),
        ]

        try:
            display_comparison(results)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_comparison_without_cost(self, capsys) -> None:
        """Test comparing results without cost tracking."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        results = [
            (
                "Model A",
                MockEvaluationResult(
                    total_latency_ms=1000.0,
                    cost=None,
                    _summary={"precision": 0.8},
                ),
            ),
            (
                "Model B",
                MockEvaluationResult(
                    total_latency_ms=1200.0,
                    cost=None,
                    _summary={"precision": 0.75},
                ),
            ),
        ]

        try:
            display_comparison(results)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out

    def test_display_comparison_zero_metrics(self, capsys) -> None:
        """Test comparing results with zero metrics."""
        display_module = sys.modules["ragnarok_ai.notebook.display"]
        original_in_notebook = display_module._in_notebook
        display_module._in_notebook = lambda: False

        results = [
            (
                "Empty",
                MockEvaluationResult(
                    total_latency_ms=100.0,
                    _summary={"precision": 0, "recall": 0, "mrr": 0, "ndcg": 0},
                ),
            ),
        ]

        try:
            display_comparison(results)
        finally:
            display_module._in_notebook = original_in_notebook

        captured = capsys.readouterr()
        assert "HTML output available" in captured.out
