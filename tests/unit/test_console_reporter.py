"""Tests for console reporter."""

from __future__ import annotations

from io import StringIO

import pytest

from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
from ragnarok_ai.reporters.console import (
    ConsoleReporter,
    Threshold,
    _supports_color,
)


class TestConsoleReporterInit:
    """Tests for ConsoleReporter initialization."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        output = StringIO()
        reporter = ConsoleReporter(output=output)

        assert reporter.output is output
        assert "precision" in reporter.thresholds
        assert "recall" in reporter.thresholds

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        custom = {"precision": Threshold(good=0.9, warning=0.7)}
        reporter = ConsoleReporter(thresholds=custom, output=StringIO())

        assert reporter.thresholds["precision"].good == 0.9
        assert reporter.thresholds["precision"].warning == 0.7

    def test_colors_disabled(self) -> None:
        """Colors can be disabled."""
        reporter = ConsoleReporter(use_colors=False, output=StringIO())

        assert reporter.use_colors is False


class TestConsoleReporterGetStatus:
    """Tests for status indicator logic."""

    @pytest.fixture
    def reporter(self) -> ConsoleReporter:
        """Reporter with colors disabled for testing."""
        return ConsoleReporter(use_colors=False, output=StringIO())

    def test_good_status(self, reporter: ConsoleReporter) -> None:
        """High values get good status."""
        status, _ = reporter._get_status("precision", 0.85)

        assert status == "✅"

    def test_warning_status(self, reporter: ConsoleReporter) -> None:
        """Medium values get warning status."""
        status, _ = reporter._get_status("precision", 0.65)

        assert status == "⚠️ "

    def test_bad_status(self, reporter: ConsoleReporter) -> None:
        """Low values get bad status."""
        status, _ = reporter._get_status("precision", 0.4)

        assert status == "❌"

    def test_hallucination_inverted(self, reporter: ConsoleReporter) -> None:
        """Hallucination metric has inverted thresholds (lower is better)."""
        # Low hallucination is good
        status_low, _ = reporter._get_status("hallucination", 0.1)
        assert status_low == "✅"

        # High hallucination is bad
        status_high, _ = reporter._get_status("hallucination", 0.5)
        assert status_high == "❌"


class TestConsoleReporterOutput:
    """Tests for reporter output."""

    @pytest.fixture
    def output(self) -> StringIO:
        """String buffer for capturing output."""
        return StringIO()

    @pytest.fixture
    def reporter(self, output: StringIO) -> ConsoleReporter:
        """Reporter with colors disabled."""
        return ConsoleReporter(use_colors=False, output=output)

    @pytest.fixture
    def sample_metrics(self) -> RetrievalMetrics:
        """Sample metrics for testing."""
        return RetrievalMetrics(
            precision=0.82,
            recall=0.74,
            mrr=0.91,
            ndcg=0.85,
            k=10,
        )

    def test_report_retrieval_metrics(
        self, reporter: ConsoleReporter, output: StringIO, sample_metrics: RetrievalMetrics
    ) -> None:
        """Report outputs metrics table."""
        reporter.report_retrieval_metrics(sample_metrics)
        result = output.getvalue()

        assert "Precision@10" in result
        assert "Recall@10" in result
        assert "MRR" in result
        assert "NDCG@10" in result
        assert "0.82" in result
        assert "0.74" in result

    def test_report_retrieval_metrics_with_title(
        self, reporter: ConsoleReporter, output: StringIO, sample_metrics: RetrievalMetrics
    ) -> None:
        """Report can include a title."""
        reporter.report_retrieval_metrics(sample_metrics, title="Test Query")
        result = output.getvalue()

        assert "Test Query" in result

    def test_report_summary(
        self, reporter: ConsoleReporter, output: StringIO, sample_metrics: RetrievalMetrics
    ) -> None:
        """Summary reports averages across multiple metrics."""
        metrics_list = [
            sample_metrics,
            RetrievalMetrics(precision=0.78, recall=0.66, mrr=0.85, ndcg=0.81, k=10),
        ]

        reporter.report_summary(metrics_list)
        result = output.getvalue()

        assert "Avg Precision@10" in result
        assert "2 queries evaluated" in result
        # Average precision: (0.82 + 0.78) / 2 = 0.80
        assert "0.80" in result

    def test_report_summary_empty_list(self, reporter: ConsoleReporter, output: StringIO) -> None:
        """Summary handles empty list gracefully."""
        reporter.report_summary([])
        result = output.getvalue()

        assert "No metrics to report" in result

    def test_print_header(self, reporter: ConsoleReporter, output: StringIO) -> None:
        """Header prints with decoration."""
        reporter.print_header("Test Header")
        result = output.getvalue()

        assert "Test Header" in result
        assert "=" in result

    def test_print_success(self, reporter: ConsoleReporter, output: StringIO) -> None:
        """Success message includes checkmark."""
        reporter.print_success("Operation completed")
        result = output.getvalue()

        assert "✅" in result
        assert "Operation completed" in result

    def test_print_warning(self, reporter: ConsoleReporter, output: StringIO) -> None:
        """Warning message includes warning emoji."""
        reporter.print_warning("Low score detected")
        result = output.getvalue()

        assert "⚠️" in result
        assert "Low score detected" in result

    def test_print_error(self, reporter: ConsoleReporter, output: StringIO) -> None:
        """Error message includes X emoji."""
        reporter.print_error("Evaluation failed")
        result = output.getvalue()

        assert "❌" in result
        assert "Evaluation failed" in result

    def test_print_info(self, reporter: ConsoleReporter, output: StringIO) -> None:
        """Info message includes info emoji."""
        reporter.print_info("Additional details")
        result = output.getvalue()

        assert "[i]" in result
        assert "Additional details" in result


class TestSupportsColor:
    """Tests for color support detection."""

    def test_stringio_no_color(self) -> None:
        """StringIO does not support colors."""
        stream = StringIO()
        assert _supports_color(stream) is False

    def test_no_isatty_attribute(self) -> None:
        """Object without isatty returns False."""

        class FakeStream:
            pass

        assert _supports_color(FakeStream()) is False  # type: ignore[arg-type]
