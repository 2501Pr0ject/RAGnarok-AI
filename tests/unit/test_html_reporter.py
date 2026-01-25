"""Tests for HTML reporter."""

from __future__ import annotations

from pathlib import Path

from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
from ragnarok_ai.reporters.console import Threshold
from ragnarok_ai.reporters.html import HTMLReporter


class TestHTMLReporterInit:
    """Tests for HTMLReporter initialization."""

    def test_init_default_thresholds(self) -> None:
        """Reporter initializes with default thresholds."""
        reporter = HTMLReporter()

        assert "precision" in reporter.thresholds
        assert "recall" in reporter.thresholds
        assert "mrr" in reporter.thresholds
        assert "ndcg" in reporter.thresholds

    def test_init_custom_thresholds(self) -> None:
        """Reporter accepts custom thresholds."""
        custom = {"precision": Threshold(good=0.9, warning=0.7)}
        reporter = HTMLReporter(thresholds=custom)

        assert reporter.thresholds["precision"].good == 0.9
        assert reporter.thresholds["precision"].warning == 0.7


class TestHTMLReporterGetStatus:
    """Tests for status determination logic."""

    def test_get_status_pass(self) -> None:
        """Returns pass for good values."""
        reporter = HTMLReporter()

        assert reporter._get_status("precision", 0.85) == "pass"
        assert reporter._get_status("recall", 0.9) == "pass"

    def test_get_status_warn(self) -> None:
        """Returns warn for warning values."""
        reporter = HTMLReporter()

        assert reporter._get_status("precision", 0.65) == "warn"
        assert reporter._get_status("recall", 0.65) == "warn"

    def test_get_status_fail(self) -> None:
        """Returns fail for bad values."""
        reporter = HTMLReporter()

        assert reporter._get_status("precision", 0.3) == "fail"
        assert reporter._get_status("recall", 0.2) == "fail"

    def test_get_status_hallucination_inverted(self) -> None:
        """Hallucination uses inverted logic (lower is better)."""
        reporter = HTMLReporter()

        assert reporter._get_status("hallucination", 0.1) == "pass"
        assert reporter._get_status("hallucination", 0.3) == "warn"
        assert reporter._get_status("hallucination", 0.6) == "fail"


class TestHTMLReporterGetOverallStatus:
    """Tests for overall status determination."""

    def test_overall_status_all_pass(self) -> None:
        """Returns pass when all metrics pass."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.9, recall=0.9, mrr=0.9, ndcg=0.9, k=10)

        assert reporter._get_overall_status(metrics) == "pass"

    def test_overall_status_one_fail(self) -> None:
        """Returns fail when any metric fails."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.9, recall=0.2, mrr=0.9, ndcg=0.9, k=10)

        assert reporter._get_overall_status(metrics) == "fail"

    def test_overall_status_one_warn(self) -> None:
        """Returns warn when worst is warning."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.9, recall=0.6, mrr=0.9, ndcg=0.9, k=10)

        assert reporter._get_overall_status(metrics) == "warn"


class TestHTMLReporterEscapeHtml:
    """Tests for HTML escaping."""

    def test_escape_html_special_chars(self) -> None:
        """Escapes HTML special characters."""
        reporter = HTMLReporter()

        assert reporter._escape_html("<script>") == "&lt;script&gt;"
        assert reporter._escape_html("a & b") == "a &amp; b"
        assert reporter._escape_html('"quoted"') == "&quot;quoted&quot;"
        assert reporter._escape_html("it's") == "it&#x27;s"

    def test_escape_html_plain_text(self) -> None:
        """Leaves plain text unchanged."""
        reporter = HTMLReporter()

        assert reporter._escape_html("Hello World") == "Hello World"


class TestHTMLReporterReport:
    """Tests for report generation."""

    def test_report_generates_html(self) -> None:
        """Report generates valid HTML structure."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("What is the capital of France?", metrics, None)]

        html = reporter.report(results)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "RAGnarok Evaluation Report" in html

    def test_report_includes_metrics(self) -> None:
        """Report includes metric values."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.82, recall=0.75, mrr=0.68, ndcg=0.71, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert "0.82" in html
        assert "0.75" in html
        assert "0.68" in html
        assert "0.71" in html

    def test_report_includes_query(self) -> None:
        """Report includes query text."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("What is the capital of France?", metrics, None)]

        html = reporter.report(results)

        assert "What is the capital of France?" in html

    def test_report_escapes_query(self) -> None:
        """Report escapes HTML in query text."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("<script>alert('xss')</script>", metrics, None)]

        html = reporter.report(results)

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_report_includes_chunks(self) -> None:
        """Report includes chunk content when provided."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        chunks = [
            {"content": "Paris is the capital of France.", "score": 0.95},
            {"content": "France is in Europe.", "score": 0.82},
        ]
        results = [("What is the capital of France?", metrics, chunks)]

        html = reporter.report(results)

        assert "Paris is the capital of France." in html
        assert "France is in Europe." in html
        assert "0.9500" in html
        assert "0.8200" in html

    def test_report_empty_results(self) -> None:
        """Report handles empty results."""
        reporter = HTMLReporter()

        html = reporter.report([])

        assert "<!DOCTYPE html>" in html
        assert "No evaluation results to display" in html
        assert "0 queries" in html

    def test_report_multiple_results(self) -> None:
        """Report handles multiple results."""
        reporter = HTMLReporter()
        metrics1 = RetrievalMetrics(precision=0.9, recall=0.85, mrr=0.8, ndcg=0.82, k=10)
        metrics2 = RetrievalMetrics(precision=0.7, recall=0.65, mrr=0.6, ndcg=0.62, k=10)
        results = [
            ("Query 1", metrics1, None),
            ("Query 2", metrics2, None),
        ]

        html = reporter.report(results)

        assert "Query 1" in html
        assert "Query 2" in html
        assert "2 queries" in html

    def test_report_includes_status_badges(self) -> None:
        """Report includes status badges."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.9, recall=0.3, mrr=0.6, ndcg=0.8, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert 'class="status-badge' in html

    def test_report_includes_javascript(self) -> None:
        """Report includes JavaScript for interactivity."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert "<script>" in html
        assert "addEventListener" in html
        assert "filterRows" in html

    def test_report_includes_css(self) -> None:
        """Report includes embedded CSS."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert "<style>" in html
        assert "--bg-primary" in html


class TestHTMLReporterReportToFile:
    """Tests for file output."""

    def test_report_to_file_creates_file(self, tmp_path: Path) -> None:
        """Report creates output file."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]
        output_path = tmp_path / "report.html"

        reporter.report_to_file(results, output_path)

        assert output_path.exists()

    def test_report_to_file_content(self, tmp_path: Path) -> None:
        """Report file contains expected content."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]
        output_path = tmp_path / "report.html"

        reporter.report_to_file(results, output_path)

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Test query" in content

    def test_report_to_file_creates_directories(self, tmp_path: Path) -> None:
        """Report creates parent directories if needed."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]
        output_path = tmp_path / "nested" / "dir" / "report.html"

        reporter.report_to_file(results, output_path)

        assert output_path.exists()

    def test_report_to_file_accepts_string_path(self, tmp_path: Path) -> None:
        """Report accepts string path."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]
        output_path = str(tmp_path / "report.html")

        reporter.report_to_file(results, output_path)

        assert Path(output_path).exists()


class TestHTMLReporterReportSummary:
    """Tests for summary report generation."""

    def test_report_summary_generates_html(self) -> None:
        """Summary generates valid HTML."""
        reporter = HTMLReporter()
        metrics = [
            RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10),
            RetrievalMetrics(precision=0.9, recall=0.85, mrr=0.8, ndcg=0.82, k=10),
        ]

        html = reporter.report_summary(metrics)

        assert "<!DOCTYPE html>" in html
        assert "RAGnarok Evaluation Report" in html

    def test_report_summary_calculates_averages(self) -> None:
        """Summary shows average metrics."""
        reporter = HTMLReporter()
        metrics = [
            RetrievalMetrics(precision=0.8, recall=0.6, mrr=0.7, ndcg=0.7, k=10),
            RetrievalMetrics(precision=1.0, recall=0.8, mrr=0.9, ndcg=0.9, k=10),
        ]

        html = reporter.report_summary(metrics)

        # Average precision = (0.8 + 1.0) / 2 = 0.9
        assert "0.90" in html

    def test_report_summary_to_file(self, tmp_path: Path) -> None:
        """Summary can be written to file."""
        reporter = HTMLReporter()
        metrics = [
            RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10),
        ]
        output_path = tmp_path / "summary.html"

        reporter.report_summary_to_file(metrics, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content


class TestHTMLReporterMetricCards:
    """Tests for metric card generation."""

    def test_metric_cards_generated(self) -> None:
        """Metric cards are generated for all metrics."""
        reporter = HTMLReporter()
        metrics = [
            RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10),
        ]

        html = reporter.report_summary(metrics)

        assert "Precision" in html
        assert "Recall" in html
        assert "MRR" in html
        assert "NDCG" in html

    def test_metric_cards_show_status(self) -> None:
        """Metric cards show appropriate status."""
        reporter = HTMLReporter()
        metrics = [
            RetrievalMetrics(precision=0.9, recall=0.3, mrr=0.6, ndcg=0.85, k=10),
        ]

        html = reporter.report_summary(metrics)

        assert 'class="metric-card pass"' in html or 'class="metric-card fail"' in html


class TestHTMLReporterDrillDown:
    """Tests for drill-down functionality."""

    def test_drill_down_content_generated(self) -> None:
        """Drill-down content is generated for each row."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert 'class="drill-down"' in html
        assert 'class="drill-down-grid"' in html

    def test_drill_down_includes_detailed_metrics(self) -> None:
        """Drill-down shows detailed metric values."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8234, recall=0.7123, mrr=0.7567, ndcg=0.7289, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert "0.8234" in html
        assert "0.7123" in html
        assert "0.7567" in html
        assert "0.7289" in html

    def test_drill_down_expandable_rows(self) -> None:
        """Rows are marked as expandable."""
        reporter = HTMLReporter()
        metrics = RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.75, ndcg=0.72, k=10)
        results = [("Test query", metrics, None)]

        html = reporter.report(results)

        assert 'class="data-row expandable"' in html
