"""Tests for JSON reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
from ragnarok_ai.reporters.console import Threshold
from ragnarok_ai.reporters.json import JSONReporter


class TestJSONReporterInit:
    """Tests for JSONReporter initialization."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        reporter = JSONReporter()

        assert reporter.indent == 2
        assert "precision" in reporter.thresholds
        assert "recall" in reporter.thresholds

    def test_custom_indent(self) -> None:
        """Custom indent is respected."""
        reporter = JSONReporter(indent=4)

        assert reporter.indent == 4

    def test_compact_output(self) -> None:
        """None indent produces compact output."""
        reporter = JSONReporter(indent=None)

        assert reporter.indent is None

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        custom = {"precision": Threshold(good=0.9, warning=0.7)}
        reporter = JSONReporter(thresholds=custom)

        assert reporter.thresholds["precision"].good == 0.9
        assert reporter.thresholds["precision"].warning == 0.7


class TestJSONReporterGetStatus:
    """Tests for status indicator logic."""

    @pytest.fixture
    def reporter(self) -> JSONReporter:
        """Default JSON reporter."""
        return JSONReporter()

    def test_pass_status(self, reporter: JSONReporter) -> None:
        """High values get pass status."""
        status = reporter._get_status("precision", 0.85)

        assert status == "pass"

    def test_warn_status(self, reporter: JSONReporter) -> None:
        """Medium values get warn status."""
        status = reporter._get_status("precision", 0.65)

        assert status == "warn"

    def test_fail_status(self, reporter: JSONReporter) -> None:
        """Low values get fail status."""
        status = reporter._get_status("precision", 0.4)

        assert status == "fail"

    def test_hallucination_inverted(self, reporter: JSONReporter) -> None:
        """Hallucination metric has inverted thresholds."""
        # Low hallucination is pass
        status_low = reporter._get_status("hallucination", 0.1)
        assert status_low == "pass"

        # High hallucination is fail
        status_high = reporter._get_status("hallucination", 0.5)
        assert status_high == "fail"


class TestJSONReporterReport:
    """Tests for report method."""

    @pytest.fixture
    def reporter(self) -> JSONReporter:
        """Default JSON reporter."""
        return JSONReporter()

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

    def test_valid_json(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics) -> None:
        """Report produces valid JSON."""
        json_str = reporter.report(sample_metrics)
        data = json.loads(json_str)

        assert isinstance(data, dict)

    def test_contains_timestamp(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics) -> None:
        """Report contains ISO 8601 timestamp."""
        json_str = reporter.report(sample_metrics)
        data = json.loads(json_str)

        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO 8601 format

    def test_contains_metrics(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics) -> None:
        """Report contains all metrics with values and status."""
        json_str = reporter.report(sample_metrics)
        data = json.loads(json_str)

        assert "metrics" in data
        metrics = data["metrics"]

        assert "precision" in metrics
        assert metrics["precision"]["value"] == 0.82
        assert metrics["precision"]["status"] == "pass"

        assert "recall" in metrics
        assert metrics["recall"]["value"] == 0.74
        assert metrics["recall"]["status"] == "warn"

        assert "mrr" in metrics
        assert "ndcg" in metrics

    def test_contains_k(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics) -> None:
        """Report contains k value."""
        json_str = reporter.report(sample_metrics)
        data = json.loads(json_str)

        assert data["k"] == 10

    def test_contains_metadata(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics) -> None:
        """Report includes metadata when provided."""
        metadata = {"query": "test query", "source": "unit_test"}
        json_str = reporter.report(sample_metrics, metadata=metadata)
        data = json.loads(json_str)

        assert data["metadata"]["query"] == "test query"
        assert data["metadata"]["source"] == "unit_test"

    def test_empty_metadata_by_default(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics) -> None:
        """Metadata is empty dict by default."""
        json_str = reporter.report(sample_metrics)
        data = json.loads(json_str)

        assert data["metadata"] == {}

    def test_compact_output(self, sample_metrics: RetrievalMetrics) -> None:
        """Compact output has no newlines."""
        reporter = JSONReporter(indent=None)
        json_str = reporter.report(sample_metrics)

        assert "\n" not in json_str


class TestJSONReporterReportToFile:
    """Tests for report_to_file method."""

    @pytest.fixture
    def reporter(self) -> JSONReporter:
        """Default JSON reporter."""
        return JSONReporter()

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

    def test_writes_file(self, reporter: JSONReporter, sample_metrics: RetrievalMetrics, tmp_path: Path) -> None:
        """Report is written to file."""
        output_file = tmp_path / "results.json"
        reporter.report_to_file(sample_metrics, output_file)

        assert output_file.exists()

    def test_file_contains_valid_json(
        self, reporter: JSONReporter, sample_metrics: RetrievalMetrics, tmp_path: Path
    ) -> None:
        """Written file contains valid JSON."""
        output_file = tmp_path / "results.json"
        reporter.report_to_file(sample_metrics, output_file)

        content = output_file.read_text()
        data = json.loads(content)

        assert data["metrics"]["precision"]["value"] == 0.82

    def test_creates_parent_directories(
        self, reporter: JSONReporter, sample_metrics: RetrievalMetrics, tmp_path: Path
    ) -> None:
        """Parent directories are created if they don't exist."""
        output_file = tmp_path / "nested" / "dir" / "results.json"
        reporter.report_to_file(sample_metrics, output_file)

        assert output_file.exists()

    def test_accepts_string_path(
        self, reporter: JSONReporter, sample_metrics: RetrievalMetrics, tmp_path: Path
    ) -> None:
        """String paths are accepted."""
        output_file = str(tmp_path / "results.json")
        reporter.report_to_file(sample_metrics, output_file)

        assert Path(output_file).exists()


class TestJSONReporterSummary:
    """Tests for report_summary method."""

    @pytest.fixture
    def reporter(self) -> JSONReporter:
        """Default JSON reporter."""
        return JSONReporter()

    @pytest.fixture
    def sample_metrics_list(self) -> list[RetrievalMetrics]:
        """Sample metrics list for testing."""
        return [
            RetrievalMetrics(precision=0.82, recall=0.74, mrr=0.91, ndcg=0.85, k=10),
            RetrievalMetrics(precision=0.78, recall=0.66, mrr=0.85, ndcg=0.81, k=10),
            RetrievalMetrics(precision=0.90, recall=0.80, mrr=1.00, ndcg=0.92, k=10),
        ]

    def test_valid_json(self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics]) -> None:
        """Summary produces valid JSON."""
        json_str = reporter.report_summary(sample_metrics_list)
        data = json.loads(json_str)

        assert isinstance(data, dict)

    def test_contains_summary(self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics]) -> None:
        """Summary contains aggregated data."""
        json_str = reporter.report_summary(sample_metrics_list)
        data = json.loads(json_str)

        assert "summary" in data
        summary = data["summary"]

        assert summary["num_queries"] == 3
        assert summary["k"] == 10

    def test_contains_averages(self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics]) -> None:
        """Summary contains average values with status."""
        json_str = reporter.report_summary(sample_metrics_list)
        data = json.loads(json_str)

        averages = data["summary"]["averages"]

        # Average precision: (0.82 + 0.78 + 0.90) / 3 = 0.833...
        assert averages["precision"]["value"] == pytest.approx(0.833, rel=0.01)
        assert averages["precision"]["status"] == "pass"

    def test_contains_ranges(self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics]) -> None:
        """Summary contains min/max ranges."""
        json_str = reporter.report_summary(sample_metrics_list)
        data = json.loads(json_str)

        ranges = data["summary"]["ranges"]

        assert ranges["precision"]["min"] == 0.78
        assert ranges["precision"]["max"] == 0.90
        assert ranges["mrr"]["min"] == 0.85
        assert ranges["mrr"]["max"] == 1.00

    def test_contains_individual_results(
        self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics]
    ) -> None:
        """Summary contains individual results."""
        json_str = reporter.report_summary(sample_metrics_list)
        data = json.loads(json_str)

        assert "results" in data
        assert len(data["results"]) == 3
        assert data["results"][0]["precision"] == 0.82

    def test_empty_list(self, reporter: JSONReporter) -> None:
        """Empty list is handled gracefully."""
        json_str = reporter.report_summary([])
        data = json.loads(json_str)

        assert data["summary"] is None
        assert data["results"] == []

    def test_summary_with_metadata(self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics]) -> None:
        """Summary includes metadata when provided."""
        metadata = {"experiment": "test_run_1"}
        json_str = reporter.report_summary(sample_metrics_list, metadata=metadata)
        data = json.loads(json_str)

        assert data["metadata"]["experiment"] == "test_run_1"


class TestJSONReporterSummaryToFile:
    """Tests for report_summary_to_file method."""

    @pytest.fixture
    def reporter(self) -> JSONReporter:
        """Default JSON reporter."""
        return JSONReporter()

    @pytest.fixture
    def sample_metrics_list(self) -> list[RetrievalMetrics]:
        """Sample metrics list for testing."""
        return [
            RetrievalMetrics(precision=0.82, recall=0.74, mrr=0.91, ndcg=0.85, k=10),
            RetrievalMetrics(precision=0.78, recall=0.66, mrr=0.85, ndcg=0.81, k=10),
        ]

    def test_writes_file(
        self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics], tmp_path: Path
    ) -> None:
        """Summary is written to file."""
        output_file = tmp_path / "summary.json"
        reporter.report_summary_to_file(sample_metrics_list, output_file)

        assert output_file.exists()

    def test_file_contains_valid_summary(
        self, reporter: JSONReporter, sample_metrics_list: list[RetrievalMetrics], tmp_path: Path
    ) -> None:
        """Written file contains valid summary JSON."""
        output_file = tmp_path / "summary.json"
        reporter.report_summary_to_file(sample_metrics_list, output_file)

        content = output_file.read_text()
        data = json.loads(content)

        assert data["summary"]["num_queries"] == 2
