"""Tests for real-testset CLI evaluation, live display, and the results viewer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragnarok_ai.cli.live import LiveEvaluationDisplay, RunningStats
from ragnarok_ai.cli.main import _load_pipeline, app

os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

runner = CliRunner()

FAKE_RAG_MODULE = """
from ragnarok_ai.core.types import Document, RAGResponse

DOCS = {f"d{i}": Document(id=f"d{i}", content=f"doc {i}") for i in range(1, 6)}
ANSWERS = {
    "What is CHF?": ("Congestive heart failure.", ["d1", "d5"]),
    "What is MI?": ("Myocardial infarction.", ["d5", "d2"]),
    "What causes hypertension?": ("Many factors.", ["d5"]),
    "What is an EKG?": ("An electrocardiogram.", ["d4"]),
}

class FakeRAG:
    async def query(self, question: str) -> RAGResponse:
        answer, doc_ids = ANSWERS[question]
        return RAGResponse(answer=answer, retrieved_docs=[DOCS[d] for d in doc_ids])

def build_rag() -> FakeRAG:
    return FakeRAG()

pipeline = FakeRAG()
not_a_pipeline = 42
"""

TESTSET = {
    "queries": [
        {"text": "What is CHF?", "ground_truth_docs": ["d1"]},
        {"text": "What is MI?", "ground_truth_docs": ["d2"]},
        {"text": "What causes hypertension?", "ground_truth_docs": ["d3"]},
        {"text": "What is an EKG?", "ground_truth_docs": ["d4"]},
    ]
}


@pytest.fixture
def rag_env(tmp_path: Path) -> dict[str, str]:
    """Write a fake RAG module + testset and put the module on sys.path."""
    (tmp_path / "cli_fakerag.py").write_text(FAKE_RAG_MODULE)
    testset_path = tmp_path / "testset.json"
    testset_path.write_text(json.dumps(TESTSET))
    sys.path.insert(0, str(tmp_path))
    yield {"testset": str(testset_path), "tmp": str(tmp_path)}
    sys.path.remove(str(tmp_path))
    sys.modules.pop("cli_fakerag", None)


class TestPipelineLoading:
    """Test suite for _load_pipeline."""

    def test_loads_object_and_factory(self, rag_env: dict[str, str]) -> None:  # noqa: ARG002
        assert hasattr(_load_pipeline("cli_fakerag:pipeline"), "query")
        assert hasattr(_load_pipeline("cli_fakerag:build_rag"), "query")

    def test_bad_spec_shapes(self, rag_env: dict[str, str]) -> None:  # noqa: ARG002
        with pytest.raises(ValueError, match="module:attribute"):
            _load_pipeline("no-colon")
        with pytest.raises(ValueError, match="Cannot import"):
            _load_pipeline("does_not_exist_xyz:thing")
        with pytest.raises(ValueError, match="no attribute"):
            _load_pipeline("cli_fakerag:missing")
        with pytest.raises(ValueError, match="query"):
            _load_pipeline("cli_fakerag:not_a_pipeline")


class TestEvaluateTestset:
    """Test suite for ragnarok evaluate --testset --pipeline."""

    def test_evaluates_and_writes_per_query_results(self, rag_env: dict[str, str]) -> None:
        out = Path(rag_env["tmp"]) / "results.json"
        result = runner.invoke(
            app,
            ["evaluate", "--testset", rag_env["testset"], "--pipeline", "cli_fakerag:pipeline", "--output", str(out)],
        )

        assert result.exit_code == 0, result.output
        assert "Results Summary" in result.output
        data = json.loads(out.read_text())
        assert data["queries_evaluated"] == 4
        assert len(data["queries"]) == 4
        assert data["pipeline"] == "cli_fakerag:pipeline"
        assert {"query", "precision", "recall", "mrr", "ndcg", "answer"} <= set(data["queries"][0])

    def test_streams_per_query_lines(self, rag_env: dict[str, str]) -> None:
        result = runner.invoke(
            app,
            ["evaluate", "--testset", rag_env["testset"], "--pipeline", "cli_fakerag:pipeline"],
        )

        assert result.exit_code == 0
        # Non-TTY fallback: one line per query with running average
        assert "1/4" in result.output
        assert "4/4" in result.output
        assert "avg=" in result.output

    def test_fail_under_gates_exit_code(self, rag_env: dict[str, str]) -> None:
        result = runner.invoke(
            app,
            ["evaluate", "--testset", rag_env["testset"], "--pipeline", "cli_fakerag:pipeline", "--fail-under", "0.9"],
        )

        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_json_mode(self, rag_env: dict[str, str]) -> None:
        result = runner.invoke(
            app,
            ["--json", "evaluate", "--testset", rag_env["testset"], "--pipeline", "cli_fakerag:pipeline"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "pass"
        assert payload["data"]["metrics"]["average"] > 0

    def test_missing_pipeline_is_a_readable_error(self, rag_env: dict[str, str]) -> None:
        result = runner.invoke(app, ["evaluate", "--testset", rag_env["testset"]])

        assert result.exit_code != 0
        assert "--pipeline" in result.output

    def test_missing_testset_file_is_a_readable_error(self, rag_env: dict[str, str]) -> None:  # noqa: ARG002
        result = runner.invoke(
            app,
            ["evaluate", "--testset", "nope.json", "--pipeline", "cli_fakerag:pipeline"],
        )

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_limit_caps_queries(self, rag_env: dict[str, str]) -> None:
        result = runner.invoke(
            app,
            ["--json", "evaluate", "--testset", rag_env["testset"], "--pipeline", "cli_fakerag:pipeline", "-n", "2"],
        )

        assert json.loads(result.output)["data"]["queries_evaluated"] == 2


class TestRunningStats:
    """Test suite for the streaming averages."""

    def test_running_average_matches_batch_average(self) -> None:
        stats = RunningStats()
        values = [(1.0, 0.5, 0.8, 0.9), (0.0, 0.5, 0.2, 0.1), (0.5, 1.0, 0.5, 0.5)]
        for p, r, m, n in values:
            stats.add(p, r, m, n)

        assert stats.count == 3
        assert stats.precision == pytest.approx(0.5)
        assert stats.recall == pytest.approx(2 / 3)
        assert stats.passed == 1  # 1.0
        assert stats.warned == 1  # 0.5
        assert stats.failed == 1  # 0.0

    def test_display_records_without_tty(self) -> None:
        from rich.console import Console

        with Path(os.devnull).open("w") as sink:
            console = Console(file=sink, force_terminal=False)
            display = LiveEvaluationDisplay(title="t", total=2, console=console)
            with display:
                display.record("q1", 1.0, 1.0, 1.0, 1.0)
                display.record("q2", 0.0, 0.0, 0.0, 0.0)

        assert display.stats.count == 2
        assert display.stats.average == pytest.approx(0.5)


class TestViewer:
    """Test suite for ragnarok view and the results loader."""

    def _results_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "testset": "t",
                    "queries_evaluated": 2,
                    "metrics": {"average": 0.5},
                    "queries": [
                        {"query": "good", "precision": 0.9, "recall": 1.0, "mrr": 1.0, "ndcg": 1.0, "answer": "a"},
                        {"query": "bad", "precision": 0.1, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "answer": ""},
                    ],
                }
            )
        )
        return path

    def test_load_results_validates(self, tmp_path: Path) -> None:
        from ragnarok_ai.cli.tui import load_results

        assert load_results(self._results_file(tmp_path))["queries_evaluated"] == 2

        with pytest.raises(ValueError, match="not found"):
            load_results(tmp_path / "missing.json")

        bad = tmp_path / "bad.json"
        bad.write_text('{"metrics": {}}')
        with pytest.raises(ValueError, match="no per-query results"):
            load_results(bad)

    def test_view_rejects_invalid_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{}")
        result = runner.invoke(app, ["view", str(bad)])

        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_app_filters_failing_rows(self, tmp_path: Path) -> None:
        from textual.widgets import DataTable

        from ragnarok_ai.cli.tui import ResultsApp, load_results

        app_ = ResultsApp(load_results(self._results_file(tmp_path)))
        async with app_.run_test() as pilot:
            table = app_.query_one(DataTable)
            assert table.row_count == 2
            await pilot.press("f")
            assert table.row_count == 1
            await pilot.press("a")
            assert table.row_count == 2
