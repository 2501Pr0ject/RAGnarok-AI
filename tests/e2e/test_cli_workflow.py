"""End-to-end tests for CLI workflow.

Tests the full generate → evaluate → benchmark pipeline using demo/dry-run modes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragnarok_ai.cli.main import app

runner = CliRunner()


class TestCLIWorkflow:
    """E2E tests for the CLI workflow."""

    def test_generate_dry_run(self) -> None:
        """Test generate command in dry-run mode."""
        result = runner.invoke(
            app, ["generate", "--demo", "--dry-run", "--seed", "42", "--num", "5"]
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "5 questions" in result.output
        assert "42" in result.output

    def test_evaluate_demo(self) -> None:
        """Test evaluate command with demo dataset."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                app, ["evaluate", "--demo", "--limit", "3", "--output", output_path]
            )

            assert result.exit_code == 0
            assert "RAGnarok-AI Demo Evaluation" in result.output
            assert "Results Summary" in result.output

            # Check output file
            output_data = json.loads(Path(output_path).read_text())
            assert output_data["status"] == "pass"
            assert "metrics" in output_data
            assert output_data["queries_evaluated"] == 3
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_benchmark_demo(self) -> None:
        """Test benchmark command with demo mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"
            output_path = Path(tmpdir) / "results.json"

            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "--demo",
                    "--storage",
                    str(storage_path),
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert "Comparison Table" in result.output
            assert "Run 1 (Baseline)" in result.output
            assert "REGRESSION DETECTED" in result.output

            # Check output file
            output_data = json.loads(output_path.read_text())
            assert output_data["status"] == "success"
            assert "runs" in output_data
            assert len(output_data["runs"]) == 3
            assert "best" in output_data
            assert "worst" in output_data

            # Check storage file
            assert storage_path.exists()

    def test_benchmark_fail_under(self) -> None:
        """Test benchmark --fail-under exits with code 1 when threshold not met."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # The demo worst run has average ~0.64, so fail-under 0.7 should fail
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "--demo",
                    "--storage",
                    str(storage_path),
                    "--fail-under",
                    "0.7",
                ],
            )

            assert result.exit_code == 1
            assert "FAIL" in result.output

    def test_full_workflow_json_output(self) -> None:
        """Test full workflow with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # Generate (dry-run with JSON)
            result = runner.invoke(
                app, ["--json", "generate", "--demo", "--dry-run", "--seed", "42"]
            )
            assert result.exit_code == 0
            gen_data = json.loads(result.output)
            assert gen_data["status"] == "dry_run"
            assert gen_data["seed"] == 42

            # Evaluate (demo with JSON)
            result = runner.invoke(
                app, ["--json", "evaluate", "--demo", "--limit", "2"]
            )
            assert result.exit_code == 0
            eval_data = json.loads(result.output)
            assert eval_data["status"] == "pass"
            assert "metrics" in eval_data

            # Benchmark (demo with JSON)
            result = runner.invoke(
                app,
                ["--json", "benchmark", "--demo", "--storage", str(storage_path)],
            )
            assert result.exit_code == 0
            bench_data = json.loads(result.output)
            assert bench_data["status"] == "success"
            assert len(bench_data["runs"]) == 3


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests requiring more setup."""

    def test_benchmark_history_after_demo(self) -> None:
        """Test that benchmark history is populated after demo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # Run demo first
            runner.invoke(
                app, ["benchmark", "--demo", "--storage", str(storage_path)]
            )

            # Now check list
            result = runner.invoke(
                app, ["benchmark", "--list", "--storage", str(storage_path)]
            )

            assert result.exit_code == 0
            assert "demo-config" in result.output
            assert "Runs: 3" in result.output

            # Check history
            result = runner.invoke(
                app,
                ["benchmark", "--history", "demo-config", "--storage", str(storage_path)],
            )

            assert result.exit_code == 0
            assert "Benchmark History: demo-config" in result.output
            assert "BASELINE" in result.output
