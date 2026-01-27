"""End-to-end tests for CLI workflow.

Tests the full generate → evaluate → benchmark pipeline using demo/dry-run modes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragnarok_ai import __version__
from ragnarok_ai.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
class TestCLIWorkflow:
    """E2E tests for the CLI workflow."""

    def test_generate_dry_run(self) -> None:
        """Test generate command in dry-run mode."""
        result = runner.invoke(app, ["--json", "generate", "--demo", "--dry-run", "--seed", "42", "--num", "5"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Structural assertions (improvement #2, #3)
        assert data["command"] == "generate"
        assert data["status"] == "dry_run"
        assert data["version"] == __version__
        assert data["data"]["seed"] == 42
        assert data["data"]["config"]["num_questions"] == 5

    def test_evaluate_demo(self) -> None:
        """Test evaluate command with demo dataset."""
        # Improvement #1: Use TemporaryDirectory instead of NamedTemporaryFile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            result = runner.invoke(app, ["evaluate", "--demo", "--limit", "3", "--output", str(output_path)])

            assert result.exit_code == 0
            # Keep one cosmetic assertion (improvement #2)
            assert "RAGnarok-AI Demo Evaluation" in result.output

            # Structural assertions on output file
            output_data = json.loads(output_path.read_text())
            assert output_data["status"] == "pass"
            assert "metrics" in output_data
            assert output_data["queries_evaluated"] == 3

    def test_evaluate_demo_json(self) -> None:
        """Test evaluate --demo with JSON output has standardized envelope."""
        result = runner.invoke(app, ["--json", "evaluate", "--demo", "--limit", "2"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Improvement #3: Standardized JSON envelope
        assert data["command"] == "evaluate"
        assert data["status"] == "pass"
        assert data["version"] == __version__
        assert "data" in data
        assert "metrics" in data["data"]
        assert data["data"]["queries_evaluated"] == 2

    def test_benchmark_demo(self) -> None:
        """Test benchmark command with demo mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"
            output_path = Path(tmpdir) / "results.json"

            result = runner.invoke(
                app,
                [
                    "--json",
                    "benchmark",
                    "--demo",
                    "--storage",
                    str(storage_path),
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)

            # Improvement #3: Standardized JSON envelope
            assert data["command"] == "benchmark"
            assert data["status"] == "success"
            assert data["version"] == __version__

            # Structural assertions (improvement #2)
            assert "data" in data
            assert len(data["data"]["runs"]) == 3
            assert "best" in data["data"]
            assert "worst" in data["data"]

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
                    "--json",
                    "benchmark",
                    "--demo",
                    "--storage",
                    str(storage_path),
                    "--fail-under",
                    "0.7",
                ],
            )

            # Improvement #4: Verify exit code
            assert result.exit_code == 1
            data = json.loads(result.output)
            assert data["status"] == "fail"
            assert "fail_reason" in data["data"]

    def test_full_workflow_json_output(self) -> None:
        """Test full workflow with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # Generate (dry-run with JSON)
            result = runner.invoke(app, ["--json", "generate", "--demo", "--dry-run", "--seed", "42"])
            assert result.exit_code == 0
            gen_data = json.loads(result.output)
            assert gen_data["command"] == "generate"
            assert gen_data["status"] == "dry_run"
            assert gen_data["version"] == __version__

            # Evaluate (demo with JSON)
            result = runner.invoke(app, ["--json", "evaluate", "--demo", "--limit", "2"])
            assert result.exit_code == 0
            eval_data = json.loads(result.output)
            assert eval_data["command"] == "evaluate"
            assert eval_data["status"] == "pass"
            assert "metrics" in eval_data["data"]

            # Benchmark (demo with JSON)
            result = runner.invoke(
                app,
                ["--json", "benchmark", "--demo", "--storage", str(storage_path)],
            )
            assert result.exit_code == 0
            bench_data = json.loads(result.output)
            assert bench_data["command"] == "benchmark"
            assert bench_data["status"] == "success"
            assert len(bench_data["data"]["runs"]) == 3


@pytest.mark.e2e
class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_bad_input_exits_with_code_2(self) -> None:
        """Test that invalid input shows usage error (exit code 2)."""
        # Improvement #4: Test bad input → exit_code=2
        result = runner.invoke(app, ["evaluate", "--limit", "invalid"])

        # Typer returns 2 for usage errors
        assert result.exit_code == 2
        assert "Invalid value" in result.output or "error" in result.output.lower()

    def test_missing_required_option_exits_with_code_1(self) -> None:
        """Test that missing required option exits with code 1."""
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code == 1
        assert "Either --demo or --testset is required" in result.output

    def test_json_output_on_error(self) -> None:
        """Test that --json produces valid JSON even on error."""
        # Improvement #8: JSON valid even on error
        result = runner.invoke(app, ["--json", "evaluate"])

        # Should return error JSON, not traceback
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["command"] == "evaluate"
        assert data["status"] == "error"
        assert data["version"] == __version__
        assert len(data["errors"]) > 0

    def test_benchmark_json_on_error(self) -> None:
        """Test benchmark returns valid JSON on error."""
        result = runner.invoke(app, ["--json", "benchmark"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["status"] == "error"
        assert len(data["errors"]) > 0

    def test_generate_json_on_error(self) -> None:
        """Test generate returns valid JSON on error."""
        result = runner.invoke(app, ["--json", "generate"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["status"] == "error"
        assert len(data["errors"]) > 0


@pytest.mark.e2e
@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests requiring more setup."""

    def test_benchmark_history_after_demo(self) -> None:
        """Test that benchmark history is populated after demo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # Run demo first
            runner.invoke(app, ["benchmark", "--demo", "--storage", str(storage_path)])

            # Now check list with JSON
            result = runner.invoke(app, ["--json", "benchmark", "--list", "--storage", str(storage_path)])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["status"] == "success"
            assert any(c["name"] == "demo-config" for c in data["data"]["configs"])

            # Check history with JSON
            result = runner.invoke(
                app,
                ["--json", "benchmark", "--history", "demo-config", "--storage", str(storage_path)],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["status"] == "success"
            assert data["data"]["config"] == "demo-config"
            assert len(data["data"]["records"]) == 3

    def test_chained_workflow_generate_to_evaluate(self) -> None:
        """Test real workflow: generate artifact → evaluate with it.

        Improvement #6: Chain generate output to evaluate input.
        Note: Uses dry-run since actual generation requires Ollama.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"

            # Generate in dry-run - verify manifest structure
            result = runner.invoke(
                app,
                [
                    "--json",
                    "generate",
                    "--demo",
                    "--dry-run",
                    "--seed",
                    "42",
                    "--num",
                    "5",
                    "--output",
                    str(testset_path),
                ],
            )

            assert result.exit_code == 0
            gen_data = json.loads(result.output)
            assert gen_data["status"] == "dry_run"
            assert gen_data["data"]["config"]["num_questions"] == 5

            # For full E2E, we'd generate real testset and evaluate it
            # Since that requires Ollama, we verify the contract is correct:
            # - generate outputs testset.json + manifest.json
            # - evaluate accepts --testset pointing to generated file

            # Verify evaluate accepts the testset path format
            # Custom testset evaluation is not yet implemented, so it returns error
            result = runner.invoke(app, ["--json", "evaluate", "--testset", str(testset_path)])

            # Should return valid JSON with error (feature not implemented yet)
            assert result.exit_code == 1
            data = json.loads(result.output)
            assert data["command"] == "evaluate"
            assert data["status"] == "error"
            assert data["version"] == __version__
            assert len(data["errors"]) > 0
