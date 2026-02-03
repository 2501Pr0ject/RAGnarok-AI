"""Tests for CLI commands."""

from __future__ import annotations

import os

from typer.testing import CliRunner

from ragnarok_ai import __version__
from ragnarok_ai.cli.main import app

# Disable rich/typer color output to avoid ANSI escape codes in test assertions
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

runner = CliRunner()


class TestVersionCommand:
    """Tests for version command."""

    def test_version_command(self) -> None:
        """Version command shows version."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_version_flag(self) -> None:
        """--version flag shows version and exits."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_version_short_flag(self) -> None:
        """-v flag shows version and exits."""
        result = runner.invoke(app, ["-v"])

        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_no_args_shows_error(self) -> None:
        """Evaluate without args shows error."""
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code == 2  # Bad input
        # Error message goes to stderr, combined output in result.output
        assert "Either --demo or --testset is required" in result.output

    def test_evaluate_help(self) -> None:
        """Evaluate --help shows usage."""
        result = runner.invoke(app, ["evaluate", "--help"])

        assert result.exit_code == 0
        assert "Evaluate a RAG pipeline" in result.stdout
        assert "--demo" in result.stdout
        assert "--testset" in result.stdout
        assert "--output" in result.stdout
        assert "--fail-under" in result.stdout

    def test_evaluate_demo(self) -> None:
        """Evaluate --demo runs demo evaluation."""
        result = runner.invoke(app, ["evaluate", "--demo", "--limit", "2"])

        assert result.exit_code == 0
        assert "RAGnarok-AI Demo Evaluation" in result.stdout
        assert "Precision@10" in result.stdout
        assert "Recall@10" in result.stdout

    def test_evaluate_demo_json(self) -> None:
        """Evaluate --demo with --json outputs JSON."""
        result = runner.invoke(app, ["--json", "evaluate", "--demo", "--limit", "2"])

        assert result.exit_code == 0
        assert '"dataset": "novatech"' in result.stdout
        assert '"precision@10"' in result.stdout

    def test_evaluate_demo_fail_under_pass(self) -> None:
        """Evaluate --demo --fail-under passes when above threshold."""
        result = runner.invoke(app, ["evaluate", "--demo", "--limit", "2", "--fail-under", "0.3"])

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    def test_evaluate_demo_fail_under_fail(self) -> None:
        """Evaluate --demo --fail-under fails when below threshold."""
        result = runner.invoke(app, ["evaluate", "--demo", "--limit", "2", "--fail-under", "0.99"])

        assert result.exit_code == 1
        assert "FAIL" in result.stdout


class TestGenerateCommand:
    """Tests for generate command."""

    def test_generate_requires_docs_or_demo(self) -> None:
        """Generate command requires --docs or --demo."""
        result = runner.invoke(app, ["generate"])

        assert result.exit_code == 2  # Bad input
        assert "Either --docs or --demo is required" in result.output

    def test_generate_help(self) -> None:
        """Generate --help shows usage."""
        result = runner.invoke(app, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate a synthetic test set" in result.output
        assert "--docs" in result.output
        assert "--num" in result.output
        assert "--demo" in result.output
        assert "--model" in result.output
        assert "--seed" in result.output
        assert "--dry-run" in result.output

    def test_generate_invalid_docs_path(self) -> None:
        """Generate with non-existent docs path shows error."""
        result = runner.invoke(app, ["generate", "--docs", "/nonexistent/path"])

        assert result.exit_code == 2  # Bad input (file not found)
        assert "Path not found" in result.output

    def test_generate_dry_run(self) -> None:
        """Generate --dry-run shows what would be generated."""
        result = runner.invoke(app, ["generate", "--demo", "--dry-run", "--seed", "42"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "42" in result.output  # seed should be shown


class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_requires_option(self) -> None:
        """Benchmark command requires --demo, --history, or --list."""
        result = runner.invoke(app, ["benchmark"])

        assert result.exit_code == 2  # Bad input
        assert "Specify --demo, --history <config>, or --list" in result.output

    def test_benchmark_help(self) -> None:
        """Benchmark --help shows usage."""
        result = runner.invoke(app, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "Track benchmark history" in result.output
        assert "--demo" in result.output
        assert "--history" in result.output
        assert "--list" in result.output
        assert "--fail-under" in result.output
        assert "--dry-run" in result.output

    def test_benchmark_list_empty(self) -> None:
        """Benchmark --list with no history shows empty message."""
        result = runner.invoke(app, ["benchmark", "--list", "--storage", "/tmp/nonexistent.json"])

        assert result.exit_code == 0
        assert "No benchmark history found" in result.output

    def test_benchmark_dry_run(self) -> None:
        """Benchmark --dry-run shows what would be benchmarked."""
        result = runner.invoke(app, ["benchmark", "--demo", "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Run 1 (Baseline)" in result.output


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_no_args_shows_help(self) -> None:
        """No arguments shows help."""
        result = runner.invoke(app, [])

        # Typer shows help when no arguments provided
        # Exit code varies by Typer/Python version (0 or 2)
        assert result.exit_code in (0, 2)
        assert "Usage" in result.stdout

    def test_help_flag(self) -> None:
        """--help shows help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Local-first RAG evaluation" in result.stdout
        assert "--json" in result.stdout
        assert "--no-color" in result.stdout

    def test_json_flag_available(self) -> None:
        """--json flag is recognized."""
        result = runner.invoke(app, ["--json", "version"])

        assert result.exit_code == 0

    def test_no_color_flag_available(self) -> None:
        """--no-color flag is recognized."""
        result = runner.invoke(app, ["--no-color", "version"])

        assert result.exit_code == 0

    def test_pii_mode_flag_available(self) -> None:
        """--pii-mode flag is recognized."""
        result = runner.invoke(app, ["--pii-mode", "hash", "version"])

        assert result.exit_code == 0

    def test_pii_mode_accepts_all_values(self) -> None:
        """--pii-mode accepts hash, redact, and full."""
        for mode in ["hash", "redact", "full"]:
            result = runner.invoke(app, ["--pii-mode", mode, "version"])
            assert result.exit_code == 0, f"Failed for mode: {mode}"

    def test_pii_mode_shown_in_help(self) -> None:
        """--pii-mode is shown in help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "--pii-mode" in result.stdout
