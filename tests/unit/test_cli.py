"""Tests for CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from ragnarok_ai import __version__
from ragnarok_ai.cli.main import app

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

    def test_evaluate_placeholder(self) -> None:
        """Evaluate command shows coming soon message."""
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code == 0
        assert "Coming Soon" in result.stdout

    def test_evaluate_help(self) -> None:
        """Evaluate --help shows usage."""
        result = runner.invoke(app, ["evaluate", "--help"])

        assert result.exit_code == 0
        assert "Evaluate a RAG pipeline" in result.stdout
        assert "--config" in result.stdout
        assert "--output" in result.stdout

    def test_evaluate_json_output(self) -> None:
        """Evaluate with --json outputs JSON."""
        result = runner.invoke(app, ["--json", "evaluate"])

        assert result.exit_code == 0
        assert "coming_soon" in result.stdout
        assert "{" in result.stdout


class TestGenerateCommand:
    """Tests for generate command."""

    def test_generate_placeholder(self) -> None:
        """Generate command shows coming soon message."""
        result = runner.invoke(app, ["generate"])

        assert result.exit_code == 0
        assert "Coming Soon" in result.stdout

    def test_generate_help(self) -> None:
        """Generate --help shows usage."""
        result = runner.invoke(app, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate a synthetic test set" in result.stdout
        assert "--docs" in result.stdout
        assert "--num" in result.stdout


class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_placeholder(self) -> None:
        """Benchmark command shows coming soon message."""
        result = runner.invoke(app, ["benchmark"])

        assert result.exit_code == 0
        assert "Coming Soon" in result.stdout

    def test_benchmark_help(self) -> None:
        """Benchmark --help shows usage."""
        result = runner.invoke(app, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "Benchmark multiple RAG configurations" in result.stdout


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_no_args_shows_help(self) -> None:
        """No arguments shows help."""
        result = runner.invoke(app, [])

        # Typer's no_args_is_help returns exit code 2 (standard for help)
        assert result.exit_code == 2
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
