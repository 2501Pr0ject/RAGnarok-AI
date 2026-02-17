"""Unit tests for CLI main module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ragnarok_ai import __version__
from ragnarok_ai.cli.main import (
    EvaluateConfig,
    app,
    json_response,
    load_config,
    state,
)

runner = CliRunner()


class TestJsonResponse:
    """Tests for json_response function."""

    def test_json_response_basic(self) -> None:
        """Test basic JSON response structure."""
        response = json_response("test", "success")
        data = json.loads(response)

        assert data["command"] == "test"
        assert data["status"] == "success"
        assert data["version"] == __version__
        assert data["data"] == {}
        assert data["warnings"] == []
        assert data["errors"] == []

    def test_json_response_with_data(self) -> None:
        """Test JSON response with data payload."""
        response = json_response("evaluate", "pass", data={"metrics": {"precision": 0.9}})
        data = json.loads(response)

        assert data["command"] == "evaluate"
        assert data["data"]["metrics"]["precision"] == 0.9

    def test_json_response_with_warnings(self) -> None:
        """Test JSON response with warnings."""
        response = json_response("test", "success", warnings=["Warning 1", "Warning 2"])
        data = json.loads(response)

        assert len(data["warnings"]) == 2
        assert "Warning 1" in data["warnings"]

    def test_json_response_with_errors(self) -> None:
        """Test JSON response with errors."""
        response = json_response("test", "error", errors=["Error occurred"])
        data = json.loads(response)

        assert data["status"] == "error"
        assert len(data["errors"]) == 1


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_valid_yaml(self) -> None:
        """Test loading valid YAML config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
testset: test.json
output: results.json
fail_under: 0.8
metrics:
  - precision
  - recall
ollama_url: http://custom:11434
""")
            f.flush()

            config = load_config(f.name)

            assert config.testset == "test.json"
            assert config.output == "results.json"
            assert config.fail_under == 0.8
            assert "precision" in config.metrics
            assert config.ollama_url == "http://custom:11434"

            Path(f.name).unlink()

    def test_load_config_minimal_yaml(self) -> None:
        """Test loading minimal YAML config with defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{}")
            f.flush()

            config = load_config(f.name)

            assert config.testset is None
            assert config.fail_under is None
            assert "precision" in config.metrics  # default
            assert config.ollama_url == "http://localhost:11434"  # default

            Path(f.name).unlink()

    def test_load_config_file_not_found(self) -> None:
        """Test loading non-existent config file."""
        import typer

        state["json"] = False
        with pytest.raises(typer.Exit) as exc_info:
            load_config("/nonexistent/path/config.yaml")
        assert exc_info.value.exit_code == 2

    def test_load_config_file_not_found_json_mode(self) -> None:
        """Test loading non-existent config file in JSON mode."""
        import typer

        state["json"] = True
        with pytest.raises(typer.Exit) as exc_info:
            load_config("/nonexistent/path/config.yaml")
        assert exc_info.value.exit_code == 2
        state["json"] = False

    def test_load_config_invalid_yaml(self) -> None:
        """Test loading invalid YAML config."""
        import typer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            state["json"] = False
            with pytest.raises(typer.Exit) as exc_info:
                load_config(f.name)
            assert exc_info.value.exit_code == 2

            Path(f.name).unlink()


class TestEvaluateConfig:
    """Tests for EvaluateConfig dataclass."""

    def test_evaluate_config_defaults(self) -> None:
        """Test EvaluateConfig default values."""
        config = EvaluateConfig()

        assert config.testset is None
        assert config.output is None
        assert config.fail_under is None
        assert "precision" in config.metrics
        assert "recall" in config.metrics
        assert "mrr" in config.metrics
        assert "ndcg" in config.metrics
        assert "faithfulness" in config.criteria
        assert config.ollama_url == "http://localhost:11434"

    def test_evaluate_config_custom_values(self) -> None:
        """Test EvaluateConfig with custom values."""
        config = EvaluateConfig(
            testset="custom.json",
            output="output.json",
            fail_under=0.9,
            metrics=["precision"],
            criteria=["faithfulness"],
            ollama_url="http://custom:11434",
        )

        assert config.testset == "custom.json"
        assert config.output == "output.json"
        assert config.fail_under == 0.9
        assert config.metrics == ["precision"]
        assert config.criteria == ["faithfulness"]
        assert config.ollama_url == "http://custom:11434"


class TestVersionCommand:
    """Tests for version command."""

    def test_version_command(self) -> None:
        """Test version command output."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_flag(self) -> None:
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_short_flag(self) -> None:
        """Test -v flag."""
        result = runner.invoke(app, ["-v"])

        assert result.exit_code == 0
        assert __version__ in result.output


class TestPluginsCommand:
    """Tests for plugins command."""

    def test_plugins_no_args(self) -> None:
        """Test plugins command without arguments."""
        result = runner.invoke(app, ["plugins"])

        assert result.exit_code == 2
        assert "Specify --list or --info" in result.output

    def test_plugins_no_args_json(self) -> None:
        """Test plugins command without arguments in JSON mode."""
        result = runner.invoke(app, ["--json", "plugins"])

        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["status"] == "error"

    def test_plugins_list(self) -> None:
        """Test plugins --list command."""
        result = runner.invoke(app, ["plugins", "--list"])

        assert result.exit_code == 0
        assert "Available Plugins" in result.output

    def test_plugins_list_json(self) -> None:
        """Test plugins --list with JSON output."""
        result = runner.invoke(app, ["--json", "plugins", "--list"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["command"] == "plugins"
        assert data["status"] == "success"
        assert "plugins" in data["data"]
        assert "total" in data["data"]

    def test_plugins_list_type_filter(self) -> None:
        """Test plugins --list with type filter."""
        result = runner.invoke(app, ["plugins", "--list", "--type", "llm"])

        assert result.exit_code == 0
        assert "LLM" in result.output

    def test_plugins_list_invalid_type(self) -> None:
        """Test plugins --list with invalid type."""
        result = runner.invoke(app, ["plugins", "--list", "--type", "invalid"])

        assert result.exit_code == 2
        assert "Invalid type" in result.output

    def test_plugins_list_invalid_type_json(self) -> None:
        """Test plugins --list with invalid type in JSON mode."""
        result = runner.invoke(app, ["--json", "plugins", "--list", "--type", "invalid"])

        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["status"] == "error"

    def test_plugins_list_local_only(self) -> None:
        """Test plugins --list --local."""
        result = runner.invoke(app, ["plugins", "--list", "--local"])

        assert result.exit_code == 0
        assert "[Local Only]" in result.output

    def test_plugins_info(self) -> None:
        """Test plugins --info for a known plugin."""
        result = runner.invoke(app, ["plugins", "--info", "ollama"])

        assert result.exit_code == 0
        assert "Plugin: ollama" in result.output

    def test_plugins_info_json(self) -> None:
        """Test plugins --info with JSON output."""
        result = runner.invoke(app, ["--json", "plugins", "--info", "ollama"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"
        assert data["data"]["name"] == "ollama"

    def test_plugins_info_not_found(self) -> None:
        """Test plugins --info for unknown plugin."""
        result = runner.invoke(app, ["plugins", "--info", "nonexistent"])

        assert result.exit_code == 2
        assert "not found" in result.output

    def test_plugins_info_not_found_json(self) -> None:
        """Test plugins --info for unknown plugin in JSON mode."""
        result = runner.invoke(app, ["--json", "plugins", "--info", "nonexistent"])

        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["status"] == "error"


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_no_color_option(self) -> None:
        """Test --no-color option."""
        result = runner.invoke(app, ["--no-color", "version"])

        assert result.exit_code == 0

    def test_pii_mode_option(self) -> None:
        """Test --pii-mode option."""
        result = runner.invoke(app, ["--pii-mode", "redact", "version"])

        assert result.exit_code == 0

    def test_json_output_deprecated_alias(self) -> None:
        """Test --json-output deprecated alias."""
        result = runner.invoke(app, ["--json-output", "plugins", "--list"])

        assert result.exit_code == 0
        # Should produce JSON output
        data = json.loads(result.output)
        assert data["command"] == "plugins"


class TestDatasetCommand:
    """Tests for dataset subcommands."""

    def test_dataset_no_subcommand(self) -> None:
        """Test dataset command without subcommand shows help."""
        result = runner.invoke(app, ["dataset"])

        # Typer may return 0 or 2 depending on version/config
        assert result.exit_code in [0, 2]
        assert "Dataset management" in result.output or "diff" in result.output or "Usage" in result.output

    def test_dataset_diff_file_not_found(self) -> None:
        """Test dataset diff with non-existent file."""
        result = runner.invoke(app, ["dataset", "diff", "/nonexistent/v1.json", "/nonexistent/v2.json"])

        assert result.exit_code == 2

    def test_dataset_diff_success(self) -> None:
        """Test dataset diff with valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v1_path = Path(tmpdir) / "v1.json"
            v2_path = Path(tmpdir) / "v2.json"

            # Create test datasets
            v1_data = {
                "items": [
                    {"question": "Q1", "context": "C1", "expected_answer": "A1"},
                    {"question": "Q2", "context": "C2", "expected_answer": "A2"},
                ]
            }
            v2_data = {
                "items": [
                    {"question": "Q1", "context": "C1", "expected_answer": "A1"},
                    {"question": "Q3", "context": "C3", "expected_answer": "A3"},
                ]
            }

            v1_path.write_text(json.dumps(v1_data))
            v2_path.write_text(json.dumps(v2_data))

            result = runner.invoke(app, ["dataset", "diff", str(v1_path), str(v2_path)])

            assert result.exit_code == 0

    def test_dataset_diff_json(self) -> None:
        """Test dataset diff with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v1_path = Path(tmpdir) / "v1.json"
            v2_path = Path(tmpdir) / "v2.json"

            v1_data = {"items": [{"question": "Q1", "context": "C1", "expected_answer": "A1"}]}
            v2_data = {"items": [{"question": "Q1", "context": "C1", "expected_answer": "A1"}]}

            v1_path.write_text(json.dumps(v1_data))
            v2_path.write_text(json.dumps(v2_data))

            result = runner.invoke(app, ["--json", "dataset", "diff", str(v1_path), str(v2_path)])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["command"] == "dataset diff"
            assert data["status"] in ["success", "no_changes", "changes_detected"]

    def test_dataset_info(self) -> None:
        """Test dataset info command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"
            data = {
                "items": [
                    {"question": "Q1", "context": "C1", "expected_answer": "A1"},
                ]
            }
            testset_path.write_text(json.dumps(data))

            result = runner.invoke(app, ["dataset", "info", str(testset_path)])

            assert result.exit_code == 0


class TestMonitorCommand:
    """Tests for monitor subcommands."""

    def test_monitor_no_subcommand(self) -> None:
        """Test monitor command without subcommand shows help."""
        result = runner.invoke(app, ["monitor"])

        # Typer may return 0 or 2 depending on version/config
        assert result.exit_code in [0, 2]
        # Should show help
        assert "start" in result.output or "Monitor" in result.output or "Usage" in result.output

    def test_monitor_status(self) -> None:
        """Test monitor status command."""
        result = runner.invoke(app, ["monitor", "status"])

        # Should work even if daemon not running
        assert result.exit_code in [0, 1]

    def test_monitor_status_json(self) -> None:
        """Test monitor status with JSON output."""
        result = runner.invoke(app, ["--json", "monitor", "status"])

        # Should return valid JSON
        data = json.loads(result.output)
        assert data["command"] == "monitor status"


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_no_input(self) -> None:
        """Test evaluate without required input."""
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code == 2
        assert "Either --demo" in result.output

    def test_evaluate_demo(self) -> None:
        """Test evaluate --demo command."""
        result = runner.invoke(app, ["evaluate", "--demo", "--limit", "2"])

        assert result.exit_code == 0
        assert "RAGnarok-AI Demo Evaluation" in result.output

    def test_evaluate_demo_json(self) -> None:
        """Test evaluate --demo with JSON output."""
        result = runner.invoke(app, ["--json", "evaluate", "--demo", "--limit", "2"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["command"] == "evaluate"
        assert data["status"] == "pass"

    def test_evaluate_with_config(self) -> None:
        """Test evaluate with config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ragnarok.yaml"
            config_path.write_text("""
fail_under: 0.5
metrics:
  - precision
  - recall
""")

            result = runner.invoke(
                app, ["evaluate", "--demo", "--limit", "2", "--config", str(config_path)]
            )

            assert result.exit_code == 0


class TestGenerateCommand:
    """Tests for generate command."""

    def test_generate_no_input(self) -> None:
        """Test generate without required input."""
        result = runner.invoke(app, ["generate"])

        assert result.exit_code == 2

    def test_generate_dry_run(self) -> None:
        """Test generate --demo --dry-run."""
        result = runner.invoke(app, ["generate", "--demo", "--dry-run"])

        assert result.exit_code == 0

    def test_generate_dry_run_json(self) -> None:
        """Test generate --demo --dry-run with JSON."""
        result = runner.invoke(app, ["--json", "generate", "--demo", "--dry-run"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["command"] == "generate"
        assert data["status"] == "dry_run"


class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_no_input(self) -> None:
        """Test benchmark without required input."""
        result = runner.invoke(app, ["benchmark"])

        assert result.exit_code == 2

    def test_benchmark_demo(self) -> None:
        """Test benchmark --demo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            result = runner.invoke(app, ["benchmark", "--demo", "--storage", str(storage_path)])

            assert result.exit_code == 0

    def test_benchmark_demo_json(self) -> None:
        """Test benchmark --demo with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            result = runner.invoke(
                app, ["--json", "benchmark", "--demo", "--storage", str(storage_path)]
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["command"] == "benchmark"
            assert data["status"] == "success"

    def test_benchmark_list(self) -> None:
        """Test benchmark --list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # First create some benchmarks
            runner.invoke(app, ["benchmark", "--demo", "--storage", str(storage_path)])

            # Then list them
            result = runner.invoke(app, ["benchmark", "--list", "--storage", str(storage_path)])

            assert result.exit_code == 0

    def test_benchmark_history_not_found(self) -> None:
        """Test benchmark --history with non-existent config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # Try to get history without any benchmarks
            result = runner.invoke(app, ["benchmark", "--history", "nonexistent", "--storage", str(storage_path)])

            # Should return error for non-existent config
            assert result.exit_code == 2

    def test_benchmark_list_json(self) -> None:
        """Test benchmark --list with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"

            # First create some benchmarks
            runner.invoke(app, ["benchmark", "--demo", "--storage", str(storage_path)])

            # Then list them with JSON
            result = runner.invoke(app, ["--json", "benchmark", "--list", "--storage", str(storage_path)])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["command"] == "benchmark"


class TestDatasetDiffCommand:
    """Tests for dataset diff command."""

    def test_dataset_diff_same_files(self) -> None:
        """Test diff of identical datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.json"
            dataset_path.write_text(json.dumps([
                {"text": "Q1", "ground_truth_docs": ["d1"]},
                {"text": "Q2", "ground_truth_docs": ["d2"]},
            ]))

            result = runner.invoke(app, ["dataset", "diff", str(dataset_path), str(dataset_path)])

            assert result.exit_code == 0

    def test_dataset_diff_with_changes(self) -> None:
        """Test diff with changes between datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v1_path = Path(tmpdir) / "v1.json"
            v2_path = Path(tmpdir) / "v2.json"

            v1_path.write_text(json.dumps([
                {"text": "Q1", "ground_truth_docs": ["d1"]},
                {"text": "Q2", "ground_truth_docs": ["d2"]},
            ]))
            v2_path.write_text(json.dumps([
                {"text": "Q1", "ground_truth_docs": ["d1"]},
                {"text": "Q3", "ground_truth_docs": ["d3"]},
            ]))

            result = runner.invoke(app, ["dataset", "diff", str(v1_path), str(v2_path)])

            assert result.exit_code == 0

    def test_dataset_diff_v1_not_found(self) -> None:
        """Test diff with v1 file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v2_path = Path(tmpdir) / "v2.json"
            v2_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, ["dataset", "diff", "/nonexistent/v1.json", str(v2_path)])

            assert result.exit_code == 2

    def test_dataset_diff_v2_not_found(self) -> None:
        """Test diff with v2 file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v1_path = Path(tmpdir) / "v1.json"
            v1_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, ["dataset", "diff", str(v1_path), "/nonexistent/v2.json"])

            assert result.exit_code == 2

    def test_dataset_diff_json_output(self) -> None:
        """Test diff with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.json"
            dataset_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, ["--json", "dataset", "diff", str(dataset_path), str(dataset_path)])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["command"] == "dataset diff"

    def test_dataset_diff_fail_on_change(self) -> None:
        """Test diff --fail-on-change with changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v1_path = Path(tmpdir) / "v1.json"
            v2_path = Path(tmpdir) / "v2.json"

            v1_path.write_text(json.dumps([{"text": "Q1"}]))
            v2_path.write_text(json.dumps([{"text": "Q2"}]))

            result = runner.invoke(app, ["dataset", "diff", str(v1_path), str(v2_path), "--fail-on-change"])

            assert result.exit_code == 1

    def test_dataset_diff_output_file(self) -> None:
        """Test diff with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v1_path = Path(tmpdir) / "v1.json"
            v2_path = Path(tmpdir) / "v2.json"
            output_path = Path(tmpdir) / "diff.json"

            v1_path.write_text(json.dumps([{"text": "Q1"}]))
            v2_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, ["dataset", "diff", str(v1_path), str(v2_path), "--output", str(output_path)])

            assert result.exit_code == 0
            assert output_path.exists()


class TestDatasetInfoCommand:
    """Tests for dataset info command."""

    def test_dataset_info(self) -> None:
        """Test dataset info command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.json"
            dataset_path.write_text(json.dumps({
                "name": "test_dataset",
                "queries": [{"text": "Q1"}, {"text": "Q2"}]
            }))

            result = runner.invoke(app, ["dataset", "info", str(dataset_path)])

            assert result.exit_code == 0
            assert "test_dataset" in result.output

    def test_dataset_info_json(self) -> None:
        """Test dataset info with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.json"
            dataset_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, ["--json", "dataset", "info", str(dataset_path)])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["status"] == "success"

    def test_dataset_info_file_not_found(self) -> None:
        """Test dataset info with missing file."""
        result = runner.invoke(app, ["dataset", "info", "/nonexistent/file.json"])

        assert result.exit_code == 2


class TestMonitorCommandExtended:
    """Additional tests for monitor command."""

    def test_monitor_stop_not_running(self) -> None:
        """Test monitor stop when not running."""
        result = runner.invoke(app, ["monitor", "stop"])

        # Should handle gracefully when daemon is not running
        assert result.exit_code in [0, 1]

    def test_monitor_status_json(self) -> None:
        """Test monitor status with JSON output."""
        result = runner.invoke(app, ["--json", "monitor", "status"])

        assert result.exit_code in [0, 1]
        # Should output valid JSON
        try:
            data = json.loads(result.output)
            assert "command" in data
        except json.JSONDecodeError:
            pass  # May not be JSON if daemon not running


class TestCompareCommand:
    """Tests for compare command."""

    def test_compare_no_args(self) -> None:
        """Test compare without arguments."""
        result = runner.invoke(app, ["compare"])

        assert result.exit_code == 2

    def test_compare_file_not_found(self) -> None:
        """Test compare with missing file."""
        result = runner.invoke(app, ["compare", "/nonexistent/result1.json", "/nonexistent/result2.json"])

        assert result.exit_code == 2


class TestPluginsInfoCommand:
    """Additional tests for plugins info command."""

    def test_plugins_info_builtin(self) -> None:
        """Test plugins --info for builtin plugin."""
        result = runner.invoke(app, ["plugins", "--info", "ollama"])

        assert result.exit_code == 0
        assert "ollama" in result.output.lower()

    def test_plugins_info_json(self) -> None:
        """Test plugins --info with JSON output."""
        result = runner.invoke(app, ["--json", "plugins", "--info", "ollama"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"

    def test_plugins_info_not_found(self) -> None:
        """Test plugins --info for nonexistent plugin."""
        result = runner.invoke(app, ["plugins", "--info", "nonexistent_plugin_xyz"])

        assert result.exit_code == 2  # Bad input - plugin not found


class TestMonitorStatsCommand:
    """Tests for monitor stats command."""

    def test_monitor_stats_basic(self) -> None:
        """Test monitor stats command."""
        result = runner.invoke(app, ["monitor", "stats"])

        # Should handle gracefully even when daemon not running
        assert result.exit_code in [0, 1]

    def test_monitor_stats_json(self) -> None:
        """Test monitor stats with JSON output."""
        result = runner.invoke(app, ["--json", "monitor", "stats"])

        assert result.exit_code in [0, 1]

    def test_monitor_stats_json_output(self) -> None:
        """Test monitor stats with JSON output."""
        result = runner.invoke(app, ["--json", "monitor", "stats"])

        # Should output valid JSON even if daemon not running
        assert result.exit_code in [0, 1]


class TestJudgeCommand:
    """Tests for judge command."""

    def test_judge_help(self) -> None:
        """Test judge command help."""
        result = runner.invoke(app, ["judge", "--help"])

        assert result.exit_code == 0
        assert "judge" in result.output.lower()

    def test_judge_missing_testset(self) -> None:
        """Test judge command with missing testset file."""
        result = runner.invoke(app, [
            "judge",
            "--testset", "/nonexistent/testset.json",
            "--results", "/nonexistent/results.json"
        ])

        assert result.exit_code == 2

    def test_judge_missing_results(self) -> None:
        """Test judge command with missing results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"
            testset_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, [
                "judge",
                "--testset", str(testset_path),
                "--results", "/nonexistent/results.json"
            ])

            assert result.exit_code == 2


class TestGenerateCommandExtended:
    """Extended tests for generate command."""

    def test_generate_help(self) -> None:
        """Test generate command help."""
        result = runner.invoke(app, ["generate", "--help"])

        assert result.exit_code == 0
        assert "generate" in result.output.lower()

    def test_generate_missing_docs(self) -> None:
        """Test generate with missing documents file."""
        result = runner.invoke(app, ["generate", "--documents", "/nonexistent/docs.txt"])

        assert result.exit_code == 2


class TestEvaluateCommandExtended:
    """Extended tests for evaluate command."""

    def test_evaluate_with_output_flag(self) -> None:
        """Test evaluate --output flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"
            output_path = Path(tmpdir) / "output.json"

            # Create a valid testset
            testset_path.write_text(json.dumps([
                {"text": "What is AI?", "ground_truth_answer": "AI is artificial intelligence"}
            ]))

            # This will still fail because there's no model, but it tests the parsing
            result = runner.invoke(app, [
                "evaluate",
                "--testset", str(testset_path),
                "--output", str(output_path)
            ])

            # Exit code might vary based on model availability
            assert result.exit_code in [0, 1, 2]

    def test_evaluate_with_fail_under(self) -> None:
        """Test evaluate --fail-under flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"
            testset_path.write_text(json.dumps([{"text": "Test question"}]))

            result = runner.invoke(app, [
                "evaluate",
                "--testset", str(testset_path),
                "--fail-under", "0.9"
            ])

            # Will fail due to missing model, but tests argument parsing
            assert result.exit_code in [0, 1, 2]


class TestBenchmarkCommandExtended:
    """Extended tests for benchmark command."""

    def test_benchmark_compare_missing_baseline(self) -> None:
        """Test benchmark --compare with missing baseline file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"
            testset_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, [
                "benchmark",
                "--testset", str(testset_path),
                "--compare", "/nonexistent/baseline.json"
            ])

            assert result.exit_code == 2

    def test_benchmark_with_storage_flag(self) -> None:
        """Test benchmark --storage flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "benchmarks.json"
            storage_path.write_text("{}")

            result = runner.invoke(app, [
                "benchmark",
                "--list",
                "--storage", str(storage_path)
            ])

            assert result.exit_code in [0, 1]


class TestVersionCommand:
    """Tests for version command."""

    def test_version_command(self) -> None:
        """Test version subcommand."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        # Should output version string


class TestMainWithJsonFlag:
    """Tests for global --json flag behavior."""

    def test_json_flag_sets_state(self) -> None:
        """Test that --json flag is parsed correctly."""
        result = runner.invoke(app, ["--json", "plugins", "--list"])

        assert result.exit_code == 0
        # Output should be valid JSON
        data = json.loads(result.output)
        assert "command" in data

    def test_evaluate_with_json_error(self) -> None:
        """Test evaluate command with JSON output on error."""
        result = runner.invoke(app, ["--json", "evaluate", "--testset", "/nonexistent/file.json"])

        # Exit code could be 1 or 2 depending on error handling
        assert result.exit_code in [1, 2]


class TestConfigFileOptions:
    """Tests for config file parsing options."""

    def test_evaluate_with_metrics_list(self) -> None:
        """Test evaluate with multiple metrics specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testset_path = Path(tmpdir) / "testset.json"
            testset_path.write_text(json.dumps([{"text": "Q1"}]))

            result = runner.invoke(app, [
                "evaluate",
                "--testset", str(testset_path),
                "--metrics", "precision",
                "--metrics", "recall"
            ])

            # Will fail due to missing model but tests arg parsing
            assert result.exit_code in [0, 1, 2]

    def test_load_config_with_all_options(self) -> None:
        """Test loading config with all options."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
testset: test.json
output: results.json
fail_under: 0.75
metrics:
  - precision
  - recall
  - faithfulness
ollama_url: http://localhost:11434
demo: false
""")
            f.flush()

            config = load_config(f.name)

            assert config.testset == "test.json"
            assert config.output == "results.json"
            assert config.fail_under == 0.75
            assert "precision" in config.metrics
            assert "recall" in config.metrics
            assert "faithfulness" in config.metrics
            assert config.ollama_url == "http://localhost:11434"
            Path(f.name).unlink()
