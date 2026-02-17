"""Tests for CI gating module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ragnarok_ai.core.gating import (
    GatingConfig,
    GatingEvaluationResult,
    GatingEvaluator,
    GatingResult,
    MetricCategory,
    MetricResult,
)


class TestGatingConfigDefaults:
    """Tests for GatingConfig default values."""

    def test_default_stable_thresholds(self) -> None:
        """Default stable thresholds are set."""
        config = GatingConfig()

        assert config.stable["precision"] == 0.8
        assert config.stable["recall"] == 0.8
        assert config.stable["mrr"] == 0.8
        assert config.stable["ndcg"] == 0.8

    def test_default_unstable_thresholds(self) -> None:
        """Default unstable thresholds are set."""
        config = GatingConfig()

        assert config.unstable["faithfulness"] == 0.7
        assert config.unstable["relevance"] == 0.7
        assert config.unstable["hallucination"] == 0.3

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        config = GatingConfig(
            stable={"precision": 0.9},
            unstable={"faithfulness": 0.8},
        )

        assert config.stable["precision"] == 0.9
        assert config.unstable["faithfulness"] == 0.8


class TestGatingConfigYaml:
    """Tests for GatingConfig YAML serialization."""

    def test_from_yaml_full_config(self, tmp_path: Path) -> None:
        """Loads full config from YAML."""
        yaml_content = """
gating:
  stable:
    precision: 0.85
    recall: 0.75
  unstable:
    faithfulness: 0.65
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = GatingConfig.from_yaml(config_file)

        assert config.stable["precision"] == 0.85
        assert config.stable["recall"] == 0.75
        assert config.unstable["faithfulness"] == 0.65

    def test_from_yaml_partial_config(self, tmp_path: Path) -> None:
        """Loads partial config from YAML."""
        yaml_content = """
gating:
  stable:
    precision: 0.9
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = GatingConfig.from_yaml(config_file)

        assert config.stable["precision"] == 0.9
        assert config.unstable == {}

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        """Handles empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = GatingConfig.from_yaml(config_file)

        # Should use defaults
        assert config.stable["precision"] == 0.8

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing file."""
        import pytest

        with pytest.raises(FileNotFoundError):
            GatingConfig.from_yaml(tmp_path / "missing.yaml")

    def test_to_yaml(self, tmp_path: Path) -> None:
        """Saves config to YAML."""
        config = GatingConfig(
            stable={"precision": 0.9},
            unstable={"faithfulness": 0.8},
        )
        output_path = tmp_path / "output.yaml"

        config.to_yaml(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "precision" in content
        assert "0.9" in content

    def test_to_yaml_creates_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if needed."""
        config = GatingConfig()
        output_path = tmp_path / "nested" / "dir" / "config.yaml"

        config.to_yaml(output_path)

        assert output_path.exists()


class TestGatingEvaluatorAllPass:
    """Tests for when all metrics pass."""

    def test_all_stable_pass(self) -> None:
        """All stable metrics pass returns exit code 0."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.85,
            "recall": 0.82,
            "mrr": 0.88,
            "ndcg": 0.84,
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.PASS
        assert result.exit_code == 0
        assert "All metrics passed" in result.summary

    def test_all_unstable_pass(self) -> None:
        """All unstable metrics pass returns exit code 0."""
        evaluator = GatingEvaluator()
        metrics = {
            "faithfulness": 0.75,
            "relevance": 0.78,
            "hallucination": 0.15,  # Lower is better
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.PASS
        assert result.exit_code == 0

    def test_all_metrics_pass(self) -> None:
        """All metrics pass returns exit code 0."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.85,
            "recall": 0.82,
            "faithfulness": 0.75,
            "hallucination": 0.1,
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.PASS
        assert result.exit_code == 0


class TestGatingEvaluatorStableFail:
    """Tests for when stable metrics fail."""

    def test_stable_metric_fails(self) -> None:
        """Stable metric failure returns exit code 1."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.5,  # Below 0.8 threshold
            "recall": 0.85,
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.FAIL
        assert result.exit_code == 1
        assert "FAILED" in result.summary
        assert "precision" in result.summary

    def test_multiple_stable_fail(self) -> None:
        """Multiple stable failures returns exit code 1."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.5,
            "recall": 0.4,
            "mrr": 0.3,
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.FAIL
        assert result.exit_code == 1

    def test_stable_fail_overrides_unstable_warn(self) -> None:
        """Stable fail takes precedence over unstable warning."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.5,  # Stable fail
            "faithfulness": 0.5,  # Unstable warn
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.FAIL
        assert result.exit_code == 1


class TestGatingEvaluatorUnstableWarn:
    """Tests for when only unstable metrics fail."""

    def test_unstable_metric_warns(self) -> None:
        """Unstable metric failure returns exit code 2."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.85,  # Pass
            "faithfulness": 0.5,  # Below 0.7 threshold
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.WARN
        assert result.exit_code == 2
        assert "WARNING" in result.summary
        assert "faithfulness" in result.summary

    def test_hallucination_higher_is_bad(self) -> None:
        """Hallucination above threshold triggers warning."""
        evaluator = GatingEvaluator()
        metrics = {
            "precision": 0.85,
            "hallucination": 0.5,  # Above 0.3 threshold (bad)
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.WARN
        assert result.exit_code == 2

    def test_multiple_unstable_warn(self) -> None:
        """Multiple unstable warnings returns exit code 2."""
        evaluator = GatingEvaluator()
        metrics = {
            "faithfulness": 0.5,
            "relevance": 0.4,
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.WARN
        assert result.exit_code == 2


class TestGatingEvaluatorCustomConfig:
    """Tests with custom configuration."""

    def test_custom_thresholds(self) -> None:
        """Custom thresholds are respected."""
        config = GatingConfig(
            stable={"precision": 0.9},
            unstable={"faithfulness": 0.8},
        )
        evaluator = GatingEvaluator(config)
        metrics = {
            "precision": 0.85,  # Below 0.9
            "faithfulness": 0.75,  # Below 0.8
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.FAIL
        assert result.exit_code == 1

    def test_unknown_metrics_ignored(self) -> None:
        """Metrics not in config are ignored."""
        config = GatingConfig(
            stable={"precision": 0.8},
            unstable={},
        )
        evaluator = GatingEvaluator(config)
        metrics = {
            "precision": 0.85,
            "unknown_metric": 0.1,
        }

        result = evaluator.evaluate(metrics)

        assert result.overall_result == GatingResult.PASS
        assert len(result.metrics) == 1


class TestGatingEvaluatorMetricResults:
    """Tests for individual metric results."""

    def test_metric_results_populated(self) -> None:
        """Metric results contain correct details."""
        evaluator = GatingEvaluator()
        metrics = {"precision": 0.85, "faithfulness": 0.5}

        result = evaluator.evaluate(metrics)

        assert len(result.metrics) == 2

        precision_result = next(m for m in result.metrics if m.name == "precision")
        assert precision_result.value == 0.85
        assert precision_result.threshold == 0.8
        assert precision_result.category == MetricCategory.STABLE
        assert precision_result.result == GatingResult.PASS

        faithfulness_result = next(m for m in result.metrics if m.name == "faithfulness")
        assert faithfulness_result.value == 0.5
        assert faithfulness_result.threshold == 0.7
        assert faithfulness_result.category == MetricCategory.UNSTABLE
        assert faithfulness_result.result == GatingResult.WARN


class TestGatingEvaluationResultToDict:
    """Tests for result serialization."""

    def test_to_dict(self) -> None:
        """Result can be converted to dictionary."""
        result = GatingEvaluationResult(
            overall_result=GatingResult.PASS,
            exit_code=0,
            metrics=[
                MetricResult(
                    name="precision",
                    value=0.85,
                    threshold=0.8,
                    category=MetricCategory.STABLE,
                    result=GatingResult.PASS,
                )
            ],
            summary="All metrics passed.",
        )

        data = result.to_dict()

        assert data["overall_result"] == "pass"
        assert data["exit_code"] == 0
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["name"] == "precision"
        assert data["metrics"][0]["category"] == "stable"


class TestMetricResultModel:
    """Tests for MetricResult model."""

    def test_create_metric_result(self) -> None:
        """MetricResult can be created with valid data."""
        result = MetricResult(
            name="precision",
            value=0.85,
            threshold=0.8,
            category=MetricCategory.STABLE,
            result=GatingResult.PASS,
        )

        assert result.name == "precision"
        assert result.value == 0.85
        assert result.threshold == 0.8
        assert result.category == MetricCategory.STABLE
        assert result.result == GatingResult.PASS

    def test_metric_result_is_frozen(self) -> None:
        """MetricResult is immutable."""
        import pytest
        from pydantic import ValidationError

        result = MetricResult(
            name="precision",
            value=0.85,
            threshold=0.8,
            category=MetricCategory.STABLE,
            result=GatingResult.PASS,
        )

        with pytest.raises(ValidationError):
            result.value = 0.9


class TestGatingResultEnum:
    """Tests for GatingResult enum."""

    def test_gating_result_values(self) -> None:
        """GatingResult has correct values."""
        assert GatingResult.PASS.value == "pass"
        assert GatingResult.WARN.value == "warn"
        assert GatingResult.FAIL.value == "fail"


class TestMetricCategoryEnum:
    """Tests for MetricCategory enum."""

    def test_metric_category_values(self) -> None:
        """MetricCategory has correct values."""
        assert MetricCategory.STABLE.value == "stable"
        assert MetricCategory.UNSTABLE.value == "unstable"
