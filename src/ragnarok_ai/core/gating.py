"""CI gating configuration for ragnarok-ai.

This module provides intelligent CI/CD gating that distinguishes
between stable and unstable metrics.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MetricCategory(str, Enum):
    """Category of metric for gating purposes."""

    STABLE = "stable"
    UNSTABLE = "unstable"


class GatingResult(str, Enum):
    """Result of gating evaluation."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MetricResult(BaseModel):
    """Result for a single metric evaluation.

    Attributes:
        name: Name of the metric.
        value: Actual value of the metric.
        threshold: Threshold value for the metric.
        category: Whether this is a stable or unstable metric.
        result: Pass, warn, or fail result.
    """

    model_config = {"frozen": True}

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Actual metric value")
    threshold: float = Field(..., description="Threshold value")
    category: MetricCategory = Field(..., description="Metric category")
    result: GatingResult = Field(..., description="Evaluation result")


class GatingConfig(BaseModel):
    """Configuration for CI gating thresholds.

    Attributes:
        stable: Thresholds for stable metrics (hard fail if below).
        unstable: Thresholds for unstable metrics (warning only if below).
    """

    model_config = {"frozen": True}

    stable: dict[str, float] = Field(
        default_factory=lambda: {
            "precision": 0.8,
            "recall": 0.8,
            "mrr": 0.8,
            "ndcg": 0.8,
        },
        description="Stable metric thresholds (fail if below)",
    )
    unstable: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": 0.7,
            "relevance": 0.7,
            "hallucination": 0.3,  # Lower is better for hallucination
        },
        description="Unstable metric thresholds (warn if below)",
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> GatingConfig:
        """Load gating configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            GatingConfig loaded from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        content = path.read_text()
        data = yaml.safe_load(content)

        if data is None:
            return cls()

        gating_data = data.get("gating", data)
        return cls(
            stable=gating_data.get("stable", {}),
            unstable=gating_data.get("unstable", {}),
        )

    def to_yaml(self, path: Path | str) -> None:
        """Save gating configuration to a YAML file.

        Args:
            path: Path to the output YAML file.
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "gating": {
                "stable": dict(self.stable),
                "unstable": dict(self.unstable),
            }
        }

        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


class GatingEvaluationResult(BaseModel):
    """Result of evaluating metrics against gating thresholds.

    Attributes:
        overall_result: The overall gating result (pass/warn/fail).
        exit_code: Exit code for CI (0=pass, 1=fail, 2=warn).
        metrics: Individual metric results.
        summary: Human-readable summary of the evaluation.
    """

    model_config = {"frozen": True}

    overall_result: GatingResult = Field(..., description="Overall result")
    exit_code: int = Field(..., ge=0, le=2, description="Exit code for CI")
    metrics: list[MetricResult] = Field(default_factory=list, description="Metric results")
    summary: str = Field(..., description="Human-readable summary")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "overall_result": self.overall_result.value,
            "exit_code": self.exit_code,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "category": m.category.value,
                    "result": m.result.value,
                }
                for m in self.metrics
            ],
            "summary": self.summary,
        }


class GatingEvaluator:
    """Evaluates metrics against gating thresholds.

    This evaluator distinguishes between stable and unstable metrics:
    - Stable metrics (retrieval, latency): Hard fail if below threshold
    - Unstable metrics (LLM-as-judge): Warning only if below threshold

    Exit codes:
    - 0: All metrics pass
    - 1: At least one stable metric failed
    - 2: Only unstable metrics below threshold (warning)

    Attributes:
        config: The gating configuration.

    Example:
        >>> config = GatingConfig()
        >>> evaluator = GatingEvaluator(config)
        >>> result = evaluator.evaluate({"precision": 0.85, "faithfulness": 0.65})
        >>> print(result.exit_code)  # 2 (warning - faithfulness below threshold)
    """

    def __init__(self, config: GatingConfig | None = None) -> None:
        """Initialize GatingEvaluator.

        Args:
            config: Gating configuration. Uses defaults if not provided.
        """
        self.config = config or GatingConfig()

    def evaluate(self, metrics: dict[str, float]) -> GatingEvaluationResult:
        """Evaluate metrics against gating thresholds.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            GatingEvaluationResult with overall result and per-metric details.
        """
        results: list[MetricResult] = []
        has_stable_fail = False
        has_unstable_warn = False

        # Check stable metrics
        for name, threshold in self.config.stable.items():
            if name in metrics:
                value = metrics[name]
                passed = value >= threshold
                result = GatingResult.PASS if passed else GatingResult.FAIL

                if not passed:
                    has_stable_fail = True

                results.append(
                    MetricResult(
                        name=name,
                        value=value,
                        threshold=threshold,
                        category=MetricCategory.STABLE,
                        result=result,
                    )
                )

        # Check unstable metrics
        for name, threshold in self.config.unstable.items():
            if name in metrics:
                value = metrics[name]

                # For hallucination, lower is better
                passed = value <= threshold if name == "hallucination" else value >= threshold

                result = GatingResult.PASS if passed else GatingResult.WARN

                if not passed:
                    has_unstable_warn = True

                results.append(
                    MetricResult(
                        name=name,
                        value=value,
                        threshold=threshold,
                        category=MetricCategory.UNSTABLE,
                        result=result,
                    )
                )

        # Determine overall result and exit code
        if has_stable_fail:
            overall_result = GatingResult.FAIL
            exit_code = 1
            summary = self._generate_fail_summary(results)
        elif has_unstable_warn:
            overall_result = GatingResult.WARN
            exit_code = 2
            summary = self._generate_warn_summary(results)
        else:
            overall_result = GatingResult.PASS
            exit_code = 0
            summary = "All metrics passed gating thresholds."

        return GatingEvaluationResult(
            overall_result=overall_result,
            exit_code=exit_code,
            metrics=results,
            summary=summary,
        )

    def _generate_fail_summary(self, results: list[MetricResult]) -> str:
        """Generate summary for failed gating.

        Args:
            results: List of metric results.

        Returns:
            Human-readable summary.
        """
        failed = [r for r in results if r.result == GatingResult.FAIL]
        failed_names = ", ".join(r.name for r in failed)
        return f"Gating FAILED: Stable metrics below threshold: {failed_names}"

    def _generate_warn_summary(self, results: list[MetricResult]) -> str:
        """Generate summary for warning gating.

        Args:
            results: List of metric results.

        Returns:
            Human-readable summary.
        """
        warned = [r for r in results if r.result == GatingResult.WARN]
        warned_names = ", ".join(r.name for r in warned)
        return f"Gating WARNING: Unstable metrics below threshold: {warned_names}"
