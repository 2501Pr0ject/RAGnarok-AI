"""Regression detection module for ragnarok-ai.

This module provides tools for detecting quality regressions in RAG
evaluation results compared to a baseline.

Example:
    >>> from ragnarok_ai.regression import RegressionDetector, RegressionThresholds
    >>>
    >>> detector = RegressionDetector(
    ...     baseline=baseline_result,
    ...     thresholds=RegressionThresholds(precision_drop=0.03),
    ... )
    >>> result = detector.detect(current_result)
    >>> if result.has_critical:
    ...     print("Critical regressions detected!")
"""

from __future__ import annotations

from ragnarok_ai.regression.detector import LOWER_IS_BETTER, RegressionDetector
from ragnarok_ai.regression.models import (
    RegressionAlert,
    RegressionResult,
    RegressionThresholds,
)

__all__ = [
    "LOWER_IS_BETTER",
    "RegressionAlert",
    "RegressionDetector",
    "RegressionResult",
    "RegressionThresholds",
]
