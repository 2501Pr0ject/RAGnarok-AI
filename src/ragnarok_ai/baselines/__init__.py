"""Baselines library for ragnarok-ai.

This module provides ready-to-use baseline configurations with expected
results for comparing RAG pipeline performance.
"""

from __future__ import annotations

from ragnarok_ai.baselines.comparison import (
    BaselineComparison,
    MetricComparison,
    compare,
)
from ragnarok_ai.baselines.configs import (
    BASELINE_CONFIGS,
    BaselineConfig,
    get_baseline_config,
    list_baselines,
)
from ragnarok_ai.baselines.results import (
    BaselineResult,
    get_baseline_result,
)

__all__ = [
    "BASELINE_CONFIGS",
    "BaselineComparison",
    "BaselineConfig",
    "BaselineResult",
    "MetricComparison",
    "compare",
    "get_baseline_config",
    "get_baseline_result",
    "list_baselines",
]
