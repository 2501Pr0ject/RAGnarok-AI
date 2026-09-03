"""Live A/B testing of RAG configurations.

Deterministically split production traffic between variants, tag monitor
traces with the assignment, and compare the variants' success rate and
latency with statistical significance tests.
"""

from ragnarok_ai.ab.analyzer import ABAnalyzer
from ragnarok_ai.ab.models import (
    ABTestConfig,
    ABTestReport,
    Experiment,
    MetricVerdict,
    VariantStats,
)

__all__ = [
    "ABAnalyzer",
    "ABTestConfig",
    "ABTestReport",
    "Experiment",
    "MetricVerdict",
    "VariantStats",
]
