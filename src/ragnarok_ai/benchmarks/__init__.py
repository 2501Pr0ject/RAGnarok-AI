"""Benchmark tracking module for ragnarok-ai.

This module provides tools for recording, storing, and querying
benchmark results for historical tracking and regression detection.

Example:
    >>> from ragnarok_ai.benchmarks import BenchmarkHistory
    >>> from ragnarok_ai.regression import RegressionThresholds
    >>>
    >>> history = BenchmarkHistory()
    >>> record = await history.record(result, "my-config", testset)
    >>> await history.set_baseline(record.id)
    >>>
    >>> # Later: compare new evaluation to baseline
    >>> regression = await history.compare_to_baseline(
    ...     "my-config", new_result, testset,
    ...     thresholds=RegressionThresholds(precision_drop=0.03),
    ... )
"""

from __future__ import annotations

from ragnarok_ai.benchmarks.history import BenchmarkHistory
from ragnarok_ai.benchmarks.models import BenchmarkRecord
from ragnarok_ai.benchmarks.storage import JSONFileStore, StorageProtocol

__all__ = [
    "BenchmarkHistory",
    "BenchmarkRecord",
    "JSONFileStore",
    "StorageProtocol",
]
