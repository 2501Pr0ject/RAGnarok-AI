"""Diff reports module for ragnarok-ai.

This module provides tools for comparing two evaluation runs
and generating detailed diff reports.

Example:
    >>> from ragnarok_ai.diff import compute_diff
    >>>
    >>> diff = compute_diff(
    ...     baseline=baseline_result,
    ...     current=current_result,
    ...     baseline_name="v1.0",
    ...     current_name="v1.1",
    ... )
    >>> print(diff.summary())
    '5 improved, 3 degraded, 42 unchanged'
    >>> print(diff.to_markdown())
"""

from __future__ import annotations

from ragnarok_ai.diff.generator import compute_diff
from ragnarok_ai.diff.models import DiffReport, QueryDiff

__all__ = [
    "DiffReport",
    "QueryDiff",
    "compute_diff",
]
