"""Dataset management module for ragnarok-ai.

This module provides tools for loading, saving, and comparing
TestSet versions for reproducibility and debugging.
"""

from __future__ import annotations

from ragnarok_ai.dataset.diff import build_index, diff_testsets
from ragnarok_ai.dataset.io import load_testset, save_testset
from ragnarok_ai.dataset.models import DatasetDiffReport, FieldChange, ModifiedItem

__all__ = [
    "DatasetDiffReport",
    "FieldChange",
    "ModifiedItem",
    "build_index",
    "diff_testsets",
    "load_testset",
    "save_testset",
]
