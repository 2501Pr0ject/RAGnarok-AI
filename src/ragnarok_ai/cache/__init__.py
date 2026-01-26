"""Cache module for ragnarok-ai.

This module provides caching for evaluation results to avoid re-evaluating
unchanged queries.
"""

from __future__ import annotations

from ragnarok_ai.cache.base import CacheProtocol
from ragnarok_ai.cache.disk import DiskCache
from ragnarok_ai.cache.memory import MemoryCache

__all__ = [
    "CacheProtocol",
    "DiskCache",
    "MemoryCache",
]
