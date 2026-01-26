"""Storage backends for benchmark tracking.

This module provides storage protocols and implementations for
persisting benchmark records.

Example:
    >>> from ragnarok_ai.benchmarks.storage import JSONFileStore
    >>> store = JSONFileStore(".ragnarok/benchmarks.json")
    >>> await store.save(record)
"""

from __future__ import annotations

from ragnarok_ai.benchmarks.storage.base import StorageProtocol
from ragnarok_ai.benchmarks.storage.json_store import JSONFileStore

__all__ = [
    "JSONFileStore",
    "StorageProtocol",
]
