"""Base cache protocol for ragnarok-ai."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import QueryResult
    from ragnarok_ai.core.types import Query


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for evaluation result caches.

    Caches store query results keyed by a hash of the query and configuration.
    """

    async def get(self, key: str) -> QueryResult | None:
        """Get a cached result by key.

        Args:
            key: Cache key.

        Returns:
            Cached QueryResult or None if not found.
        """
        ...

    async def set(self, key: str, result: QueryResult) -> None:
        """Store a result in the cache.

        Args:
            key: Cache key.
            result: QueryResult to cache.
        """
        ...

    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists, False otherwise.
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete a cached result.

        Args:
            key: Cache key.
        """
        ...

    async def clear(self) -> None:
        """Clear all cached results."""
        ...

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit/miss counts.
        """
        ...


class CacheStats:
    """Cache statistics."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.size = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def __repr__(self) -> str:
        return f"CacheStats(hits={self.hits}, misses={self.misses}, size={self.size}, hit_rate={self.hit_rate:.2%})"


def compute_cache_key(
    query: Query,
    *,
    pipeline_id: str | None = None,
    k: int = 10,
    extra: dict[str, Any] | None = None,
) -> str:
    """Compute a cache key for a query.

    The key is based on:
    - Query text
    - Ground truth document IDs
    - Pipeline identifier (if provided)
    - k parameter
    - Any extra configuration

    Args:
        query: The query to cache.
        pipeline_id: Optional pipeline identifier for cache isolation.
        k: The k parameter for @K metrics.
        extra: Additional configuration to include in the key.

    Returns:
        A hex digest cache key.
    """
    key_data = {
        "query_text": query.text,
        "ground_truth_docs": sorted(query.ground_truth_docs),
        "pipeline_id": pipeline_id or "default",
        "k": k,
    }

    if extra:
        key_data["extra"] = extra

    # Stable JSON serialization
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:32]
