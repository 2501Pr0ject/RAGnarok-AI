"""In-memory cache implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.cache.base import CacheStats

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import QueryResult


class MemoryCache:
    """In-memory cache for evaluation results.

    Simple dictionary-based cache. Data is lost when the process exits.

    Example:
        >>> cache = MemoryCache()
        >>> result = await evaluate(rag, testset, cache=cache)
        >>> print(cache.stats())
        CacheStats(hits=5, misses=10, hit_rate=33.33%)
    """

    def __init__(self, max_size: int | None = None) -> None:
        """Initialize the memory cache.

        Args:
            max_size: Maximum number of entries. None for unlimited.
        """
        self._cache: dict[str, QueryResult] = {}
        self._max_size = max_size
        self._stats = CacheStats()

    async def get(self, key: str) -> QueryResult | None:
        """Get a cached result by key."""
        result = self._cache.get(key)
        if result is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1
        return result

    async def set(self, key: str, result: QueryResult) -> None:
        """Store a result in the cache."""
        # Don't cache errors
        if result.error is not None:
            return

        # Evict oldest if at max size
        if self._max_size and len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result
        self._stats.size = len(self._cache)

    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    async def delete(self, key: str) -> None:
        """Delete a cached result."""
        self._cache.pop(key, None)
        self._stats.size = len(self._cache)

    async def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._stats.size = 0

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)
