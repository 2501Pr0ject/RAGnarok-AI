"""Disk-based cache implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.cache.base import CacheStats

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import QueryResult


class DiskCache:
    """Disk-based cache for evaluation results.

    Stores results as JSON files in a directory. Persists across process restarts.

    Example:
        >>> cache = DiskCache("./eval_cache/")
        >>> result = await evaluate(rag, testset, cache=cache)
        >>> # Next run will use cached results
        >>> result = await evaluate(rag, testset, cache=cache)
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the disk cache.

        Args:
            path: Directory path for cache files.
        """
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats()
        self._stats.size = len(list(self._path.glob("*.json")))

    def _key_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self._path / f"{key}.json"

    async def get(self, key: str) -> QueryResult | None:
        """Get a cached result by key."""
        from ragnarok_ai.core.evaluate import QueryResult
        from ragnarok_ai.core.types import Query
        from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

        path = self._key_path(key)
        if not path.exists():
            self._stats.misses += 1
            return None

        try:
            data = json.loads(path.read_text())
            self._stats.hits += 1

            # Reconstruct QueryResult
            query_data = data["query"]
            query = Query(
                text=query_data["text"],
                ground_truth_docs=query_data["ground_truth_docs"],
            )

            metric_data = data["metric"]
            metric = RetrievalMetrics(
                precision=metric_data["precision"],
                recall=metric_data["recall"],
                mrr=metric_data["mrr"],
                ndcg=metric_data["ndcg"],
                k=metric_data["k"],
            )

            return QueryResult(
                query=query,
                metric=metric,
                answer=data["answer"],
                latency_ms=data["latency_ms"],
                error=None,
            )
        except (json.JSONDecodeError, KeyError):
            self._stats.misses += 1
            return None

    async def set(self, key: str, result: QueryResult) -> None:
        """Store a result in the cache."""
        if result.error is not None:
            # Don't cache errors
            return

        path = self._key_path(key)
        data: dict[str, Any] = {
            "query": {
                "text": result.query.text,
                "ground_truth_docs": result.query.ground_truth_docs,
            },
            "metric": {
                "precision": result.metric.precision,
                "recall": result.metric.recall,
                "mrr": result.metric.mrr,
                "ndcg": result.metric.ndcg,
                "k": result.metric.k,
            },
            "answer": result.answer,
            "latency_ms": result.latency_ms,
        }

        path.write_text(json.dumps(data, indent=2))
        self._stats.size = len(list(self._path.glob("*.json")))

    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self._key_path(key).exists()

    async def delete(self, key: str) -> None:
        """Delete a cached result."""
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            self._stats.size = len(list(self._path.glob("*.json")))

    async def clear(self) -> None:
        """Clear all cached results."""
        for path in self._path.glob("*.json"):
            path.unlink()
        self._stats.size = 0

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(list(self._path.glob("*.json")))
