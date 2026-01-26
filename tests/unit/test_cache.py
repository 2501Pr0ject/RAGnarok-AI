"""Unit tests for the cache module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ragnarok_ai.cache import DiskCache, MemoryCache
from ragnarok_ai.cache.base import CacheProtocol, CacheStats, compute_cache_key
from ragnarok_ai.core.evaluate import QueryResult
from ragnarok_ai.core.types import Query
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

if TYPE_CHECKING:
    from ragnarok_ai.core.types import RAGResponse


# ============================================================================
# Test Fixtures
# ============================================================================


def make_query_result(query_text: str = "test query") -> QueryResult:
    """Create a test QueryResult."""
    return QueryResult(
        query=Query(text=query_text, ground_truth_docs=["doc_1"]),
        metric=RetrievalMetrics(precision=0.8, recall=0.6, mrr=0.9, ndcg=0.75, k=10),
        answer="Test answer",
        latency_ms=100.0,
    )


# ============================================================================
# CacheStats Tests
# ============================================================================


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_values(self) -> None:
        """Test initial stats values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0

    def test_hit_rate_zero_total(self) -> None:
        """Test hit rate with zero total."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats()
        stats.hits = 3
        stats.misses = 7
        assert stats.hit_rate == 0.3

    def test_repr(self) -> None:
        """Test string representation."""
        stats = CacheStats()
        stats.hits = 5
        stats.misses = 5
        assert "50.00%" in repr(stats)


# ============================================================================
# compute_cache_key Tests
# ============================================================================


class TestComputeCacheKey:
    """Tests for compute_cache_key."""

    def test_same_query_same_key(self) -> None:
        """Test that same query produces same key."""
        query = Query(text="What is X?", ground_truth_docs=["doc_1", "doc_2"])
        key1 = compute_cache_key(query)
        key2 = compute_cache_key(query)
        assert key1 == key2

    def test_different_query_different_key(self) -> None:
        """Test that different queries produce different keys."""
        query1 = Query(text="What is X?", ground_truth_docs=["doc_1"])
        query2 = Query(text="What is Y?", ground_truth_docs=["doc_1"])
        key1 = compute_cache_key(query1)
        key2 = compute_cache_key(query2)
        assert key1 != key2

    def test_different_k_different_key(self) -> None:
        """Test that different k values produce different keys."""
        query = Query(text="What is X?", ground_truth_docs=["doc_1"])
        key1 = compute_cache_key(query, k=5)
        key2 = compute_cache_key(query, k=10)
        assert key1 != key2

    def test_different_pipeline_different_key(self) -> None:
        """Test that different pipeline IDs produce different keys."""
        query = Query(text="What is X?", ground_truth_docs=["doc_1"])
        key1 = compute_cache_key(query, pipeline_id="pipeline_a")
        key2 = compute_cache_key(query, pipeline_id="pipeline_b")
        assert key1 != key2

    def test_ground_truth_order_independent(self) -> None:
        """Test that ground truth order doesn't affect key."""
        query1 = Query(text="What is X?", ground_truth_docs=["doc_1", "doc_2"])
        query2 = Query(text="What is X?", ground_truth_docs=["doc_2", "doc_1"])
        key1 = compute_cache_key(query1)
        key2 = compute_cache_key(query2)
        assert key1 == key2

    def test_key_length(self) -> None:
        """Test that key has expected length."""
        query = Query(text="test", ground_truth_docs=[])
        key = compute_cache_key(query)
        assert len(key) == 32


# ============================================================================
# MemoryCache Tests
# ============================================================================


class TestMemoryCache:
    """Tests for MemoryCache."""

    @pytest.mark.asyncio
    async def test_get_missing_key(self) -> None:
        """Test getting a missing key."""
        cache = MemoryCache()
        result = await cache.get("nonexistent")
        assert result is None
        assert cache.stats().misses == 1

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """Test setting and getting a value."""
        cache = MemoryCache()
        query_result = make_query_result()
        await cache.set("key1", query_result)
        result = await cache.get("key1")
        assert result is not None
        assert result.answer == "Test answer"
        assert cache.stats().hits == 1

    @pytest.mark.asyncio
    async def test_has(self) -> None:
        """Test checking if key exists."""
        cache = MemoryCache()
        assert not await cache.has("key1")
        await cache.set("key1", make_query_result())
        assert await cache.has("key1")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test deleting a key."""
        cache = MemoryCache()
        await cache.set("key1", make_query_result())
        assert await cache.has("key1")
        await cache.delete("key1")
        assert not await cache.has("key1")

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing all keys."""
        cache = MemoryCache()
        await cache.set("key1", make_query_result())
        await cache.set("key2", make_query_result())
        assert len(cache) == 2
        await cache.clear()
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_max_size(self) -> None:
        """Test max size eviction."""
        cache = MemoryCache(max_size=2)
        await cache.set("key1", make_query_result("q1"))
        await cache.set("key2", make_query_result("q2"))
        await cache.set("key3", make_query_result("q3"))
        assert len(cache) == 2
        # Oldest key should be evicted
        assert not await cache.has("key1")
        assert await cache.has("key2")
        assert await cache.has("key3")

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Test stats tracking."""
        cache = MemoryCache()
        await cache.set("key1", make_query_result())

        await cache.get("key1")  # hit
        await cache.get("key1")  # hit
        await cache.get("nonexistent")  # miss

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.size == 1

    @pytest.mark.asyncio
    async def test_does_not_cache_errors(self) -> None:
        """Test that errors are not cached."""
        cache = MemoryCache()
        error_result = QueryResult(
            query=Query(text="test", ground_truth_docs=[]),
            metric=RetrievalMetrics(precision=0, recall=0, mrr=0, ndcg=0, k=10),
            answer="",
            latency_ms=0,
            error=ValueError("test error"),
        )
        await cache.set("error_key", error_result)
        assert not await cache.has("error_key")
        assert len(cache) == 0

    def test_protocol_compliance(self) -> None:
        """Test that MemoryCache implements CacheProtocol."""
        cache = MemoryCache()
        assert isinstance(cache, CacheProtocol)


# ============================================================================
# DiskCache Tests
# ============================================================================


class TestDiskCache:
    """Tests for DiskCache."""

    @pytest.mark.asyncio
    async def test_get_missing_key(self) -> None:
        """Test getting a missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            result = await cache.get("nonexistent")
            assert result is None
            assert cache.stats().misses == 1

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """Test setting and getting a value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            query_result = make_query_result()
            await cache.set("key1", query_result)
            result = await cache.get("key1")
            assert result is not None
            assert result.answer == "Test answer"
            assert result.metric.precision == 0.8
            assert cache.stats().hits == 1

    @pytest.mark.asyncio
    async def test_persistence(self) -> None:
        """Test that data persists across cache instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write with first cache instance
            cache1 = DiskCache(tmpdir)
            await cache1.set("key1", make_query_result())

            # Read with second cache instance
            cache2 = DiskCache(tmpdir)
            result = await cache2.get("key1")
            assert result is not None
            assert result.answer == "Test answer"

    @pytest.mark.asyncio
    async def test_has(self) -> None:
        """Test checking if key exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            assert not await cache.has("key1")
            await cache.set("key1", make_query_result())
            assert await cache.has("key1")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test deleting a key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            await cache.set("key1", make_query_result())
            assert await cache.has("key1")
            await cache.delete("key1")
            assert not await cache.has("key1")

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing all keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            await cache.set("key1", make_query_result())
            await cache.set("key2", make_query_result())
            assert len(cache) == 2
            await cache.clear()
            assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_creates_directory(self) -> None:
        """Test that cache creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "subdir" / "cache"
            _cache = DiskCache(cache_dir)
            assert cache_dir.exists()

    @pytest.mark.asyncio
    async def test_does_not_cache_errors(self) -> None:
        """Test that errors are not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            error_result = QueryResult(
                query=Query(text="test", ground_truth_docs=[]),
                metric=RetrievalMetrics(precision=0, recall=0, mrr=0, ndcg=0, k=10),
                answer="",
                latency_ms=0,
                error=ValueError("test error"),
            )
            await cache.set("error_key", error_result)
            assert not await cache.has("error_key")

    def test_protocol_compliance(self) -> None:
        """Test that DiskCache implements CacheProtocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            assert isinstance(cache, CacheProtocol)


# ============================================================================
# Integration with Evaluate Tests
# ============================================================================


class MockRAG:
    """Mock RAG for testing cache integration."""

    def __init__(self) -> None:
        self.call_count = 0

    async def query(self, question: str) -> RAGResponse:
        from ragnarok_ai.core.types import Document
        from ragnarok_ai.core.types import RAGResponse as RAGResponseType

        self.call_count += 1
        return RAGResponseType(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


class TestEvaluateWithCache:
    """Tests for evaluate with caching."""

    @pytest.mark.asyncio
    async def test_cache_reduces_calls(self) -> None:
        """Test that caching reduces RAG calls."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = MockRAG()
        cache = MemoryCache()
        testset = TestSet(
            queries=[
                Query(text="q1", ground_truth_docs=["doc_1"]),
                Query(text="q2", ground_truth_docs=["doc_1"]),
            ]
        )

        # First evaluation - all calls
        await evaluate(rag, testset, cache=cache)
        assert rag.call_count == 2

        # Second evaluation - cached
        await evaluate(rag, testset, cache=cache)
        assert rag.call_count == 2  # No additional calls

        # Check cache stats
        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 2

    @pytest.mark.asyncio
    async def test_cache_with_parallel(self) -> None:
        """Test caching with parallel evaluation."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = MockRAG()
        cache = MemoryCache()
        testset = TestSet(
            queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(5)]
        )

        # First evaluation - all calls
        await evaluate(rag, testset, cache=cache, max_concurrency=5)
        assert rag.call_count == 5

        # Second evaluation - cached
        await evaluate(rag, testset, cache=cache, max_concurrency=5)
        assert rag.call_count == 5  # No additional calls

    @pytest.mark.asyncio
    async def test_pipeline_id_isolation(self) -> None:
        """Test that different pipeline IDs use different cache entries."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = MockRAG()
        cache = MemoryCache()
        testset = TestSet(queries=[Query(text="q1", ground_truth_docs=["doc_1"])])

        # Pipeline A
        await evaluate(rag, testset, cache=cache, pipeline_id="pipeline_a")
        assert rag.call_count == 1

        # Pipeline B - different cache key
        await evaluate(rag, testset, cache=cache, pipeline_id="pipeline_b")
        assert rag.call_count == 2

        # Pipeline A again - cached
        await evaluate(rag, testset, cache=cache, pipeline_id="pipeline_a")
        assert rag.call_count == 2


# ============================================================================
# Cache + Timeout/Retry Integration Tests
# ============================================================================


class TimeoutRAG:
    """Mock RAG that times out."""

    def __init__(self, delay: float = 1.0) -> None:
        self._delay = delay
        self.call_count = 0

    async def query(self, question: str) -> RAGResponse:
        import asyncio

        from ragnarok_ai.core.types import Document
        from ragnarok_ai.core.types import RAGResponse as RAGResponseType

        self.call_count += 1
        await asyncio.sleep(self._delay)
        return RAGResponseType(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


class FailThenSucceedRAG:
    """Mock RAG that fails a few times then succeeds."""

    def __init__(self, fail_count: int = 2) -> None:
        self._fail_count = fail_count
        self._attempts: dict[str, int] = {}
        self.call_count = 0

    async def query(self, question: str) -> RAGResponse:
        from ragnarok_ai.core.types import Document
        from ragnarok_ai.core.types import RAGResponse as RAGResponseType

        self.call_count += 1
        self._attempts[question] = self._attempts.get(question, 0) + 1
        if self._attempts[question] <= self._fail_count:
            raise ValueError(f"Simulated failure #{self._attempts[question]}")
        return RAGResponseType(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


class TestCacheWithTimeoutRetry:
    """Tests for cache integration with timeout and retry."""

    @pytest.mark.asyncio
    async def test_cache_not_store_timeouts(self) -> None:
        """Test that timeouts are not cached."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = TimeoutRAG(delay=1.0)
        cache = MemoryCache()
        testset = TestSet(queries=[Query(text="q1", ground_truth_docs=["doc_1"])])

        # First attempt - times out
        await evaluate(rag, testset, cache=cache, timeout=0.1, fail_fast=False)

        # Cache should be empty (timeout not cached)
        assert len(cache) == 0
        assert rag.call_count == 1

        # Second attempt - also times out (not cached)
        await evaluate(rag, testset, cache=cache, timeout=0.1, fail_fast=False)
        assert rag.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_not_store_errors_after_retry(self) -> None:
        """Test that errors after retry exhaustion are not cached."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = FailThenSucceedRAG(fail_count=10)  # Always fails
        cache = MemoryCache()
        testset = TestSet(queries=[Query(text="q1", ground_truth_docs=["doc_1"])])

        # First attempt with retries - all fail
        await evaluate(
            rag, testset, cache=cache, max_retries=2, retry_delay=0.01, fail_fast=False
        )

        # Cache should be empty (error not cached)
        assert len(cache) == 0
        # 3 attempts (1 initial + 2 retries)
        assert rag.call_count == 3

    @pytest.mark.asyncio
    async def test_cache_stores_retry_success(self) -> None:
        """Test that successful retries are cached."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = FailThenSucceedRAG(fail_count=2)  # Fails 2x, succeeds on 3rd
        cache = MemoryCache()
        testset = TestSet(queries=[Query(text="q1", ground_truth_docs=["doc_1"])])

        # First attempt with retries - succeeds on 3rd try
        result = await evaluate(
            rag, testset, cache=cache, max_retries=3, retry_delay=0.01
        )

        # Should have succeeded
        assert len(result.responses) == 1
        assert result.responses[0] == "Answer to q1"

        # Cache should have the result
        assert len(cache) == 1
        # 3 attempts (2 failures + 1 success)
        assert rag.call_count == 3

        # Second evaluation - cached
        await evaluate(rag, testset, cache=cache, max_retries=3, retry_delay=0.01)
        # No additional calls
        assert rag.call_count == 3

    @pytest.mark.asyncio
    async def test_cache_hit_skips_timeout_retry(self) -> None:
        """Test that cached results skip timeout/retry logic entirely."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        # First, populate cache with a fast RAG
        fast_rag = MockRAG()
        cache = MemoryCache()
        testset = TestSet(queries=[Query(text="q1", ground_truth_docs=["doc_1"])])

        await evaluate(fast_rag, testset, cache=cache)
        assert fast_rag.call_count == 1
        assert len(cache) == 1

        # Now use a slow RAG that would timeout
        slow_rag = TimeoutRAG(delay=10.0)

        # With very short timeout - but result is cached so it's fine
        result = await evaluate(slow_rag, testset, cache=cache, timeout=0.001)

        # Slow RAG not called (cache hit)
        assert slow_rag.call_count == 0
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_cache_parallel_with_timeout(self) -> None:
        """Test cache with parallel evaluation and timeout."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        rag = MockRAG()
        cache = MemoryCache()
        testset = TestSet(
            queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(5)]
        )

        # First evaluation with timeout
        await evaluate(
            rag, testset, cache=cache, max_concurrency=5, timeout=5.0
        )
        assert rag.call_count == 5
        assert len(cache) == 5

        # Second evaluation - all cached
        await evaluate(
            rag, testset, cache=cache, max_concurrency=5, timeout=5.0
        )
        assert rag.call_count == 5  # No additional calls
