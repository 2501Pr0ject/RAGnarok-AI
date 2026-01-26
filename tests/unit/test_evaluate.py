import pytest

from ragnarok_ai.core.evaluate import EvaluationResult, evaluate
from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.core.types import Document, Query, RAGResponse, TestSet


class MockRAG:
    """Mock RAG pipeline for testing."""

    async def query(self, question: str) -> RAGResponse:
        if "fail" in question:
            raise ValueError("Simulated failure")

        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[
                Document(id="doc_1", content="Content 1"),
                Document(id="doc_2", content="Content 2"),
            ],
        )


@pytest.mark.asyncio
async def test_evaluate_success():
    rag = MockRAG()
    testset = TestSet(
        queries=[
            Query(text="q1", ground_truth_docs=["doc_1"]),
            Query(text="q2", ground_truth_docs=["doc_3"]),
        ]
    )

    result = await evaluate(rag, testset)

    assert isinstance(result, EvaluationResult)
    assert len(result.responses) == 2
    assert len(result.metrics) == 2
    summary = result.summary()
    assert summary["num_queries"] == 2
    assert "precision" in summary
    assert "recall" in summary


@pytest.mark.asyncio
async def test_evaluate_failure():
    rag = MockRAG()
    testset = TestSet(
        queries=[
            Query(text="fail this query", ground_truth_docs=[]),
        ]
    )

    with pytest.raises(EvaluationError) as excinfo:
        await evaluate(rag, testset)

    assert "Failed to evaluate query" in str(excinfo.value)
    assert "fail this query" in str(excinfo.value)


@pytest.mark.asyncio
async def test_evaluate_empty_testset():
    rag = MockRAG()
    testset = TestSet(queries=[])

    result = await evaluate(rag, testset)

    assert len(result.responses) == 0
    assert result.summary() == {}


# ============================================================================
# Parallel Evaluation Tests
# ============================================================================


class SlowMockRAG:
    """Mock RAG that simulates slow responses."""

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay
        self._call_count = 0

    async def query(self, question: str) -> RAGResponse:
        import asyncio

        self._call_count += 1
        await asyncio.sleep(self._delay)
        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


@pytest.mark.asyncio
async def test_evaluate_parallel_basic():
    """Test basic parallel evaluation."""
    rag = SlowMockRAG(delay=0.05)
    testset = TestSet(
        queries=[
            Query(text=f"q{i}", ground_truth_docs=["doc_1"])
            for i in range(5)
        ]
    )

    result = await evaluate(rag, testset, max_concurrency=5)

    assert len(result.responses) == 5
    assert len(result.metrics) == 5


@pytest.mark.asyncio
async def test_evaluate_parallel_faster_than_sequential():
    """Test that parallel is faster than sequential."""
    import time

    rag = SlowMockRAG(delay=0.05)
    testset = TestSet(
        queries=[
            Query(text=f"q{i}", ground_truth_docs=["doc_1"])
            for i in range(4)
        ]
    )

    # Sequential
    start = time.perf_counter()
    await evaluate(rag, testset, max_concurrency=1)
    sequential_time = time.perf_counter() - start

    # Parallel
    start = time.perf_counter()
    await evaluate(rag, testset, max_concurrency=4)
    parallel_time = time.perf_counter() - start

    # Parallel should be significantly faster
    assert parallel_time < sequential_time * 0.75


@pytest.mark.asyncio
async def test_evaluate_parallel_preserves_order():
    """Test that parallel evaluation preserves result order."""

    class OrderedMockRAG:
        async def query(self, question: str) -> RAGResponse:
            import asyncio
            import random

            await asyncio.sleep(random.uniform(0.01, 0.05))
            return RAGResponse(
                answer=question,
                retrieved_docs=[Document(id="doc_1", content="Content")],
            )

    rag = OrderedMockRAG()
    queries = [f"query_{i}" for i in range(10)]
    testset = TestSet(
        queries=[Query(text=q, ground_truth_docs=["doc_1"]) for q in queries]
    )

    result = await evaluate(rag, testset, max_concurrency=5)

    # Responses should match query order
    for i, response in enumerate(result.responses):
        assert response == f"query_{i}"


@pytest.mark.asyncio
async def test_evaluate_parallel_with_errors_fail_fast():
    """Test parallel evaluation with errors and fail_fast=True."""

    class FailingRAG:
        def __init__(self) -> None:
            self._count = 0

        async def query(self, question: str) -> RAGResponse:
            self._count += 1
            if self._count == 3:
                raise ValueError("Simulated failure")
            return RAGResponse(
                answer=f"Answer to {question}",
                retrieved_docs=[Document(id="doc_1", content="Content")],
            )

    rag = FailingRAG()
    testset = TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(5)]
    )

    with pytest.raises(EvaluationError):
        await evaluate(rag, testset, max_concurrency=5, fail_fast=True)


@pytest.mark.asyncio
async def test_evaluate_parallel_with_errors_continue():
    """Test parallel evaluation continues on error when fail_fast=False."""

    class FailingRAG:
        async def query(self, question: str) -> RAGResponse:
            if "fail" in question:
                raise ValueError("Simulated failure")
            return RAGResponse(
                answer=f"Answer to {question}",
                retrieved_docs=[Document(id="doc_1", content="Content")],
            )

    rag = FailingRAG()
    testset = TestSet(
        queries=[
            Query(text="q1", ground_truth_docs=["doc_1"]),
            Query(text="fail_q2", ground_truth_docs=["doc_1"]),
            Query(text="q3", ground_truth_docs=["doc_1"]),
        ]
    )

    result = await evaluate(rag, testset, max_concurrency=3, fail_fast=False)

    # Should have 2 successful results, 1 error
    assert len(result.responses) == 2
    assert len(result.errors) == 1


# ============================================================================
# Progress Callback Tests
# ============================================================================


@pytest.mark.asyncio
async def test_evaluate_with_progress_callback():
    """Test evaluation with progress callback."""
    rag = MockRAG()
    testset = TestSet(
        queries=[
            Query(text=f"q{i}", ground_truth_docs=["doc_1"])
            for i in range(3)
        ]
    )

    progress_updates = []

    def on_progress(info):
        progress_updates.append((info.current, info.total))

    result = await evaluate(rag, testset, on_progress=on_progress)

    assert len(result.responses) == 3
    assert len(progress_updates) == 3
    assert progress_updates[-1] == (3, 3)


@pytest.mark.asyncio
async def test_evaluate_parallel_with_progress_callback():
    """Test parallel evaluation with progress callback."""
    rag = SlowMockRAG(delay=0.02)
    testset = TestSet(
        queries=[
            Query(text=f"q{i}", ground_truth_docs=["doc_1"])
            for i in range(5)
        ]
    )

    progress_updates = []

    def on_progress(info):
        progress_updates.append(info.current)

    result = await evaluate(rag, testset, max_concurrency=5, on_progress=on_progress)

    assert len(result.responses) == 5
    assert len(progress_updates) == 5
    # All numbers 1-5 should be present (order may vary)
    assert set(progress_updates) == {1, 2, 3, 4, 5}


@pytest.mark.asyncio
async def test_evaluate_with_async_progress_callback():
    """Test evaluation with async progress callback."""
    import asyncio

    rag = MockRAG()
    testset = TestSet(
        queries=[Query(text="q1", ground_truth_docs=["doc_1"])]
    )

    callback_called = False

    async def async_callback(_info):
        nonlocal callback_called
        await asyncio.sleep(0.01)
        callback_called = True

    await evaluate(rag, testset, on_progress=async_callback)

    assert callback_called


# ============================================================================
# Timeout Tests
# ============================================================================


class TimeoutMockRAG:
    """Mock RAG that takes too long to respond."""

    def __init__(self, delay: float = 1.0) -> None:
        self._delay = delay

    async def query(self, question: str) -> RAGResponse:
        import asyncio

        await asyncio.sleep(self._delay)
        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


@pytest.mark.asyncio
async def test_evaluate_timeout():
    """Test that queries timeout after the specified duration."""
    from ragnarok_ai.core.evaluate import QueryTimeoutError

    rag = TimeoutMockRAG(delay=1.0)  # 1 second delay
    testset = TestSet(
        queries=[Query(text="q1", ground_truth_docs=["doc_1"])]
    )

    # Set timeout to 0.1s, query takes 1s -> should timeout
    with pytest.raises(EvaluationError) as excinfo:
        await evaluate(rag, testset, timeout=0.1, fail_fast=True)

    assert "timed out" in str(excinfo.value).lower() or isinstance(excinfo.value.__cause__, QueryTimeoutError)


@pytest.mark.asyncio
async def test_evaluate_timeout_parallel():
    """Test timeout in parallel evaluation."""
    rag = TimeoutMockRAG(delay=1.0)
    testset = TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(3)]
    )

    # With fail_fast=False, should collect errors
    result = await evaluate(
        rag, testset, timeout=0.1, max_concurrency=3, fail_fast=False
    )

    # All queries should have timed out
    assert len(result.errors) == 3
    assert len(result.responses) == 0


# ============================================================================
# Retry Tests
# ============================================================================


class FailThenSucceedRAG:
    """Mock RAG that fails a specified number of times before succeeding."""

    def __init__(self, fail_count: int = 2) -> None:
        self._fail_count = fail_count
        self._attempts: dict[str, int] = {}

    async def query(self, question: str) -> RAGResponse:
        self._attempts[question] = self._attempts.get(question, 0) + 1
        if self._attempts[question] <= self._fail_count:
            raise ValueError(f"Simulated failure #{self._attempts[question]}")
        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


@pytest.mark.asyncio
async def test_evaluate_retry_success():
    """Test that retry eventually succeeds after transient failures."""
    rag = FailThenSucceedRAG(fail_count=2)  # Fail 2 times, succeed on 3rd
    testset = TestSet(
        queries=[Query(text="q1", ground_truth_docs=["doc_1"])]
    )

    # 3 retries (4 total attempts) should be enough
    result = await evaluate(
        rag, testset, max_retries=3, retry_delay=0.01
    )

    assert len(result.responses) == 1
    assert result.responses[0] == "Answer to q1"
    assert rag._attempts["q1"] == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_evaluate_retry_exhausted():
    """Test that evaluation fails when all retries are exhausted."""
    rag = FailThenSucceedRAG(fail_count=5)  # Always fails within our retry limit
    testset = TestSet(
        queries=[Query(text="q1", ground_truth_docs=["doc_1"])]
    )

    # Only 2 retries (3 total attempts) - not enough
    with pytest.raises(EvaluationError):
        await evaluate(
            rag, testset, max_retries=2, retry_delay=0.01, fail_fast=True
        )


@pytest.mark.asyncio
async def test_evaluate_retry_parallel():
    """Test retry in parallel evaluation."""
    rag = FailThenSucceedRAG(fail_count=1)  # Fail once, succeed on 2nd
    testset = TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(3)]
    )

    result = await evaluate(
        rag, testset, max_retries=2, retry_delay=0.01, max_concurrency=3
    )

    # All should succeed after retry
    assert len(result.responses) == 3


# ============================================================================
# Concurrency Tests
# ============================================================================


class ConcurrencyTrackingRAG:
    """Mock RAG that tracks concurrent execution."""

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay
        self._max_concurrent = 0
        self._current_concurrent = 0
        self._lock = None

    async def query(self, question: str) -> RAGResponse:
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self._current_concurrent += 1
            if self._current_concurrent > self._max_concurrent:
                self._max_concurrent = self._current_concurrent

        await asyncio.sleep(self._delay)

        async with self._lock:
            self._current_concurrent -= 1

        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


@pytest.mark.asyncio
async def test_concurrency_limit_respected():
    """Test that max_concurrency limit is respected."""
    rag = ConcurrencyTrackingRAG(delay=0.05)
    testset = TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(10)]
    )

    # Set max_concurrency to 3
    await evaluate(rag, testset, max_concurrency=3)

    # Should never exceed 3 concurrent queries
    assert rag._max_concurrent <= 3


@pytest.mark.asyncio
async def test_concurrency_limit_allows_full_parallelism():
    """Test that queries run in parallel up to the limit."""
    rag = ConcurrencyTrackingRAG(delay=0.05)
    testset = TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(5)]
    )

    # Set max_concurrency to 5 (same as number of queries)
    await evaluate(rag, testset, max_concurrency=5)

    # Should reach full concurrency
    assert rag._max_concurrent >= 4  # Allow some timing variance


@pytest.mark.asyncio
async def test_invalid_concurrency_clamped_to_one():
    """Test that max_concurrency < 1 is clamped to 1."""
    rag = ConcurrencyTrackingRAG(delay=0.01)
    testset = TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(3)]
    )

    # Invalid values should be clamped to 1 (sequential)
    await evaluate(rag, testset, max_concurrency=0)
    assert rag._max_concurrent == 1

    rag._max_concurrent = 0  # Reset
    await evaluate(rag, testset, max_concurrency=-5)
    assert rag._max_concurrent == 1
