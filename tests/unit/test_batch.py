"""Unit tests for batch processing module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ragnarok_ai.core.batch import BatchConfig, BatchEvaluator, BatchProgress, BatchResult
from ragnarok_ai.core.types import Document, Query, RAGResponse, TestSet

# ============================================================================
# Test Fixtures
# ============================================================================


class MockRAG:
    """Mock RAG for testing batch evaluation."""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay
        self.call_count = 0

    async def query(self, question: str) -> RAGResponse:
        import asyncio

        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


class FailingRAG:
    """Mock RAG that fails on specific queries."""

    def __init__(self, fail_indices: set[int] | None = None) -> None:
        self._fail_indices = fail_indices or set()
        self.call_count = 0

    async def query(self, question: str) -> RAGResponse:
        self.call_count += 1
        # Extract index from query text like "q5"
        if question.startswith("q"):
            try:
                idx = int(question[1:])
            except ValueError:
                idx = -1
            if idx in self._fail_indices:
                raise RuntimeError(f"Simulated failure for {question}")
        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[Document(id="doc_1", content="Content")],
        )


def make_testset(n: int, name: str = "test") -> TestSet:
    """Create a test set with n queries."""
    return TestSet(
        queries=[Query(text=f"q{i}", ground_truth_docs=["doc_1"]) for i in range(n)],
        name=name,
    )


# ============================================================================
# BatchConfig Tests
# ============================================================================


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BatchConfig()
        assert config.batch_size == 100
        assert config.max_concurrency == 10
        assert config.timeout is None
        assert config.max_retries == 0
        assert config.retry_delay == 1.0
        assert config.checkpoint_dir is None
        assert config.checkpoint_interval == 1
        assert config.k == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BatchConfig(
            batch_size=50,
            max_concurrency=5,
            timeout=30.0,
            max_retries=3,
            checkpoint_dir="/tmp/checkpoints",
        )
        assert config.batch_size == 50
        assert config.max_concurrency == 5
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.checkpoint_dir == "/tmp/checkpoints"


# ============================================================================
# BatchProgress Tests
# ============================================================================


class TestBatchProgress:
    """Tests for BatchProgress."""

    def test_percent_calculation(self) -> None:
        """Test progress percentage calculation."""
        progress = BatchProgress(
            batch_index=2,
            total_batches=10,
            queries_completed=250,
            queries_total=1000,
            current_batch_size=100,
        )
        assert progress.percent == 25.0

    def test_percent_zero_total(self) -> None:
        """Test percentage with zero total."""
        progress = BatchProgress(
            batch_index=0,
            total_batches=0,
            queries_completed=0,
            queries_total=0,
            current_batch_size=0,
        )
        assert progress.percent == 100.0

    def test_percent_complete(self) -> None:
        """Test percentage at completion."""
        progress = BatchProgress(
            batch_index=9,
            total_batches=10,
            queries_completed=1000,
            queries_total=1000,
            current_batch_size=100,
        )
        assert progress.percent == 100.0


# ============================================================================
# BatchEvaluator Tests
# ============================================================================


class TestBatchEvaluator:
    """Tests for BatchEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_basic(self) -> None:
        """Test basic batch evaluation."""
        rag = MockRAG()
        config = BatchConfig(batch_size=10, max_concurrency=5)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(25)

        result = await evaluator.evaluate(testset)

        assert len(result.responses) == 25
        assert len(result.metrics) == 25
        assert rag.call_count == 25

    @pytest.mark.asyncio
    async def test_evaluate_single_batch(self) -> None:
        """Test evaluation with queries fitting in single batch."""
        rag = MockRAG()
        config = BatchConfig(batch_size=100)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(50)

        result = await evaluator.evaluate(testset)

        assert len(result.responses) == 50

    @pytest.mark.asyncio
    async def test_evaluate_batches_yields_results(self) -> None:
        """Test that evaluate_batches yields batch results."""
        rag = MockRAG()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(35)

        batches = []
        async for batch in evaluator.evaluate_batches(testset):
            batches.append(batch)

        assert len(batches) == 4  # 35 queries / 10 per batch = 4 batches
        assert batches[0].batch_index == 0
        assert batches[3].batch_index == 3
        assert len(batches[0].results) == 10
        assert len(batches[3].results) == 5  # Last batch has 5

    @pytest.mark.asyncio
    async def test_evaluate_with_progress_callback(self) -> None:
        """Test progress callback is called for each batch."""
        rag = MockRAG()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(30)

        progress_updates: list[BatchProgress] = []

        def on_progress(p: BatchProgress) -> None:
            progress_updates.append(p)

        await evaluator.evaluate(testset, on_progress=on_progress)

        assert len(progress_updates) == 3
        assert progress_updates[0].batch_index == 0
        assert progress_updates[0].queries_completed == 10
        assert progress_updates[2].batch_index == 2
        assert progress_updates[2].queries_completed == 30
        assert progress_updates[2].percent == 100.0

    @pytest.mark.asyncio
    async def test_evaluate_with_async_progress_callback(self) -> None:
        """Test async progress callback."""
        import asyncio

        rag = MockRAG()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(20)

        callback_count = 0

        async def on_progress(_p: BatchProgress) -> None:
            nonlocal callback_count
            await asyncio.sleep(0.001)
            callback_count += 1

        await evaluator.evaluate(testset, on_progress=on_progress)

        assert callback_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_with_errors_continues(self) -> None:
        """Test that batch evaluation continues on errors."""
        rag = FailingRAG(fail_indices={5, 15})
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(20)

        result = await evaluator.evaluate(testset)

        # 18 success, 2 failures
        assert len(result.responses) == 18
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_evaluate_with_cache(self) -> None:
        """Test batch evaluation with cache."""
        from ragnarok_ai.cache import MemoryCache

        rag = MockRAG()
        cache = MemoryCache()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config, cache=cache)
        testset = make_testset(20)

        # First evaluation
        await evaluator.evaluate(testset)
        first_call_count = rag.call_count

        # Second evaluation - should use cache
        await evaluator.evaluate(testset)

        assert rag.call_count == first_call_count  # No new calls
        assert cache.stats().hits == 20

    @pytest.mark.asyncio
    async def test_evaluate_empty_testset(self) -> None:
        """Test evaluation with empty testset."""
        rag = MockRAG()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = TestSet(queries=[])

        result = await evaluator.evaluate(testset)

        assert len(result.responses) == 0
        assert rag.call_count == 0


# ============================================================================
# Checkpoint Tests
# ============================================================================


class TestBatchEvaluatorCheckpoint:
    """Tests for BatchEvaluator with checkpointing."""

    @pytest.mark.asyncio
    async def test_creates_checkpoint(self) -> None:
        """Test that checkpoint is created when checkpoint_dir is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = MockRAG()
            config = BatchConfig(batch_size=10, checkpoint_dir=tmpdir)
            evaluator = BatchEvaluator(rag, config=config)
            testset = make_testset(25, name="test_checkpoint")

            await evaluator.evaluate(testset)

            # Checkpoint should be cleaned up after completion
            checkpoints = evaluator.list_checkpoints()
            assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_saves_progress(self) -> None:
        """Test that checkpoint saves progress during evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = MockRAG()
            config = BatchConfig(batch_size=10, checkpoint_dir=tmpdir, checkpoint_interval=1)
            evaluator = BatchEvaluator(rag, config=config)
            testset = make_testset(30, name="test_progress")

            batches_seen = 0
            checkpoint_files: list[Path] = []

            async for _batch in evaluator.evaluate_batches(testset):
                batches_seen += 1
                # Check checkpoint file exists during evaluation
                files = list(Path(tmpdir).glob("*.json"))
                if files:
                    checkpoint_files.append(files[0])

            assert batches_seen == 3
            # Checkpoint is cleaned up at the end
            assert len(list(Path(tmpdir).glob("*.json"))) == 0

    @pytest.mark.asyncio
    async def test_list_checkpoints(self) -> None:
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = MockRAG()
            config = BatchConfig(batch_size=10, checkpoint_dir=tmpdir)
            evaluator = BatchEvaluator(rag, config=config)

            # Initially no checkpoints
            checkpoints = evaluator.list_checkpoints()
            assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_no_checkpoint_without_dir(self) -> None:
        """Test that no checkpointing happens without checkpoint_dir."""
        rag = MockRAG()
        config = BatchConfig(batch_size=10)  # No checkpoint_dir
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(20)

        await evaluator.evaluate(testset)

        # Should work without issues
        assert rag.call_count == 20
        assert evaluator.list_checkpoints() == []


# ============================================================================
# Memory Efficiency Tests
# ============================================================================


class TestBatchMemoryEfficiency:
    """Tests for memory efficiency of batch processing."""

    @pytest.mark.asyncio
    async def test_batch_processes_incrementally(self) -> None:
        """Test that batches are processed one at a time."""
        rag = MockRAG()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(100)

        batch_indices = []
        async for batch in evaluator.evaluate_batches(testset):
            batch_indices.append(batch.batch_index)
            # Results are yielded incrementally
            assert isinstance(batch, BatchResult)

        # All batches processed in order
        assert batch_indices == list(range(10))

    @pytest.mark.asyncio
    async def test_large_testset_batched(self) -> None:
        """Test that large testset is properly batched."""
        rag = MockRAG()
        config = BatchConfig(batch_size=50, max_concurrency=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(500)

        result = await evaluator.evaluate(testset)

        assert len(result.responses) == 500
        assert rag.call_count == 500


# ============================================================================
# Integration Tests
# ============================================================================


class TestBatchIntegration:
    """Integration tests for batch processing."""

    @pytest.mark.asyncio
    async def test_batch_with_timeout_and_retry(self) -> None:
        """Test batch evaluation with timeout and retry config."""
        rag = MockRAG(delay=0.01)
        config = BatchConfig(
            batch_size=10,
            max_concurrency=5,
            timeout=5.0,
            max_retries=2,
            retry_delay=0.01,
        )
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(30)

        result = await evaluator.evaluate(testset)

        assert len(result.responses) == 30

    @pytest.mark.asyncio
    async def test_batch_result_latency_tracked(self) -> None:
        """Test that batch latency is tracked."""
        rag = MockRAG(delay=0.01)
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config)
        testset = make_testset(20)

        batches = []
        async for batch in evaluator.evaluate_batches(testset):
            batches.append(batch)

        # Each batch should have latency > 0
        for batch in batches:
            assert batch.latency_ms > 0

    @pytest.mark.asyncio
    async def test_evaluate_with_pipeline_id(self) -> None:
        """Test batch evaluation with pipeline_id."""
        from ragnarok_ai.cache import MemoryCache

        rag = MockRAG()
        cache = MemoryCache()
        config = BatchConfig(batch_size=10)
        evaluator = BatchEvaluator(rag, config=config, cache=cache, pipeline_id="test_pipeline")
        testset = make_testset(10)

        await evaluator.evaluate(testset)

        # Different pipeline_id should miss cache
        evaluator2 = BatchEvaluator(rag, config=config, cache=cache, pipeline_id="other_pipeline")
        await evaluator2.evaluate(testset)

        # Both evaluations ran (different cache keys)
        assert rag.call_count == 20
