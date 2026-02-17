"""Unit tests for the benchmarks module."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ragnarok_ai.benchmarks import (
    BenchmarkHistory,
    BenchmarkRecord,
    JSONFileStore,
    StorageProtocol,
)
from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_testset() -> TestSet:
    """Create a sample testset for testing."""
    return TestSet(
        queries=[
            Query(text="What is RAG?", ground_truth_docs=["doc1", "doc2"]),
            Query(text="How does it work?", ground_truth_docs=["doc3"]),
        ],
        name="test_set",
    )


@pytest.fixture
def sample_record() -> BenchmarkRecord:
    """Create a sample benchmark record."""
    return BenchmarkRecord(
        id="test-id-123",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        config_name="my-config",
        testset_hash="abc123def456",
        testset_name="test_set",
        metrics={"precision": 0.85, "recall": 0.78, "mrr": 0.90, "ndcg": 0.82},
        metadata={"version": "1.0.0"},
        is_baseline=False,
    )


@pytest.fixture
def sample_evaluation(sample_testset: TestSet) -> EvaluationResult:
    """Create a sample evaluation result."""
    metrics = RetrievalMetrics(precision=0.80, recall=0.70, mrr=0.90, ndcg=0.85, k=10)
    return EvaluationResult(
        testset=sample_testset,
        metrics=[metrics, metrics],
        responses=["Answer 1", "Answer 2"],
        query_results=[
            QueryResult(
                query=sample_testset.queries[0],
                metric=metrics,
                answer="Answer 1",
                latency_ms=100.0,
            ),
            QueryResult(
                query=sample_testset.queries[1],
                metric=metrics,
                answer="Answer 2",
                latency_ms=100.0,
            ),
        ],
        total_latency_ms=200.0,
    )


@pytest.fixture
def json_store(tmp_path: Path) -> JSONFileStore:
    """Create a JSONFileStore with temp path."""
    return JSONFileStore(tmp_path / "benchmarks.json")


# ============================================================================
# BenchmarkRecord Tests
# ============================================================================


class TestBenchmarkRecord:
    """Tests for BenchmarkRecord dataclass."""

    def test_to_dict(self, sample_record: BenchmarkRecord) -> None:
        """to_dict() returns correct dictionary."""
        data = sample_record.to_dict()

        assert data["id"] == "test-id-123"
        assert data["config_name"] == "my-config"
        assert data["testset_hash"] == "abc123def456"
        assert data["metrics"]["precision"] == 0.85
        assert data["metadata"]["version"] == "1.0.0"
        assert data["is_baseline"] is False
        assert "timestamp" in data

    def test_from_dict(self) -> None:
        """from_dict() creates correct record."""
        data = {
            "id": "abc123",
            "timestamp": "2024-01-15T10:30:00",
            "config_name": "test-config",
            "testset_hash": "deadbeef",
            "testset_name": "my_test",
            "metrics": {"precision": 0.90},
            "metadata": {"key": "value"},
            "is_baseline": True,
        }

        record = BenchmarkRecord.from_dict(data)

        assert record.id == "abc123"
        assert record.config_name == "test-config"
        assert record.testset_hash == "deadbeef"
        assert record.metrics["precision"] == 0.90
        assert record.metadata["key"] == "value"
        assert record.is_baseline is True

    def test_from_dict_with_defaults(self) -> None:
        """from_dict() uses defaults for optional fields."""
        data = {
            "id": "abc123",
            "timestamp": "2024-01-15T10:30:00",
            "config_name": "test-config",
            "testset_hash": "deadbeef",
            "testset_name": "my_test",
            "metrics": {"precision": 0.90},
        }

        record = BenchmarkRecord.from_dict(data)

        assert record.metadata == {}
        assert record.is_baseline is False


# ============================================================================
# JSONFileStore Tests
# ============================================================================


class TestJSONFileStore:
    """Tests for JSONFileStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, json_store: JSONFileStore, sample_record: BenchmarkRecord) -> None:
        """save() and get() work correctly."""
        await json_store.save(sample_record)
        retrieved = await json_store.get(sample_record.id)

        assert retrieved is not None
        assert retrieved.id == sample_record.id
        assert retrieved.config_name == sample_record.config_name
        assert retrieved.metrics == sample_record.metrics

    @pytest.mark.asyncio
    async def test_get_not_found(self, json_store: JSONFileStore) -> None:
        """get() returns None for missing record."""
        result = await json_store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, json_store: JSONFileStore) -> None:
        """list() returns all records."""
        records = [
            BenchmarkRecord(
                id=f"id-{i}",
                timestamp=datetime(2024, 1, i + 1),
                config_name=f"config-{i % 2}",
                testset_hash="hash",
                testset_name="test",
                metrics={},
            )
            for i in range(5)
        ]

        for r in records:
            await json_store.save(r)

        result = await json_store.list()
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_list_by_config(self, json_store: JSONFileStore) -> None:
        """list() filters by config_name."""
        for i in range(4):
            await json_store.save(
                BenchmarkRecord(
                    id=f"id-{i}",
                    timestamp=datetime(2024, 1, i + 1),
                    config_name=f"config-{i % 2}",
                    testset_hash="hash",
                    testset_name="test",
                    metrics={},
                )
            )

        result = await json_store.list(config_name="config-0")
        assert len(result) == 2
        assert all(r.config_name == "config-0" for r in result)

    @pytest.mark.asyncio
    async def test_list_limit(self, json_store: JSONFileStore) -> None:
        """list() respects limit parameter."""
        for i in range(10):
            await json_store.save(
                BenchmarkRecord(
                    id=f"id-{i}",
                    timestamp=datetime(2024, 1, i + 1),
                    config_name="config",
                    testset_hash="hash",
                    testset_name="test",
                    metrics={},
                )
            )

        result = await json_store.list(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_delete(self, json_store: JSONFileStore, sample_record: BenchmarkRecord) -> None:
        """delete() removes record."""
        await json_store.save(sample_record)
        deleted = await json_store.delete(sample_record.id)

        assert deleted is True
        assert await json_store.get(sample_record.id) is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, json_store: JSONFileStore) -> None:
        """delete() returns False for missing record."""
        result = await json_store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_baseline(self, json_store: JSONFileStore) -> None:
        """set_baseline() marks record as baseline."""
        record = BenchmarkRecord(
            id="baseline-id",
            timestamp=datetime.now(),
            config_name="my-config",
            testset_hash="hash",
            testset_name="test",
            metrics={},
            is_baseline=False,
        )
        await json_store.save(record)
        await json_store.set_baseline(record.id)

        retrieved = await json_store.get(record.id)
        assert retrieved is not None
        assert retrieved.is_baseline is True

    @pytest.mark.asyncio
    async def test_set_baseline_unsets_previous(self, json_store: JSONFileStore) -> None:
        """set_baseline() unsets previous baseline for same config."""
        # Save two records for same config
        record1 = BenchmarkRecord(
            id="id-1",
            timestamp=datetime.now(),
            config_name="my-config",
            testset_hash="hash",
            testset_name="test",
            metrics={},
        )
        record2 = BenchmarkRecord(
            id="id-2",
            timestamp=datetime.now(),
            config_name="my-config",
            testset_hash="hash",
            testset_name="test",
            metrics={},
        )
        await json_store.save(record1)
        await json_store.save(record2)

        # Set first as baseline
        await json_store.set_baseline("id-1")
        assert (await json_store.get("id-1")).is_baseline is True  # type: ignore[union-attr]

        # Set second as baseline
        await json_store.set_baseline("id-2")
        assert (await json_store.get("id-1")).is_baseline is False  # type: ignore[union-attr]
        assert (await json_store.get("id-2")).is_baseline is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_set_baseline_not_found(self, json_store: JSONFileStore) -> None:
        """set_baseline() raises ValueError for missing record."""
        with pytest.raises(ValueError, match="not found"):
            await json_store.set_baseline("nonexistent")

    @pytest.mark.asyncio
    async def test_get_baseline(self, json_store: JSONFileStore) -> None:
        """get_baseline() returns the baseline record."""
        record = BenchmarkRecord(
            id="baseline-id",
            timestamp=datetime.now(),
            config_name="my-config",
            testset_hash="hash",
            testset_name="test",
            metrics={},
        )
        await json_store.save(record)
        await json_store.set_baseline(record.id)

        baseline = await json_store.get_baseline("my-config")
        assert baseline is not None
        assert baseline.id == record.id

    @pytest.mark.asyncio
    async def test_get_baseline_not_set(self, json_store: JSONFileStore) -> None:
        """get_baseline() returns None when no baseline set."""
        result = await json_store.get_baseline("no-baseline-config")
        assert result is None

    @pytest.mark.asyncio
    async def test_max_records_rotation(self, tmp_path: Path) -> None:
        """Records are rotated when max_records exceeded."""
        store = JSONFileStore(tmp_path / "benchmarks.json", max_records=3)

        for i in range(5):
            await store.save(
                BenchmarkRecord(
                    id=f"id-{i}",
                    timestamp=datetime(2024, 1, i + 1),
                    config_name="config",
                    testset_hash="hash",
                    testset_name="test",
                    metrics={},
                )
            )

        records = await store.list()
        assert len(records) == 3
        # Should keep most recent (id-2, id-3, id-4)
        ids = [r.id for r in records]
        assert "id-0" not in ids
        assert "id-1" not in ids

    @pytest.mark.asyncio
    async def test_rotation_preserves_baselines(self, tmp_path: Path) -> None:
        """Baselines are preserved during rotation."""
        store = JSONFileStore(tmp_path / "benchmarks.json", max_records=3)

        # Save first record and mark as baseline
        await store.save(
            BenchmarkRecord(
                id="baseline-id",
                timestamp=datetime(2024, 1, 1),
                config_name="config",
                testset_hash="hash",
                testset_name="test",
                metrics={},
            )
        )
        await store.set_baseline("baseline-id")

        # Save more records to trigger rotation
        for i in range(5):
            await store.save(
                BenchmarkRecord(
                    id=f"id-{i}",
                    timestamp=datetime(2024, 1, i + 2),
                    config_name="config",
                    testset_hash="hash",
                    testset_name="test",
                    metrics={},
                )
            )

        # Baseline should be preserved
        baseline = await store.get_baseline("config")
        assert baseline is not None
        assert baseline.id == "baseline-id"

    @pytest.mark.asyncio
    async def test_atomic_write(self, tmp_path: Path) -> None:
        """Atomic write creates file correctly."""
        store = JSONFileStore(tmp_path / "benchmarks.json")

        await store.save(
            BenchmarkRecord(
                id="test-id",
                timestamp=datetime.now(),
                config_name="config",
                testset_hash="hash",
                testset_name="test",
                metrics={"precision": 0.85},
            )
        )

        # File should exist and be valid JSON
        file_path = tmp_path / "benchmarks.json"
        assert file_path.exists()

        data = json.loads(file_path.read_text())
        assert "records" in data
        assert len(data["records"]) == 1

    @pytest.mark.asyncio
    async def test_corrupted_file_handled(self, tmp_path: Path) -> None:
        """Corrupted JSON file is handled gracefully."""
        file_path = tmp_path / "benchmarks.json"
        file_path.write_text("{ invalid json }")

        store = JSONFileStore(file_path)

        # Should return empty list, not crash
        records = await store.list()
        assert records == []

        # Should be able to save new record
        await store.save(
            BenchmarkRecord(
                id="new-id",
                timestamp=datetime.now(),
                config_name="config",
                testset_hash="hash",
                testset_name="test",
                metrics={},
            )
        )

        records = await store.list()
        assert len(records) == 1


# ============================================================================
# BenchmarkHistory Tests
# ============================================================================


class TestBenchmarkHistory:
    """Tests for BenchmarkHistory."""

    @pytest.mark.asyncio
    async def test_record(self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet) -> None:
        """record() creates and stores benchmark record."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        record = await history.record(
            result=sample_evaluation,
            config_name="my-config",
            testset=sample_testset,
            metadata={"version": "1.0.0"},
        )

        assert record.config_name == "my-config"
        assert record.testset_name == "test_set"
        assert record.metadata["version"] == "1.0.0"
        assert "precision" in record.metrics
        assert "latency_ms" in record.metrics

        # Should be retrievable
        retrieved = await store.get(record.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_latest(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """get_latest() returns most recent record."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        # Record multiple
        await history.record(sample_evaluation, "my-config", sample_testset)
        record2 = await history.record(sample_evaluation, "my-config", sample_testset)

        latest = await history.get_latest("my-config")
        assert latest is not None
        assert latest.id == record2.id

    @pytest.mark.asyncio
    async def test_get_latest_not_found(self, tmp_path: Path) -> None:
        """get_latest() returns None when no records."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        result = await history.get_latest("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_history(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """get_history() returns records for config."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        for _ in range(3):
            await history.record(sample_evaluation, "my-config", sample_testset)

        records = await history.get_history("my-config")
        assert len(records) == 3

    @pytest.mark.asyncio
    async def test_get_baseline(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """get_baseline() returns baseline record."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        record = await history.record(sample_evaluation, "my-config", sample_testset)
        await history.set_baseline(record.id)

        baseline = await history.get_baseline("my-config")
        assert baseline is not None
        assert baseline.id == record.id

    @pytest.mark.asyncio
    async def test_set_baseline(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """set_baseline() marks record as baseline."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        record = await history.record(sample_evaluation, "my-config", sample_testset)
        await history.set_baseline(record.id)

        retrieved = await store.get(record.id)
        assert retrieved is not None
        assert retrieved.is_baseline is True

    @pytest.mark.asyncio
    async def test_compare_to_baseline_no_baseline(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """compare_to_baseline() returns None when no baseline."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        result = await history.compare_to_baseline("my-config", sample_evaluation, sample_testset)
        assert result is None

    @pytest.mark.asyncio
    async def test_compare_to_baseline_success(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """compare_to_baseline() returns RegressionResult."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        # Record baseline
        record = await history.record(sample_evaluation, "my-config", sample_testset)
        await history.set_baseline(record.id)

        # Compare (same metrics = no regression)
        result = await history.compare_to_baseline("my-config", sample_evaluation, sample_testset)
        assert result is not None
        assert not result.has_regressions

    @pytest.mark.asyncio
    async def test_compare_to_baseline_hash_mismatch(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet
    ) -> None:
        """compare_to_baseline() raises ValueError on hash mismatch."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        history = BenchmarkHistory(store=store)

        # Record baseline
        record = await history.record(sample_evaluation, "my-config", sample_testset)
        await history.set_baseline(record.id)

        # Create different testset
        different_testset = TestSet(
            queries=[Query(text="Different question", ground_truth_docs=["doc5"])],
            name="different",
        )

        with pytest.raises(ValueError, match="hash mismatch"):
            await history.compare_to_baseline("my-config", sample_evaluation, different_testset)

    @pytest.mark.asyncio
    async def test_default_store(
        self, tmp_path: Path, sample_evaluation: EvaluationResult, sample_testset: TestSet, monkeypatch: Any
    ) -> None:
        """BenchmarkHistory uses default JSONFileStore."""
        # Change working directory to tmp_path
        monkeypatch.chdir(tmp_path)

        history = BenchmarkHistory()

        record = await history.record(sample_evaluation, "my-config", sample_testset)
        assert record is not None

        # File should be created
        default_path = tmp_path / ".ragnarok" / "benchmarks.json"
        assert default_path.exists()


# ============================================================================
# StorageProtocol Tests
# ============================================================================


class TestStorageProtocol:
    """Tests for StorageProtocol."""

    def test_json_file_store_implements_protocol(self, tmp_path: Path) -> None:
        """JSONFileStore implements StorageProtocol."""
        store = JSONFileStore(tmp_path / "benchmarks.json")
        assert isinstance(store, StorageProtocol)
