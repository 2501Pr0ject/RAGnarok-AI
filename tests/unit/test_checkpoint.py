"""Unit tests for the checkpointing system."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ragnarok_ai.core import (
    CheckpointData,
    CheckpointManager,
    get_default_checkpoint_manager,
)

# ============================================================================
# CheckpointData Tests
# ============================================================================


class TestCheckpointData:
    """Tests for CheckpointData model."""

    def test_create_checkpoint_data(self) -> None:
        """Test creating checkpoint data."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test_123",
            task_type="evaluation",
            created_at=now,
            updated_at=now,
            total_items=100,
            completed_items=50,
            results=[{"score": 0.9}],
            config={"metric": "faithfulness"},
            metadata={"version": "1.0"},
        )

        assert data.checkpoint_id == "test_123"
        assert data.task_type == "evaluation"
        assert data.total_items == 100
        assert data.completed_items == 50
        assert len(data.results) == 1

    def test_is_complete_false(self) -> None:
        """Test is_complete when not finished."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test",
            task_type="test",
            created_at=now,
            updated_at=now,
            total_items=100,
            completed_items=50,
        )
        assert data.is_complete is False

    def test_is_complete_true(self) -> None:
        """Test is_complete when finished."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test",
            task_type="test",
            created_at=now,
            updated_at=now,
            total_items=100,
            completed_items=100,
        )
        assert data.is_complete is True

    def test_is_complete_over(self) -> None:
        """Test is_complete when over total."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test",
            task_type="test",
            created_at=now,
            updated_at=now,
            total_items=100,
            completed_items=110,
        )
        assert data.is_complete is True

    def test_progress_percent(self) -> None:
        """Test progress percentage calculation."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test",
            task_type="test",
            created_at=now,
            updated_at=now,
            total_items=100,
            completed_items=25,
        )
        assert data.progress_percent == 25.0

    def test_progress_percent_zero_total(self) -> None:
        """Test progress percentage with zero total."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test",
            task_type="test",
            created_at=now,
            updated_at=now,
            total_items=0,
            completed_items=0,
        )
        assert data.progress_percent == 100.0

    def test_default_values(self) -> None:
        """Test default values."""
        now = datetime.now(timezone.utc).isoformat()
        data = CheckpointData(
            checkpoint_id="test",
            task_type="test",
            created_at=now,
            updated_at=now,
            total_items=10,
        )
        assert data.completed_items == 0
        assert data.results == []
        assert data.config == {}
        assert data.metadata == {}


# ============================================================================
# CheckpointManager Tests
# ============================================================================


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_init_default_dir(self, tmp_path: Path) -> None:
        """Test initialization with default directory."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        assert manager.checkpoint_dir.exists()

    def test_init_creates_dir(self, tmp_path: Path) -> None:
        """Test that initialization creates directory."""
        checkpoint_dir = tmp_path / "new" / "nested" / "checkpoints"
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        assert checkpoint_dir.exists()
        assert manager.checkpoint_dir == checkpoint_dir


class TestCheckpointManagerCreate:
    """Tests for CheckpointManager.create method."""

    def test_create_checkpoint(self, tmp_path: Path) -> None:
        """Test creating a new checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(
            task_type="evaluation",
            total_items=100,
            config={"metric": "faithfulness"},
            metadata={"test": True},
        )

        assert checkpoint.task_type == "evaluation"
        assert checkpoint.total_items == 100
        assert checkpoint.completed_items == 0
        assert checkpoint.config == {"metric": "faithfulness"}
        assert "evaluation_" in checkpoint.checkpoint_id

    def test_create_checkpoint_saves_file(self, tmp_path: Path) -> None:
        """Test that create saves checkpoint to file."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        path = manager.get_checkpoint_path(checkpoint.checkpoint_id)
        assert path.exists()

        # Verify file contents
        data = json.loads(path.read_text())
        assert data["checkpoint_id"] == checkpoint.checkpoint_id


class TestCheckpointManagerSaveProgress:
    """Tests for CheckpointManager.save_progress method."""

    def test_save_progress(self, tmp_path: Path) -> None:
        """Test saving progress."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        result = {"score": 0.9, "query_id": "q1"}
        updated = manager.save_progress(checkpoint, result)

        assert updated.completed_items == 1
        assert len(updated.results) == 1
        assert updated.results[0] == result

    def test_save_progress_multiple(self, tmp_path: Path) -> None:
        """Test saving multiple progress updates."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        for i in range(5):
            checkpoint = manager.save_progress(checkpoint, {"index": i})

        assert checkpoint.completed_items == 5
        assert len(checkpoint.results) == 5

    def test_save_progress_updates_file(self, tmp_path: Path) -> None:
        """Test that save_progress updates the file."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        updated = manager.save_progress(checkpoint, {"score": 0.8})

        # Load from file and verify
        loaded = manager.load_by_id(updated.checkpoint_id)
        assert loaded.completed_items == 1


class TestCheckpointManagerSaveBatchProgress:
    """Tests for CheckpointManager.save_batch_progress method."""

    def test_save_batch_progress(self, tmp_path: Path) -> None:
        """Test saving batch progress."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        results = [{"index": i} for i in range(5)]
        updated = manager.save_batch_progress(checkpoint, results)

        assert updated.completed_items == 5
        assert len(updated.results) == 5

    def test_save_batch_progress_empty(self, tmp_path: Path) -> None:
        """Test saving empty batch."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        updated = manager.save_batch_progress(checkpoint, [])

        assert updated.completed_items == 0
        assert updated is checkpoint  # Returns same instance


class TestCheckpointManagerLoad:
    """Tests for CheckpointManager.load methods."""

    def test_load_by_path(self, tmp_path: Path) -> None:
        """Test loading checkpoint by path."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        original = manager.create(task_type="test", total_items=50)

        path = manager.get_checkpoint_path(original.checkpoint_id)
        loaded = manager.load(path)

        assert loaded.checkpoint_id == original.checkpoint_id
        assert loaded.total_items == 50

    def test_load_by_id(self, tmp_path: Path) -> None:
        """Test loading checkpoint by ID."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        original = manager.create(task_type="test", total_items=50)

        loaded = manager.load_by_id(original.checkpoint_id)

        assert loaded.checkpoint_id == original.checkpoint_id

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading non-existent file raises error."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.load(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON raises error."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("not valid json")

        with pytest.raises(ValueError, match="Invalid checkpoint"):
            manager.load(invalid_path)


class TestCheckpointManagerExists:
    """Tests for CheckpointManager.exists method."""

    def test_exists_true(self, tmp_path: Path) -> None:
        """Test exists returns True for existing checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        assert manager.exists(checkpoint.checkpoint_id) is True

    def test_exists_false(self, tmp_path: Path) -> None:
        """Test exists returns False for non-existing checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        assert manager.exists("nonexistent_id") is False


class TestCheckpointManagerCleanup:
    """Tests for CheckpointManager cleanup methods."""

    def test_cleanup(self, tmp_path: Path) -> None:
        """Test cleanup removes checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        path = manager.get_checkpoint_path(checkpoint.checkpoint_id)
        assert path.exists()

        manager.cleanup(checkpoint)
        assert not path.exists()

    def test_cleanup_by_id(self, tmp_path: Path) -> None:
        """Test cleanup_by_id removes checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        result = manager.cleanup_by_id(checkpoint.checkpoint_id)

        assert result is True
        assert not manager.exists(checkpoint.checkpoint_id)

    def test_cleanup_by_id_nonexistent(self, tmp_path: Path) -> None:
        """Test cleanup_by_id returns False for non-existent."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        result = manager.cleanup_by_id("nonexistent")

        assert result is False

    def test_cleanup_completed(self, tmp_path: Path) -> None:
        """Test cleanup_completed removes only completed checkpoints."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create incomplete checkpoint
        incomplete = manager.create(task_type="test", total_items=10)

        # Create complete checkpoint
        complete = manager.create(task_type="test", total_items=1)
        complete = manager.save_progress(complete, {"done": True})

        removed = manager.cleanup_completed()

        assert removed == 1
        assert manager.exists(incomplete.checkpoint_id)
        assert not manager.exists(complete.checkpoint_id)


class TestCheckpointManagerList:
    """Tests for CheckpointManager.list_checkpoints method."""

    def test_list_checkpoints(self, tmp_path: Path) -> None:
        """Test listing all checkpoints."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        manager.create(task_type="eval", total_items=10)
        manager.create(task_type="gen", total_items=20)
        manager.create(task_type="eval", total_items=30)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3

    def test_list_checkpoints_filtered(self, tmp_path: Path) -> None:
        """Test listing checkpoints filtered by type."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        manager.create(task_type="eval", total_items=10)
        manager.create(task_type="gen", total_items=20)
        manager.create(task_type="eval", total_items=30)

        eval_checkpoints = manager.list_checkpoints(task_type="eval")
        gen_checkpoints = manager.list_checkpoints(task_type="gen")

        assert len(eval_checkpoints) == 2
        assert len(gen_checkpoints) == 1

    def test_list_checkpoints_sorted_by_updated(self, tmp_path: Path) -> None:
        """Test that checkpoints are sorted by updated_at descending."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        cp1 = manager.create(task_type="test", total_items=10)
        time.sleep(0.01)  # Ensure different timestamps
        _cp2 = manager.create(task_type="test", total_items=10)
        time.sleep(0.01)
        cp3 = manager.create(task_type="test", total_items=10)

        checkpoints = manager.list_checkpoints()

        # Most recent first
        assert checkpoints[0].checkpoint_id == cp3.checkpoint_id
        assert checkpoints[2].checkpoint_id == cp1.checkpoint_id


class TestCheckpointManagerCleanupOlderThan:
    """Tests for CheckpointManager.cleanup_older_than method."""

    def test_cleanup_older_than(self, tmp_path: Path) -> None:
        """Test cleanup_older_than removes old checkpoints."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create checkpoint and manually set old date
        checkpoint = manager.create(task_type="test", total_items=10)
        path = manager.get_checkpoint_path(checkpoint.checkpoint_id)

        # Modify the checkpoint to have an old date
        data = json.loads(path.read_text())
        old_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        data["updated_at"] = old_date
        path.write_text(json.dumps(data))

        # Create a recent checkpoint
        recent = manager.create(task_type="test", total_items=10)

        removed = manager.cleanup_older_than(days=5)

        assert removed == 1
        assert not manager.exists(checkpoint.checkpoint_id)
        assert manager.exists(recent.checkpoint_id)


class TestCheckpointManagerAtomicWrite:
    """Tests for atomic write functionality."""

    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        """Test that atomic write creates the file."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        checkpoint = manager.create(task_type="test", total_items=10)

        path = manager.get_checkpoint_path(checkpoint.checkpoint_id)
        assert path.exists()

        # Verify content is valid JSON
        data = json.loads(path.read_text())
        assert data["checkpoint_id"] == checkpoint.checkpoint_id

    def test_atomic_write_no_partial_files(self, tmp_path: Path) -> None:
        """Test that no temporary files are left behind."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        for _ in range(10):
            checkpoint = manager.create(task_type="test", total_items=10)
            manager.save_progress(checkpoint, {"data": "test"})

        # Check for any .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestGetDefaultCheckpointManager:
    """Tests for get_default_checkpoint_manager function."""

    def test_returns_manager(self) -> None:
        """Test that function returns a CheckpointManager."""
        manager = get_default_checkpoint_manager()
        assert isinstance(manager, CheckpointManager)

    def test_uses_default_dir(self) -> None:
        """Test that default manager uses default directory."""
        manager = get_default_checkpoint_manager()
        assert manager.checkpoint_dir == Path(CheckpointManager.DEFAULT_CHECKPOINT_DIR)
