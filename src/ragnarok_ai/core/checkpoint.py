"""Checkpointing system for ragnarok-ai.

This module provides checkpointing capabilities to resume interrupted
generation or evaluation runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CheckpointData(BaseModel):
    """Data stored in a checkpoint file.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        task_type: Type of task (generation, evaluation).
        created_at: When the checkpoint was created.
        updated_at: When the checkpoint was last updated.
        total_items: Total number of items to process.
        completed_items: Number of items completed.
        results: Results collected so far.
        config: Configuration used for the task.
        metadata: Additional metadata.
    """

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    task_type: str = Field(..., description="Type of task")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    total_items: int = Field(..., description="Total items to process")
    completed_items: int = Field(default=0, description="Items completed")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Results so far")
    config: dict[str, Any] = Field(default_factory=dict, description="Task configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def is_complete(self) -> bool:
        """Check if the task is complete."""
        return self.completed_items >= self.total_items

    @property
    def progress_percent(self) -> float:
        """Get progress as a percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100


class CheckpointManager:
    """Manages checkpoints for long-running tasks.

    Provides atomic saves, resume capabilities, and auto-cleanup.

    Attributes:
        checkpoint_dir: Directory where checkpoints are stored.

    Example:
        >>> manager = CheckpointManager()
        >>> checkpoint = manager.create("evaluation", total_items=100)
        >>> for i, result in enumerate(results):
        ...     checkpoint = manager.save_progress(checkpoint, result)
        >>> manager.cleanup(checkpoint)  # Remove completed checkpoint
    """

    DEFAULT_CHECKPOINT_DIR = ".ragnarok/checkpoints"

    def __init__(self, checkpoint_dir: Path | str | None = None) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
                           Defaults to .ragnarok/checkpoints/
        """
        self.checkpoint_dir = Path(checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        task_type: str,
        total_items: int,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointData:
        """Create a new checkpoint.

        Args:
            task_type: Type of task (e.g., "generation", "evaluation").
            total_items: Total number of items to process.
            config: Configuration for the task.
            metadata: Additional metadata.

        Returns:
            New CheckpointData instance.
        """
        now = datetime.now(timezone.utc).isoformat()
        checkpoint_id = f"{task_type}_{uuid.uuid4().hex[:8]}"

        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            task_type=task_type,
            created_at=now,
            updated_at=now,
            total_items=total_items,
            completed_items=0,
            results=[],
            config=config or {},
            metadata=metadata or {},
        )

        # Save initial checkpoint
        self._save_atomic(checkpoint)
        return checkpoint

    def save_progress(
        self,
        checkpoint: CheckpointData,
        result: dict[str, Any],
    ) -> CheckpointData:
        """Save progress with a new result.

        Args:
            checkpoint: Current checkpoint data.
            result: New result to add.

        Returns:
            Updated CheckpointData instance.
        """
        # Create updated checkpoint (immutable update)
        updated = CheckpointData(
            checkpoint_id=checkpoint.checkpoint_id,
            task_type=checkpoint.task_type,
            created_at=checkpoint.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            total_items=checkpoint.total_items,
            completed_items=checkpoint.completed_items + 1,
            results=[*checkpoint.results, result],
            config=checkpoint.config,
            metadata=checkpoint.metadata,
        )

        # Save atomically
        self._save_atomic(updated)
        return updated

    def save_batch_progress(
        self,
        checkpoint: CheckpointData,
        results: list[dict[str, Any]],
    ) -> CheckpointData:
        """Save progress with multiple results.

        Args:
            checkpoint: Current checkpoint data.
            results: New results to add.

        Returns:
            Updated CheckpointData instance.
        """
        if not results:
            return checkpoint

        updated = CheckpointData(
            checkpoint_id=checkpoint.checkpoint_id,
            task_type=checkpoint.task_type,
            created_at=checkpoint.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            total_items=checkpoint.total_items,
            completed_items=checkpoint.completed_items + len(results),
            results=[*checkpoint.results, *results],
            config=checkpoint.config,
            metadata=checkpoint.metadata,
        )

        self._save_atomic(updated)
        return updated

    def load(self, checkpoint_path: Path | str) -> CheckpointData:
        """Load a checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Loaded CheckpointData.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint file is invalid.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            msg = f"Checkpoint file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            data = json.loads(path.read_text())
            return CheckpointData(**data)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid checkpoint file: {e}"
            raise ValueError(msg) from e

    def load_by_id(self, checkpoint_id: str) -> CheckpointData:
        """Load a checkpoint by its ID.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            Loaded CheckpointData.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        path = self._get_checkpoint_path(checkpoint_id)
        return self.load(path)

    def exists(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint exists.
        """
        return self._get_checkpoint_path(checkpoint_id).exists()

    def cleanup(self, checkpoint: CheckpointData) -> None:
        """Remove a completed checkpoint.

        Args:
            checkpoint: The checkpoint to remove.
        """
        path = self._get_checkpoint_path(checkpoint.checkpoint_id)
        if path.exists():
            path.unlink()

    def cleanup_by_id(self, checkpoint_id: str) -> bool:
        """Remove a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was removed, False if it didn't exist.
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_checkpoints(self, task_type: str | None = None) -> list[CheckpointData]:
        """List all checkpoints, optionally filtered by task type.

        Args:
            task_type: Optional filter by task type.

        Returns:
            List of checkpoint data.
        """
        checkpoints: list[CheckpointData] = []

        for path in self.checkpoint_dir.glob("*.json"):
            try:
                checkpoint = self.load(path)
                if task_type is None or checkpoint.task_type == task_type:
                    checkpoints.append(checkpoint)
            except (ValueError, FileNotFoundError):
                continue

        return sorted(checkpoints, key=lambda c: c.updated_at, reverse=True)

    def cleanup_completed(self) -> int:
        """Remove all completed checkpoints.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        for checkpoint in self.list_checkpoints():
            if checkpoint.is_complete:
                self.cleanup(checkpoint)
                removed += 1
        return removed

    def cleanup_older_than(self, days: int) -> int:
        """Remove checkpoints older than specified days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        cutoff = datetime.now(timezone.utc)

        for checkpoint in self.list_checkpoints():
            updated = datetime.fromisoformat(checkpoint.updated_at)
            age = (cutoff - updated).days
            if age > days:
                self.cleanup(checkpoint)
                removed += 1

        return removed

    def get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            Path to the checkpoint file.
        """
        return self._get_checkpoint_path(checkpoint_id)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint ID.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            Path to the checkpoint file.
        """
        return self.checkpoint_dir / f"{checkpoint_id}.json"

    def _save_atomic(self, checkpoint: CheckpointData) -> None:
        """Save checkpoint atomically using temp file + rename.

        Args:
            checkpoint: The checkpoint data to save.
        """
        target_path = self._get_checkpoint_path(checkpoint.checkpoint_id)

        # Write to temp file in same directory (for atomic rename)
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="checkpoint_",
            dir=self.checkpoint_dir,
        )

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(checkpoint.model_dump(), f, indent=2)

            # Atomic rename
            Path(temp_path).replace(target_path)
        except Exception:
            # Clean up temp file on error
            temp_path_obj = Path(temp_path)
            if temp_path_obj.exists():
                temp_path_obj.unlink()
            raise


def get_default_checkpoint_manager() -> CheckpointManager:
    """Get a checkpoint manager with default settings.

    Returns:
        CheckpointManager instance.
    """
    return CheckpointManager()
