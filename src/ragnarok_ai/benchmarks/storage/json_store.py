"""JSON file storage for benchmarks.

This module provides a simple JSON file-based storage backend
for benchmark records.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ragnarok_ai.benchmarks.models import BenchmarkRecord

logger = logging.getLogger(__name__)


class JSONFileStore:
    """JSON file storage for benchmarks.

    Uses atomic writes (temp file + rename) for safety.
    Supports max_records with FIFO rotation (preserving baselines).

    Example:
        >>> store = JSONFileStore(".ragnarok/benchmarks.json")
        >>> await store.save(record)
        >>> records = await store.list(config_name="my-config")
    """

    def __init__(
        self,
        path: str | Path = ".ragnarok/benchmarks.json",
        max_records: int | None = None,
    ) -> None:
        """Initialize the JSON file store.

        Args:
            path: Path to the JSON file.
            max_records: Maximum records to keep (None = unlimited).
        """
        self._path = Path(path)
        self._max_records = max_records

    async def _load(self) -> list[dict[str, Any]]:
        """Load records from JSON file.

        Returns:
            List of record dictionaries.
        """
        if not self._path.exists():
            return []

        try:
            content = self._path.read_text()
            if not content.strip():
                return []
            data: dict[str, Any] = json.loads(content)
            records: list[dict[str, Any]] = data.get("records", [])
            return records
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load benchmarks from {self._path}: {e}")
            return []

    async def _save(self, records: list[dict[str, Any]]) -> None:
        """Save records to JSON file with atomic write.

        Uses temp file + rename for atomic operation.

        Args:
            records: List of record dictionaries.
        """
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data = {"records": records}
        content = json.dumps(data, indent=2, default=str)

        # Atomic write: write to temp file, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=".benchmarks_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                f.write(content)
            # Atomic rename
            Path(temp_path).replace(self._path)
        except Exception:
            # Clean up temp file on failure
            Path(temp_path).unlink(missing_ok=True)
            raise

    def _rotate_if_needed(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove oldest non-baseline records if over max_records.

        Args:
            records: List of record dictionaries.

        Returns:
            Rotated list of records.
        """
        if self._max_records is None or len(records) <= self._max_records:
            return records

        # Separate baselines from non-baselines
        baselines = [r for r in records if r.get("is_baseline", False)]
        non_baselines = [r for r in records if not r.get("is_baseline", False)]

        # Sort non-baselines by timestamp (oldest first)
        non_baselines.sort(key=lambda r: r.get("timestamp", ""))

        # Calculate how many non-baselines to keep
        keep_count = max(0, self._max_records - len(baselines))
        kept_non_baselines = non_baselines[-keep_count:] if keep_count > 0 else []

        # Combine and re-sort by timestamp descending
        result = baselines + kept_non_baselines
        result.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        return result

    async def save(self, record: BenchmarkRecord) -> None:
        """Save a benchmark record.

        Args:
            record: The benchmark record to save.
        """
        records = await self._load()
        records.append(record.to_dict())

        # Rotate if needed
        records = self._rotate_if_needed(records)

        await self._save(records)

    async def get(self, record_id: str) -> BenchmarkRecord | None:
        """Get a record by ID.

        Args:
            record_id: The unique record identifier.

        Returns:
            The record if found, None otherwise.
        """
        records = await self._load()
        for data in records:
            if data.get("id") == record_id:
                return BenchmarkRecord.from_dict(data)
        return None

    async def list(
        self,
        config_name: str | None = None,
        limit: int = 100,
    ) -> list[BenchmarkRecord]:
        """List records, optionally filtered by config.

        Args:
            config_name: Filter by config name (None for all).
            limit: Maximum number of records to return.

        Returns:
            List of records, sorted by timestamp descending.
        """
        records = await self._load()

        # Filter by config if specified
        if config_name is not None:
            records = [r for r in records if r.get("config_name") == config_name]

        # Sort by timestamp descending (most recent first)
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        # Apply limit
        records = records[:limit]

        return [BenchmarkRecord.from_dict(r) for r in records]

    async def delete(self, record_id: str) -> bool:
        """Delete a record.

        Args:
            record_id: The unique record identifier.

        Returns:
            True if deleted, False if not found.
        """
        records = await self._load()
        initial_count = len(records)

        records = [r for r in records if r.get("id") != record_id]

        if len(records) == initial_count:
            return False

        await self._save(records)
        return True

    async def set_baseline(self, record_id: str) -> None:
        """Mark a record as baseline for its config.

        This will unset any existing baseline for the same config.

        Args:
            record_id: The record to mark as baseline.

        Raises:
            ValueError: If record not found.
        """
        records = await self._load()

        # Find the record
        target_record = None
        target_config = None
        for r in records:
            if r.get("id") == record_id:
                target_record = r
                target_config = r.get("config_name")
                break

        if target_record is None:
            raise ValueError(f"Record not found: {record_id}")

        # Unset any existing baseline for this config
        for r in records:
            if r.get("config_name") == target_config and r.get("is_baseline", False):
                r["is_baseline"] = False

        # Set the new baseline
        target_record["is_baseline"] = True

        await self._save(records)

    async def get_baseline(self, config_name: str) -> BenchmarkRecord | None:
        """Get the baseline record for a config.

        Args:
            config_name: The configuration name.

        Returns:
            The baseline record if set, None otherwise.
        """
        records = await self._load()
        for data in records:
            if data.get("config_name") == config_name and data.get("is_baseline", False):
                return BenchmarkRecord.from_dict(data)
        return None
