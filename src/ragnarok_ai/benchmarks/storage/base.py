"""Base protocol for benchmark storage backends.

This module defines the StorageProtocol that all storage backends must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragnarok_ai.benchmarks.models import BenchmarkRecord


@runtime_checkable
class StorageProtocol(Protocol):
    """Protocol for benchmark storage backends.

    All storage backends must implement these async methods.

    Example:
        >>> class MyStorage:
        ...     async def save(self, record: BenchmarkRecord) -> None: ...
        ...     # ... implement other methods
        >>> isinstance(MyStorage(), StorageProtocol)
        True
    """

    async def save(self, record: BenchmarkRecord) -> None:  # pragma: no cover
        """Save a benchmark record.

        Args:
            record: The benchmark record to save.
        """
        ...

    async def get(self, record_id: str) -> BenchmarkRecord | None:  # pragma: no cover
        """Get a record by ID.

        Args:
            record_id: The unique record identifier.

        Returns:
            The record if found, None otherwise.
        """
        ...

    async def list(  # pragma: no cover
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
        ...

    async def delete(self, record_id: str) -> bool:  # pragma: no cover
        """Delete a record.

        Args:
            record_id: The unique record identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def set_baseline(self, record_id: str) -> None:  # pragma: no cover
        """Mark a record as baseline for its config.

        This will unset any existing baseline for the same config.

        Args:
            record_id: The record to mark as baseline.

        Raises:
            ValueError: If record not found.
        """
        ...

    async def get_baseline(self, config_name: str) -> BenchmarkRecord | None:  # pragma: no cover
        """Get the baseline record for a config.

        Args:
            config_name: The configuration name.

        Returns:
            The baseline record if set, None otherwise.
        """
        ...
