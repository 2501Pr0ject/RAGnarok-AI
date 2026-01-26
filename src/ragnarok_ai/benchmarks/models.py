"""Models for benchmark tracking.

This module provides the BenchmarkRecord dataclass for storing
benchmark results in history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkRecord:
    """A single benchmark run stored in history.

    Attributes:
        id: Unique identifier (UUID).
        timestamp: When the benchmark was recorded.
        config_name: Name of the configuration.
        testset_hash: Hash of testset for validation.
        testset_name: Human-readable testset name.
        metrics: Aggregated metrics (precision, recall, mrr, ndcg, latency_ms).
        metadata: Custom metadata (version, git commit, etc.).
        is_baseline: Whether this record is marked as baseline.

    Example:
        >>> record = BenchmarkRecord(
        ...     id="abc123",
        ...     timestamp=datetime.now(),
        ...     config_name="my-rag",
        ...     testset_hash="deadbeef",
        ...     testset_name="test_set",
        ...     metrics={"precision": 0.85, "recall": 0.78},
        ... )
    """

    id: str
    timestamp: datetime
    config_name: str
    testset_hash: str
    testset_name: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for serialization.

        Returns:
            Dictionary representation of the record.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "config_name": self.config_name,
            "testset_hash": self.testset_hash,
            "testset_name": self.testset_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "is_baseline": self.is_baseline,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRecord:
        """Create record from dictionary.

        Args:
            data: Dictionary with record fields.

        Returns:
            BenchmarkRecord instance.
        """
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config_name=data["config_name"],
            testset_hash=data["testset_hash"],
            testset_name=data["testset_name"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
            is_baseline=data.get("is_baseline", False),
        )
