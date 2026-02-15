"""Data models for dataset diff operations.

This module defines the schema for dataset comparison results,
including added, removed, and modified items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldChange:
    """A single field change within a modified item."""

    field: str
    before: Any
    after: Any


@dataclass(frozen=True)
class ModifiedItem:
    """A modified item in the dataset diff.

    Attributes:
        key: Stable identifier for the item.
        changes: List of field changes.
    """

    key: str
    changes: tuple[FieldChange, ...]

    @property
    def fields_changed(self) -> list[str]:
        """List of field names that changed."""
        return [c.field for c in self.changes]


@dataclass
class DatasetDiffReport:
    """Result of comparing two datasets.

    Attributes:
        v1_hash: Hash of the first dataset.
        v2_hash: Hash of the second dataset.
        added: Keys of items added in v2.
        removed: Keys of items removed from v1.
        modified: Items that changed between v1 and v2.
        unchanged: Keys of items that are identical.
    """

    v1_hash: str
    v2_hash: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[ModifiedItem] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if there are any differences."""
        return bool(self.added or self.removed or self.modified)

    @property
    def total_items_v1(self) -> int:
        """Total items in v1."""
        return len(self.removed) + len(self.modified) + len(self.unchanged)

    @property
    def total_items_v2(self) -> int:
        """Total items in v2."""
        return len(self.added) + len(self.modified) + len(self.unchanged)

    def summary(self) -> dict[str, int]:
        """Summary statistics."""
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "unchanged": len(self.unchanged),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export to JSON-serializable dict."""
        return {
            "v1_hash": self.v1_hash,
            "v2_hash": self.v2_hash,
            "summary": self.summary(),
            "added": self.added,
            "removed": self.removed,
            "modified": [
                {
                    "key": m.key,
                    "fields_changed": m.fields_changed,
                    "changes": [{"field": c.field, "before": c.before, "after": c.after} for c in m.changes],
                }
                for m in self.modified
            ],
            "unchanged_count": len(self.unchanged),
        }
