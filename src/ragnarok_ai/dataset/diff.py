"""Dataset diff operations.

This module provides functions for comparing two TestSet versions
and identifying additions, removals, and modifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.hashing import compute_hash, sha256_short
from ragnarok_ai.dataset.models import DatasetDiffReport, FieldChange, ModifiedItem

if TYPE_CHECKING:
    from collections.abc import Callable

    from ragnarok_ai.core.types import Query, TestSet


def _default_key_fn(query: Query) -> str:
    """Generate a stable key for a query.

    Uses query.metadata["id"] if available, otherwise computes
    a deterministic hash from the query content.

    Args:
        query: Query to generate key for.

    Returns:
        Stable 16-character key.
    """
    # Check for explicit ID in metadata
    if query.metadata and "id" in query.metadata:
        return str(query.metadata["id"])

    # Generate deterministic key from content
    key_data = {
        "text": query.text,
        "ground_truth_docs": sorted(query.ground_truth_docs),
        "expected_answer": query.expected_answer,
    }
    return sha256_short(compute_hash(key_data), length=16)


def _canonicalize_query(query: Query, ignore_metadata: bool = True) -> dict[str, Any]:
    """Convert query to canonical dict for comparison.

    Sorts lists for stable comparison regardless of order.

    Args:
        query: Query to canonicalize.
        ignore_metadata: If True, exclude metadata from comparison.

    Returns:
        Canonical dict representation.
    """
    result: dict[str, Any] = {
        "text": query.text,
        "ground_truth_docs": sorted(query.ground_truth_docs),
    }

    if query.expected_answer is not None:
        result["expected_answer"] = query.expected_answer

    if not ignore_metadata and query.metadata:
        # Exclude volatile fields from metadata comparison
        volatile_fields = {"id", "created_at", "updated_at", "timestamp"}
        filtered_metadata = {k: v for k, v in query.metadata.items() if k not in volatile_fields}
        if filtered_metadata:
            result["metadata"] = filtered_metadata

    return result


def _find_changes(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[FieldChange]:
    """Find field-level changes between two canonical dicts.

    Args:
        before: Canonical dict from v1.
        after: Canonical dict from v2.

    Returns:
        List of field changes.
    """
    changes: list[FieldChange] = []
    all_keys = set(before.keys()) | set(after.keys())

    for key in sorted(all_keys):
        val_before = before.get(key)
        val_after = after.get(key)

        if val_before != val_after:
            changes.append(
                FieldChange(
                    field=key,
                    before=val_before,
                    after=val_after,
                )
            )

    return changes


def build_index(
    testset: TestSet,
    key_fn: Callable[[Query], str] | None = None,
) -> dict[str, Query]:
    """Build a key-to-query index for a testset.

    Args:
        testset: TestSet to index.
        key_fn: Function to generate key from query. Defaults to _default_key_fn.

    Returns:
        Dict mapping keys to queries.

    Raises:
        ValueError: If duplicate keys are detected.
    """
    key_fn = key_fn or _default_key_fn
    index: dict[str, Query] = {}
    duplicates: list[str] = []

    for query in testset.queries:
        key = key_fn(query)
        if key in index:
            duplicates.append(key)
        else:
            index[key] = query

    if duplicates:
        raise ValueError(
            f"Duplicate keys detected: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}. "
            "Ensure each query has a unique identifier."
        )

    return index


def diff_testsets(
    v1: TestSet,
    v2: TestSet,
    *,
    key_fn: Callable[[Query], str] | None = None,
    ignore_metadata: bool = True,
) -> DatasetDiffReport:
    """Compare two TestSet versions and produce a diff report.

    Args:
        v1: First (baseline) TestSet.
        v2: Second (current) TestSet.
        key_fn: Function to generate stable key for each query.
               Defaults to using metadata["id"] or content hash.
        ignore_metadata: If True, don't consider metadata changes as modifications.

    Returns:
        DatasetDiffReport with added, removed, modified, and unchanged items.

    Raises:
        ValueError: If duplicate keys are detected in either testset.
    """
    key_fn = key_fn or _default_key_fn

    # Build indexes
    index_v1 = build_index(v1, key_fn)
    index_v2 = build_index(v2, key_fn)

    keys_v1 = set(index_v1.keys())
    keys_v2 = set(index_v2.keys())

    # Find added and removed
    added = sorted(keys_v2 - keys_v1)
    removed = sorted(keys_v1 - keys_v2)

    # Find modified and unchanged
    modified: list[ModifiedItem] = []
    unchanged: list[str] = []

    common_keys = keys_v1 & keys_v2
    for key in sorted(common_keys):
        query_v1 = index_v1[key]
        query_v2 = index_v2[key]

        canonical_v1 = _canonicalize_query(query_v1, ignore_metadata)
        canonical_v2 = _canonicalize_query(query_v2, ignore_metadata)

        if canonical_v1 == canonical_v2:
            unchanged.append(key)
        else:
            changes = _find_changes(canonical_v1, canonical_v2)
            modified.append(
                ModifiedItem(
                    key=key,
                    changes=tuple(changes),
                )
            )

    return DatasetDiffReport(
        v1_hash=v1.hash_short,
        v2_hash=v2.hash_short,
        added=added,
        removed=removed,
        modified=modified,
        unchanged=unchanged,
    )
