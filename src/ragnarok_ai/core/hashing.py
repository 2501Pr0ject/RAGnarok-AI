"""Shared hashing utilities for ragnarok-ai.

This module provides canonical JSON serialization and SHA256 hashing
functions used across the codebase for fingerprinting and versioning.

Design goals:
- Deterministic: same input always produces same hash
- Canonical JSON: sorted keys, no extra whitespace
- Full SHA256 stored; short form only for display
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """
    Stable JSON serialization used for fingerprints.

    Args:
        data: JSON-serializable data (dict/list/str/int/float/bool/None).

    Returns:
        Canonical JSON string with sorted keys and no extra whitespace.

    Example:
        >>> canonical_json({"b": 1, "a": 2})
        '{"a":2,"b":1}'
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """
    Compute full SHA256 hex digest.

    Args:
        text: Input string to hash.

    Returns:
        64-character hexadecimal SHA256 digest.

    Example:
        >>> len(sha256_hex("hello"))
        64
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_short(text: str, length: int = 16) -> str:
    """
    Compute truncated SHA256 hex digest for display.

    Args:
        text: Input string to hash.
        length: Number of characters to return (default 16).

    Returns:
        Truncated hexadecimal SHA256 digest.

    Example:
        >>> len(sha256_short("hello"))
        16
    """
    return sha256_hex(text)[:length]


def compute_hash(data: Any) -> str:
    """
    Compute full SHA256 hash of JSON-serializable data.

    Args:
        data: JSON-serializable data structure.

    Returns:
        64-character hexadecimal SHA256 digest.

    Example:
        >>> compute_hash({"key": "value"})  # doctest: +ELLIPSIS
        '...'
    """
    return sha256_hex(canonical_json(data))


def compute_hash_short(data: Any, length: int = 16) -> str:
    """
    Compute truncated SHA256 hash of JSON-serializable data.

    Args:
        data: JSON-serializable data structure.
        length: Number of characters to return (default 16).

    Returns:
        Truncated hexadecimal SHA256 digest.

    Example:
        >>> len(compute_hash_short({"key": "value"}))
        16
    """
    return compute_hash(data)[:length]
