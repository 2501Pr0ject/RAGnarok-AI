"""Privacy utilities for PII handling.

This module provides functions to sanitize data that may contain
Personally Identifiable Information (PII) like file paths, usernames,
email addresses, etc.

PII Modes:
    - full: No filtering, data passes through unchanged (API default for compatibility)
    - hash: SHA256 hash values that look like PII (CLI default for safety)
    - redact: Replace PII with [REDACTED]

Usage:
    from ragnarok_ai.privacy import PiiMode, sanitize_dict

    # Hash PII values in a dictionary
    clean = sanitize_dict(data, mode=PiiMode.HASH)

    # Redact PII values
    clean = sanitize_dict(data, mode=PiiMode.REDACT)
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any


class PiiMode(str, Enum):
    """PII handling mode.

    Attributes:
        FULL: No filtering, pass through unchanged.
        HASH: SHA256 hash values that look like PII.
        REDACT: Replace PII with [REDACTED].
    """

    FULL = "full"
    HASH = "hash"
    REDACT = "redact"


# Patterns that indicate potential PII
_PII_PATTERNS = [
    # File paths (Unix and Windows)
    re.compile(r"^(/[^/\s]+)+/?$"),  # Unix path
    re.compile(r"^[A-Za-z]:\\.*$"),  # Windows path
    re.compile(r"^~[/\\].*$"),  # Home directory
    # Email addresses
    re.compile(r"^[\w.+-]+@[\w.-]+\.\w+$"),
    # IP addresses
    re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
    # Usernames in common formats
    re.compile(r"^user[_-]?\d+$", re.IGNORECASE),
    re.compile(r"^[a-z]{2,}\d{2,}$", re.IGNORECASE),  # alice42, bob123
]

# Keys that typically contain PII
_PII_KEYS = frozenset({
    "source",
    "source_uri",
    "source_path",
    "file_path",
    "path",
    "filename",
    "file",
    "user",
    "username",
    "user_id",
    "author",
    "email",
    "ip",
    "ip_address",
    "host",
    "hostname",
})


def _looks_like_pii(value: str) -> bool:
    """Check if a string value looks like PII."""
    if not value or len(value) < 3:
        return False

    return any(pattern.match(value) for pattern in _PII_PATTERNS)


def _hash_value(value: str) -> str:
    """Hash a value using SHA256."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_value(value: str, mode: PiiMode, key: str | None = None) -> str:
    """Sanitize a single string value based on PII mode.

    Args:
        value: The string value to sanitize.
        mode: The PII handling mode.
        key: Optional key name (used to check if key is PII-sensitive).

    Returns:
        Sanitized value based on mode.
    """
    if mode == PiiMode.FULL:
        return value

    # Check if value looks like PII or if key is a PII key
    is_pii = _looks_like_pii(value)
    if key and key.lower() in _PII_KEYS:
        is_pii = True

    if not is_pii:
        return value

    if mode == PiiMode.HASH:
        return _hash_value(value)
    elif mode == PiiMode.REDACT:
        return "[REDACTED]"

    return value


def sanitize_dict(
    data: dict[str, Any],
    mode: PiiMode = PiiMode.FULL,
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    """Sanitize a dictionary, filtering PII values.

    Args:
        data: Dictionary to sanitize.
        mode: PII handling mode.
        recursive: Whether to recursively sanitize nested dicts.

    Returns:
        New dictionary with sanitized values.

    Example:
        >>> data = {"source": "/home/alice/docs/file.txt", "text": "Hello"}
        >>> sanitize_dict(data, PiiMode.HASH)
        {"source": "abc123...", "text": "Hello"}
    """
    if mode == PiiMode.FULL:
        return data

    result: dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_value(value, mode, key)
        elif isinstance(value, dict) and recursive:
            result[key] = sanitize_dict(value, mode, recursive=True)
        elif isinstance(value, list) and recursive:
            result[key] = [
                sanitize_dict(item, mode, recursive=True) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result
