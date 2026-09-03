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
_PII_KEYS = frozenset(
    {
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
    }
)


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

    return value  # pragma: no cover


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
                sanitize_dict(item, mode, recursive=True) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value

    return result


# ── Free-text scrubbing ─────────────────────────────────────────────────────
#
# The patterns above match whole values (anchored) — suited to metadata
# fields. Free text (e.g. captured user queries) needs *inline* scrubbing:
# "email me at bob@corp.com" must keep the sentence and lose the address.
# Patterns are deliberately conservative to avoid mangling legitimate text.

_INLINE_PII_PATTERNS: list[re.Pattern[str]] = [
    # Email addresses
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    # IPv4 addresses
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # US SSN-like sequences
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Long digit runs (card/account numbers): 13-19 digits with optional separators
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # Unix home paths embedded in text
    re.compile(r"(?:^|(?<=\s))/(?:home|Users)/[^\s]+"),
]


def scrub_text(text: str, mode: PiiMode = PiiMode.REDACT) -> str:
    """Scrub inline PII from free text.

    Unlike ``sanitize_value``, which classifies whole values, this replaces
    PII *occurrences inside* the text (email addresses, IPs, SSN-like and
    card-like numbers, home paths) while keeping the rest intact.

    Args:
        text: The free text to scrub.
        mode: FULL passes through unchanged; HASH replaces each occurrence
            with a short hash; REDACT (default) with ``[REDACTED]``.

    Returns:
        The scrubbed text.

    Example:
        >>> scrub_text("email me at bob@corp.com about CHF", PiiMode.REDACT)
        'email me at [REDACTED] about CHF'
    """
    if mode == PiiMode.FULL:
        return text

    def _replace(match: re.Match[str]) -> str:
        if mode == PiiMode.HASH:
            return _hash_value(match.group())[:12]
        return "[REDACTED]"

    scrubbed = text
    for pattern in _INLINE_PII_PATTERNS:
        scrubbed = pattern.sub(_replace, scrubbed)
    return scrubbed
