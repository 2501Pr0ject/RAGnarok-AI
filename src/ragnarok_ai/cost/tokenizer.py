"""Token counting utilities.

This module provides token counting functionality with tiktoken support
and a fallback estimation for when tiktoken is not installed.
"""

from __future__ import annotations

import functools
from typing import Any

# Try to import tiktoken, but don't fail if not available
_tiktoken_available = False
_tiktoken_module: Any = None
try:
    import tiktoken

    _tiktoken_module = tiktoken
    _tiktoken_available = True
except ImportError:
    pass


def is_tiktoken_available() -> bool:
    """Check if tiktoken is installed and available."""
    return _tiktoken_available


@functools.lru_cache(maxsize=8)
def _get_encoding(model: str) -> Any:
    """Get tiktoken encoding for a model (cached).

    Args:
        model: Model name.

    Returns:
        tiktoken Encoding or None if not available.
    """
    if not _tiktoken_available or _tiktoken_module is None:
        return None

    try:
        # Try to get encoding for the specific model
        return _tiktoken_module.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base (GPT-4, ChatGPT models)
        try:
            return _tiktoken_module.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text.

    Uses tiktoken if available, otherwise falls back to estimation.

    Args:
        text: Text to count tokens for.
        model: Model name for accurate counting (default: "gpt-4").

    Returns:
        Number of tokens (exact with tiktoken, estimated otherwise).
    """
    if not text:
        return 0

    # Try tiktoken first
    encoding = _get_encoding(model)
    if encoding is not None:
        return len(encoding.encode(text))

    # Fallback: estimate ~4 characters per token (reasonable for English)
    # This is a common approximation used when tiktoken isn't available
    return len(text) // 4


def estimate_tokens(text: str) -> int:
    """Estimate token count without tiktoken.

    Uses the approximation of ~4 characters per token.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0
    return len(text) // 4
