"""Custom exceptions for ragnarok-ai.

This module defines the exception hierarchy used throughout the library.
All exceptions inherit from RagnarokError for easy catching.
"""

from __future__ import annotations


class RagnarokError(Exception):
    """Base exception for all ragnarok-ai errors.

    All custom exceptions in ragnarok-ai inherit from this class,
    making it easy to catch all library-specific errors.

    Example:
        >>> try:
        ...     # ragnarok-ai operations
        ...     pass
        ... except RagnarokError as e:
        ...     print(f"ragnarok-ai error: {e}")
    """


class LLMConnectionError(RagnarokError):
    """Raised when connection to an LLM provider fails.

    This exception is raised when the library cannot connect to
    the configured LLM provider (e.g., Ollama, vLLM).

    Example:
        >>> raise LLMConnectionError("Failed to connect to Ollama at localhost:11434")
    """


class EvaluationError(RagnarokError):
    """Raised when an evaluation operation fails.

    This exception is raised when metric calculation fails,
    such as when receiving invalid scores or malformed data.

    Example:
        >>> raise EvaluationError("Invalid faithfulness score: -0.5 (must be 0-1)")
    """


class ConfigurationError(RagnarokError):
    """Raised when configuration is invalid or missing.

    This exception is raised when required configuration is missing
    or when provided values are invalid.

    Example:
        >>> raise ConfigurationError("Missing required env var: RAGNAROK_OLLAMA_URL")
    """
