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


class VectorStoreConnectionError(RagnarokError):
    """Raised when connection to a vector store fails.

    This exception is raised when the library cannot connect to
    the configured vector store (e.g., Qdrant, ChromaDB).

    Example:
        >>> raise VectorStoreConnectionError("Failed to connect to Qdrant at localhost:6333")
    """


class CheckpointFormatError(RagnarokError):
    """Raised when a checkpoint cannot be resumed because of its format.

    Batch-evaluation checkpoints are versioned (``CheckpointData.format_version``).
    A checkpoint written by an earlier version of ragnarok-ai, which stored
    results by batch number instead of by query, cannot be resumed.

    Example:
        >>> raise CheckpointFormatError("Checkpoint batch_evaluation_1a2b3c4d uses format 1; version 2 is required")
    """
class JudgeModelNotFoundError(ConfigurationError, RuntimeError):
    """Raised when the judge model is not installed on the Ollama server.

    Distinct from ``LLMConnectionError`` (the server could not be reached):
    the server answered, but does not serve the requested model. The message
    carries the command to install it. Also a ``RuntimeError`` so code
    written for earlier versions, which raised one, keeps catching it.

    Attributes:
        model: The requested model name.
        available: Model names the server reported.
    """

    def __init__(self, model: str, available: list[str], base_url: str) -> None:
        """Initialize JudgeModelNotFoundError."""
        self.model = model
        self.available = list(available)
        listed = ", ".join(self.available) if self.available else "none"
        msg = (
            f"Judge model '{model}' is not installed on the Ollama server at {base_url}. "
            f"Install it with: ollama pull {model} (installed models: {listed})"
        )
        super().__init__(msg)
