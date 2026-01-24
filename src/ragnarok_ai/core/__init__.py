"""Core module for ragnarok-ai.

This module contains the fundamental types, protocols, exceptions,
and configuration used throughout the library.
"""

from __future__ import annotations

from ragnarok_ai.core.config import Settings
from ragnarok_ai.core.exceptions import (
    ConfigurationError,
    EvaluationError,
    LLMConnectionError,
    RagnarokError,
)
from ragnarok_ai.core.protocols import (
    EvaluatorProtocol,
    LLMProtocol,
    VectorStoreProtocol,
)
from ragnarok_ai.core.types import (
    Document,
    Query,
    RetrievalResult,
    TestSet,
)

__all__ = [
    "ConfigurationError",
    "Document",
    "EvaluationError",
    "EvaluatorProtocol",
    "LLMConnectionError",
    "LLMProtocol",
    "Query",
    "RagnarokError",
    "RetrievalResult",
    "Settings",
    "TestSet",
    "VectorStoreProtocol",
]
