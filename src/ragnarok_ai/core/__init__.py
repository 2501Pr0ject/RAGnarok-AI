"""Core module for ragnarok-ai.

This module contains the fundamental types, protocols, exceptions,
and configuration used throughout the library.
"""

from __future__ import annotations

from ragnarok_ai.core.artifacts import (
    Environment,
    Experiment,
    ExperimentVariant,
    Fingerprint,
    GeneratorConfig,
    JudgeConfig,
    PipelineConfig,
    RetrieverConfig,
    Run,
    Trace,
)
from ragnarok_ai.core.checkpoint import (
    CheckpointData,
    CheckpointManager,
    get_default_checkpoint_manager,
)
from ragnarok_ai.core.compare import (
    ComparisonProgress,
    ComparisonResult,
    ConfigResult,
    compare,
)
from ragnarok_ai.core.config import Settings
from ragnarok_ai.core.exceptions import (
    CheckpointFormatError,
    ConfigurationError,
    EvaluationError,
    JudgeConnectionError,
    JudgeDependencyError,
    JudgeParseError,
    LLMConnectionError,
    RagnarokError,
)
from ragnarok_ai.core.gating import (
    GatingConfig,
    GatingEvaluationResult,
    GatingEvaluator,
    GatingResult,
    MetricCategory,
    MetricResult,
)
from ragnarok_ai.core.protocols import (
    EvaluatorProtocol,
    JudgeProtocol,
    LLMProtocol,
    LLMResponse,
    VectorStoreProtocol,
)
from ragnarok_ai.core.types import (
    Document,
    Query,
    RetrievalResult,
    TestSet,
)

__all__ = [
    "CheckpointData",
    "CheckpointFormatError",
    "CheckpointManager",
    "ComparisonProgress",
    "ComparisonResult",
    "ConfigResult",
    "ConfigurationError",
    "Document",
    "Environment",
    "EvaluationError",
    "EvaluatorProtocol",
    "Experiment",
    "ExperimentVariant",
    "Fingerprint",
    "GatingConfig",
    "GatingEvaluationResult",
    "GatingEvaluator",
    "GatingResult",
    "GeneratorConfig",
    "JudgeConfig",
    "JudgeConnectionError",
    "JudgeDependencyError",
    "JudgeParseError",
    "JudgeProtocol",
    "LLMConnectionError",
    "LLMProtocol",
    "LLMResponse",
    "MetricCategory",
    "MetricResult",
    "PipelineConfig",
    "Query",
    "RagnarokError",
    "RetrievalResult",
    "RetrieverConfig",
    "Run",
    "Settings",
    "TestSet",
    "Trace",
    "VectorStoreProtocol",
    "compare",
    "get_default_checkpoint_manager",
]
