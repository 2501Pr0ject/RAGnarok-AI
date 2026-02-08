"""ragnarok-ai: A local-first RAG evaluation framework for LLM applications."""

from __future__ import annotations

from ragnarok_ai.core.batch import BatchConfig, BatchEvaluator, BatchProgress, BatchResult
from ragnarok_ai.core.compare import ComparisonResult, compare
from ragnarok_ai.core.evaluate import EvaluationResult, evaluate, evaluate_stream
from ragnarok_ai.evaluators.judge import JudgeResult, JudgeResults, LLMJudge
from ragnarok_ai.loaders.forge_bundle import (
    ForgeLoadError,
    load_forge_bundle,
    load_forge_documents,
)
from ragnarok_ai.privacy import PiiMode, sanitize_dict, sanitize_value

__version__ = "1.2.0"
__all__ = [
    # Batch evaluation
    "BatchConfig",
    "BatchEvaluator",
    "BatchProgress",
    "BatchResult",
    # Comparison
    "ComparisonResult",
    "compare",
    # Evaluation
    "EvaluationResult",
    "evaluate",
    "evaluate_stream",
    # Forge loader
    "ForgeLoadError",
    "load_forge_bundle",
    "load_forge_documents",
    # LLM-as-Judge (Prometheus 2)
    "JudgeResult",
    "JudgeResults",
    "LLMJudge",
    # Privacy
    "PiiMode",
    "sanitize_dict",
    "sanitize_value",
    # Version
    "__version__",
]
