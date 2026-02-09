"""ragnarok-ai: A local-first RAG evaluation framework for LLM applications."""

from __future__ import annotations

from ragnarok_ai.core.batch import BatchConfig, BatchEvaluator, BatchProgress, BatchResult
from ragnarok_ai.core.compare import ComparisonResult, compare
from ragnarok_ai.core.evaluate import EvaluationResult, evaluate, evaluate_stream
from ragnarok_ai.cost.tracker import CostSummary, CostTracker
from ragnarok_ai.evaluators.judge import JudgeResult, JudgeResults, LLMJudge
from ragnarok_ai.loaders.forge_bundle import (
    ForgeLoadError,
    load_forge_bundle,
    load_forge_documents,
)
from ragnarok_ai.privacy import PiiMode, sanitize_dict, sanitize_value

__version__ = "1.3.1"
__all__ = [
    "BatchConfig",
    "BatchEvaluator",
    "BatchProgress",
    "BatchResult",
    "ComparisonResult",
    "CostSummary",
    "CostTracker",
    "EvaluationResult",
    "ForgeLoadError",
    "JudgeResult",
    "JudgeResults",
    "LLMJudge",
    "PiiMode",
    "__version__",
    "compare",
    "evaluate",
    "evaluate_stream",
    "load_forge_bundle",
    "load_forge_documents",
    "sanitize_dict",
    "sanitize_value",
]
