"""ragnarok-ai: A local-first RAG evaluation framework for LLM applications."""

from __future__ import annotations

from ragnarok_ai.core.batch import BatchConfig, BatchEvaluator, BatchProgress, BatchResult
from ragnarok_ai.core.compare import ComparisonResult, compare
from ragnarok_ai.core.evaluate import EvaluationResult, evaluate, evaluate_stream

__version__ = "1.1.0"
__all__ = [
    "BatchConfig",
    "BatchEvaluator",
    "BatchProgress",
    "BatchResult",
    "ComparisonResult",
    "EvaluationResult",
    "__version__",
    "compare",
    "evaluate",
    "evaluate_stream",
]
