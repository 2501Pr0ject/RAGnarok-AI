"""ragnarok-ai: A local-first RAG evaluation framework for LLM applications."""

from __future__ import annotations

from ragnarok_ai.core.evaluate import EvaluationResult, evaluate, evaluate_stream

__version__ = "0.3.0"
__all__ = ["EvaluationResult", "__version__", "evaluate", "evaluate_stream"]
