"""Evaluators for agent behavior and reasoning.

This module provides metrics for evaluating agent behavior,
including tool-use correctness and multi-step reasoning quality.

Example:
    >>> from ragnarok_ai.agents.evaluators import (
    ...     # Tool correctness
    ...     evaluate_tool_use,
    ...     ExpectedToolCall,
    ...     ToolUseMetrics,
    ...     # Reasoning evaluation
    ...     ReasoningCoherenceEvaluator,
    ...     GoalProgressEvaluator,
    ...     ReasoningEfficiencyEvaluator,
    ... )
    >>>
    >>> # Tool use evaluation (pure function)
    >>> metrics = evaluate_tool_use(response)
    >>> print(f"Success rate: {metrics.success_rate:.1%}")
    >>>
    >>> # Reasoning coherence (LLM-based)
    >>> coherence = ReasoningCoherenceEvaluator(llm)
    >>> score = await coherence.evaluate(response)
"""

from __future__ import annotations

from ragnarok_ai.agents.evaluators.reasoning import (
    CoherenceResult,
    EfficiencyResult,
    GoalProgressEvaluator,
    GoalProgressResult,
    ReasoningCoherenceEvaluator,
    ReasoningEfficiencyEvaluator,
    StepCoherence,
    StepProgress,
)
from ragnarok_ai.agents.evaluators.tool_correctness import (
    ExpectedToolCall,
    ToolCallEvaluation,
    ToolUseMetrics,
    arg_presence_rate,
    evaluate_tool_use,
    tool_error_rate,
    tool_success_rate,
)

__all__ = [
    "CoherenceResult",
    "EfficiencyResult",
    "ExpectedToolCall",
    "GoalProgressEvaluator",
    "GoalProgressResult",
    "ReasoningCoherenceEvaluator",
    "ReasoningEfficiencyEvaluator",
    "StepCoherence",
    "StepProgress",
    "ToolCallEvaluation",
    "ToolUseMetrics",
    "arg_presence_rate",
    "evaluate_tool_use",
    "tool_error_rate",
    "tool_success_rate",
]
