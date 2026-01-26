"""Evaluators for agent tool usage.

This module provides metrics for evaluating agent behavior,
including tool-use correctness.

Example:
    >>> from ragnarok_ai.agents.evaluators import (
    ...     evaluate_tool_use,
    ...     ExpectedToolCall,
    ...     ToolCallEvaluation,
    ...     ToolUseMetrics,
    ... )
    >>>
    >>> metrics = evaluate_tool_use(response)
    >>> print(f"Success rate: {metrics.success_rate:.1%}")
"""

from __future__ import annotations

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
    "ExpectedToolCall",
    "ToolCallEvaluation",
    "ToolUseMetrics",
    "arg_presence_rate",
    "evaluate_tool_use",
    "tool_error_rate",
    "tool_success_rate",
]
