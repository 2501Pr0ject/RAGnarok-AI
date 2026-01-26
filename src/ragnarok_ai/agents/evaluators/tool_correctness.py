"""Tool-use correctness metrics for agent evaluation.

This module provides pure-function metrics (no LLM required) for evaluating
whether an agent uses its tools correctly: right tools, right arguments,
proper error handling.

Example:
    >>> from ragnarok_ai.agents import AgentResponse, AgentStep, ToolCall
    >>> from ragnarok_ai.agents.evaluators import evaluate_tool_use, ExpectedToolCall
    >>>
    >>> response = AgentResponse(
    ...     answer="Result",
    ...     steps=[
    ...         AgentStep(
    ...             step_type="action",
    ...             content="Calling tool",
    ...             tool_call=ToolCall(name="search", input={"q": "test"}, output="found"),
    ...         ),
    ...     ],
    ... )
    >>> metrics = evaluate_tool_use(response)
    >>> print(f"Success rate: {metrics.success_rate:.1%}")
    Success rate: 100.0%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.agents.types import AgentResponse, ToolCall


@dataclass(frozen=True)
class ExpectedToolCall:
    """Expected tool call for ground truth comparison.

    Use this to define what tools should be called and with what arguments.
    Only the tool name is required; argument validation is optional.

    Attributes:
        name: Name of the expected tool.
        required_args: Set of argument names that must be present.
        optional_args: Set of argument names that may be present.

    Example:
        >>> expected = ExpectedToolCall(
        ...     name="get_weather",
        ...     required_args={"city"},
        ...     optional_args={"date", "units"},
        ... )
    """

    name: str
    required_args: frozenset[str] = field(default_factory=frozenset)
    optional_args: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolCallEvaluation:
    """Evaluation result for a single tool call.

    Captures whether the tool call was correct in terms of name,
    arguments, and execution success.

    Attributes:
        tool_call: The evaluated tool call.
        name_correct: Whether tool name matches expected (True if no expected).
        args_present: Proportion of required args present (0.0-1.0).
        success: Whether the tool call succeeded (no error).

    Example:
        >>> evaluation = ToolCallEvaluation(
        ...     tool_call=tool_call,
        ...     name_correct=True,
        ...     args_present=0.5,
        ...     success=True,
        ... )
        >>> evaluation.score
        0.75
    """

    tool_call: ToolCall
    name_correct: bool
    args_present: float
    success: bool

    @property
    def score(self) -> float:
        """Weighted score for this tool call.

        Formula: 0.4 * name_correct + 0.3 * args_present + 0.3 * success

        Returns:
            Score between 0.0 and 1.0.
        """
        return 0.4 * float(self.name_correct) + 0.3 * self.args_present + 0.3 * float(self.success)


@dataclass
class ToolUseMetrics:
    """Aggregated metrics for tool usage in an agent response.

    Contains two categories of metrics:
    1. Execution metrics (always computed): success rate, arg correctness
    2. Selection metrics (only with expected_tools): precision, recall, f1

    Attributes:
        total_calls: Total number of tool calls made.
        successful_calls: Number of calls that succeeded (no error).
        failed_calls: Number of calls that failed (with error).
        success_rate: Proportion of successful calls (0.0-1.0).
        arg_correctness: Average required args presence across calls (0.0-1.0).
        precision: Correct calls / total actual (None without expected_tools).
        recall: Correct calls / total expected (None without expected_tools).
        f1: Harmonic mean of precision and recall (None without expected_tools).
        correct_calls: Number of tools called that match expected.
        unnecessary_calls: Number of tools called not in expected.
        missing_calls: Number of expected tools not called.
        evaluations: Per-call evaluation breakdown.

    Example:
        >>> metrics = evaluate_tool_use(response)
        >>> print(f"Success: {metrics.success_rate:.1%}")
        >>> print(metrics.summary())
    """

    # Execution metrics (always computed)
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    arg_correctness: float

    # Selection metrics (vs expected tools - None if no expected_tools)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    # Selection counts (vs expected tools)
    correct_calls: int = 0
    unnecessary_calls: int = 0
    missing_calls: int = 0

    # Per-call breakdown
    evaluations: list[ToolCallEvaluation] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of metrics.

        Returns:
            Multi-line string with key metrics.
        """
        lines = [
            "Tool Use Metrics:",
            f"  Total calls: {self.total_calls}",
            f"  Success rate: {self.success_rate:.1%}",
            f"  Arg correctness: {self.arg_correctness:.1%}",
        ]

        if self.precision is not None:
            lines.extend(
                [
                    f"  Precision: {self.precision:.1%}",
                    f"  Recall: {self.recall:.1%}",
                    f"  F1: {self.f1:.1%}",
                    f"  Correct: {self.correct_calls}, "
                    f"Unnecessary: {self.unnecessary_calls}, "
                    f"Missing: {self.missing_calls}",
                ]
            )

        return "\n".join(lines)


def tool_success_rate(response: AgentResponse) -> float:
    """Proportion of tool calls that succeeded (no error).

    Args:
        response: Agent response with tool calls.

    Returns:
        Success rate between 0.0 and 1.0. Returns 1.0 if no tool calls.

    Example:
        >>> rate = tool_success_rate(response)
        >>> print(f"Success rate: {rate:.1%}")
    """
    tool_calls = response.tool_calls
    if not tool_calls:
        return 1.0
    successful = sum(1 for tc in tool_calls if tc.success)
    return successful / len(tool_calls)


def tool_error_rate(response: AgentResponse) -> float:
    """Proportion of tool calls that failed.

    Args:
        response: Agent response with tool calls.

    Returns:
        Error rate between 0.0 and 1.0. Returns 0.0 if no tool calls.

    Example:
        >>> rate = tool_error_rate(response)
        >>> print(f"Error rate: {rate:.1%}")
    """
    return 1.0 - tool_success_rate(response)


def arg_presence_rate(
    tool_call: ToolCall,
    expected: ExpectedToolCall,
) -> float:
    """Proportion of required args present in tool call.

    Args:
        tool_call: The actual tool call made.
        expected: Expected tool call with required_args.

    Returns:
        Proportion between 0.0 and 1.0. Returns 1.0 if no required args.

    Example:
        >>> expected = ExpectedToolCall(name="search", required_args=frozenset({"query", "limit"}))
        >>> tool_call = ToolCall(name="search", input={"query": "test"}, output="ok")
        >>> arg_presence_rate(tool_call, expected)
        0.5
    """
    if not expected.required_args:
        return 1.0

    present = sum(1 for arg in expected.required_args if arg in tool_call.input)
    return present / len(expected.required_args)


def _find_matching_expected(
    tool_call: ToolCall,
    expected_tools: list[ExpectedToolCall],
    used_indices: set[int],
) -> tuple[ExpectedToolCall | None, int | None]:
    """Find matching expected tool for a tool call.

    Args:
        tool_call: The actual tool call.
        expected_tools: List of expected tools.
        used_indices: Set of already matched indices.

    Returns:
        Tuple of (matching ExpectedToolCall or None, index or None).
    """
    for i, expected in enumerate(expected_tools):
        if i not in used_indices and expected.name == tool_call.name:
            return expected, i
    return None, None


def evaluate_tool_use(
    response: AgentResponse,
    expected_tools: list[ExpectedToolCall] | None = None,
) -> ToolUseMetrics:
    """Evaluate tool usage in an agent response.

    Computes execution metrics (success rate, arg correctness) always.
    Computes selection metrics (precision, recall, f1) only when
    expected_tools is provided.

    Args:
        response: Agent response with tool calls.
        expected_tools: Optional ground truth for tool names/args.

    Returns:
        ToolUseMetrics with detailed breakdown.

    Example:
        >>> # Basic evaluation (no ground truth)
        >>> metrics = evaluate_tool_use(response)
        >>> print(f"Success rate: {metrics.success_rate:.1%}")

        >>> # With ground truth
        >>> expected = [
        ...     ExpectedToolCall(name="search", required_args=frozenset({"query"})),
        ...     ExpectedToolCall(name="calculate"),
        ... ]
        >>> metrics = evaluate_tool_use(response, expected_tools=expected)
        >>> print(f"F1: {metrics.f1:.1%}")
    """
    tool_calls = response.tool_calls

    # Handle empty case
    if not tool_calls:
        missing = len(expected_tools) if expected_tools else 0
        return ToolUseMetrics(
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            success_rate=1.0,
            arg_correctness=1.0,
            precision=1.0 if expected_tools is not None and missing == 0 else (0.0 if expected_tools else None),
            recall=0.0 if expected_tools and missing > 0 else (1.0 if expected_tools is not None else None),
            f1=0.0
            if expected_tools and missing > 0
            else (1.0 if expected_tools is not None and missing == 0 else None),
            correct_calls=0,
            unnecessary_calls=0,
            missing_calls=missing,
            evaluations=[],
        )

    # Track which expected tools have been matched
    used_indices: set[int] = set()
    evaluations: list[ToolCallEvaluation] = []

    # Evaluate each tool call
    for tc in tool_calls:
        if expected_tools:
            matched, idx = _find_matching_expected(tc, expected_tools, used_indices)
            if matched and idx is not None:
                used_indices.add(idx)
                name_correct = True
                args_present = arg_presence_rate(tc, matched)
            else:
                name_correct = False
                args_present = 1.0  # No expected args to validate
        else:
            name_correct = True  # No expected tools to compare
            args_present = 1.0

        evaluations.append(
            ToolCallEvaluation(
                tool_call=tc,
                name_correct=name_correct,
                args_present=args_present,
                success=tc.success,
            )
        )

    # Compute execution metrics
    total = len(tool_calls)
    successful = sum(1 for tc in tool_calls if tc.success)
    failed = total - successful
    success_rate = successful / total if total > 0 else 1.0
    arg_correctness = sum(e.args_present for e in evaluations) / len(evaluations) if evaluations else 1.0

    # Compute selection metrics if expected_tools provided
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    correct_calls = 0
    unnecessary_calls = 0
    missing_calls = 0

    if expected_tools is not None:
        correct_calls = len(used_indices)
        unnecessary_calls = total - correct_calls
        missing_calls = len(expected_tools) - correct_calls

        # Compute precision (ratio of correct to actual calls)
        precision = correct_calls / total if total > 0 else 1.0

        # Compute recall (ratio of correct to expected calls)
        recall = correct_calls / len(expected_tools) if expected_tools else 1.0

        # F1 = harmonic mean of precision and recall
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    return ToolUseMetrics(
        total_calls=total,
        successful_calls=successful,
        failed_calls=failed,
        success_rate=success_rate,
        arg_correctness=arg_correctness,
        precision=precision,
        recall=recall,
        f1=f1,
        correct_calls=correct_calls,
        unnecessary_calls=unnecessary_calls,
        missing_calls=missing_calls,
        evaluations=evaluations,
    )
