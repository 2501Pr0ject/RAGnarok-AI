"""Type definitions for agent evaluation.

This module provides the core types for representing agent execution
trajectories, tool calls, and responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    """A single tool/function call made by an agent.

    Immutable to prevent accidental modification after creation.

    Attributes:
        name: Name of the tool called.
        input: Arguments passed to the tool.
        output: Result returned by the tool.
        error: Error message if the tool call failed.
        latency_ms: Time taken to execute the tool call.

    Example:
        >>> tool_call = ToolCall(
        ...     name="search",
        ...     input={"query": "python docs"},
        ...     output="Found 10 results...",
        ...     latency_ms=150.0,
        ... )
        >>> tool_call.success
        True
    """

    name: str
    input: dict[str, Any]
    output: str
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Whether the tool call succeeded (no error)."""
        return self.error is None


@dataclass(frozen=True)
class AgentStep:
    """A single step in agent execution.

    Captures both reasoning (thought) and actions (tool calls).
    Supports ReAct pattern: Thought -> Action -> Observation

    Immutable to prevent accidental modification after creation.

    Attributes:
        step_type: Type of step (thought, action, observation, final_answer).
        content: The actual content of the step.
        tool_call: Tool call details if this is an action step.
        latency_ms: Time taken for this step.
        metadata: Additional metadata for the step.

    Example:
        >>> step = AgentStep(
        ...     step_type="thought",
        ...     content="I need to search for this information.",
        ... )
    """

    step_type: Literal["thought", "action", "observation", "final_answer"]
    content: str
    tool_call: ToolCall | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent execution.

    Similar to RAGResponse but with trajectory instead of retrieved_docs.
    Contains the final answer and the full execution trajectory.

    Attributes:
        answer: The final answer from the agent.
        steps: Full execution trajectory as list of steps.
        total_latency_ms: Total time for agent execution.
        metadata: Additional metadata.

    Example:
        >>> response = AgentResponse(
        ...     answer="The capital of France is Paris.",
        ...     steps=[
        ...         AgentStep(step_type="thought", content="Let me recall..."),
        ...         AgentStep(step_type="final_answer", content="Paris"),
        ...     ],
        ... )
        >>> response.num_steps
        2
    """

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Extract all tool calls from steps."""
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def num_steps(self) -> int:
        """Number of steps in the trajectory."""
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        """Number of tool calls made."""
        return len(self.tool_calls)

    @property
    def thoughts(self) -> list[str]:
        """Extract all thought step contents."""
        return [s.content for s in self.steps if s.step_type == "thought"]

    @property
    def reasoning_trace(self) -> str:
        """Format full reasoning trace as string.

        Returns:
            Multi-line string showing the execution flow.
        """
        lines = []
        for i, step in enumerate(self.steps, 1):
            prefix = f"[{i}] {step.step_type.upper()}"
            if step.tool_call:
                lines.append(f"{prefix}: {step.tool_call.name}({step.tool_call.input})")
                lines.append(f"    -> {step.tool_call.output[:100]}...")
            else:
                content_preview = step.content[:100]
                if len(step.content) > 100:
                    content_preview += "..."
                lines.append(f"{prefix}: {content_preview}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "answer": self.answer,
            "steps": [
                {
                    "step_type": s.step_type,
                    "content": s.content,
                    "tool_call": {
                        "name": s.tool_call.name,
                        "input": s.tool_call.input,
                        "output": s.tool_call.output,
                        "error": s.tool_call.error,
                        "latency_ms": s.tool_call.latency_ms,
                    }
                    if s.tool_call
                    else None,
                    "latency_ms": s.latency_ms,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "total_latency_ms": self.total_latency_ms,
            "metadata": self.metadata,
        }
