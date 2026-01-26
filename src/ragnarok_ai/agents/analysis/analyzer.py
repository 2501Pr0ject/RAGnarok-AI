"""Trajectory analysis for agent evaluation.

This module provides tools for analyzing agent execution trajectories,
including summary statistics, loop detection, and failure identification.

Example:
    >>> from ragnarok_ai.agents.analysis import TrajectoryAnalyzer
    >>>
    >>> analyzer = TrajectoryAnalyzer()
    >>> summary = analyzer.analyze(response)
    >>> print(summary)
    Steps: 5
    Tools: search (2x), calculate (1x)
    Latency: 2.3s
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ragnarok_ai.agents.types import AgentResponse


@dataclass
class TrajectorySummary:
    """Summary statistics for an agent trajectory.

    Attributes:
        total_steps: Total number of steps in trajectory.
        tool_calls: Mapping of tool names to call counts.
        total_latency_ms: Total execution time in milliseconds.
        step_types: Mapping of step types to counts.
        loops_detected: Number of detected reasoning loops.
        dead_ends: Number of dead-end steps detected.
        failed_tool_calls: Number of tool calls that failed.

    Example:
        >>> print(summary)
        Steps: 5
        Tools: search (2x), calculate (1x)
        Latency: 2.3s
        Loops: 0, Dead ends: 1, Failed: 0
    """

    total_steps: int
    tool_calls: dict[str, int] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    step_types: dict[str, int] = field(default_factory=dict)
    loops_detected: int = 0
    dead_ends: int = 0
    failed_tool_calls: int = 0

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [f"Steps: {self.total_steps}"]

        if self.tool_calls:
            tools_str = ", ".join(f"{name} ({count}x)" for name, count in sorted(self.tool_calls.items()))
            lines.append(f"Tools: {tools_str}")

        if self.total_latency_ms > 0:
            if self.total_latency_ms >= 1000:
                lines.append(f"Latency: {self.total_latency_ms / 1000:.1f}s")
            else:
                lines.append(f"Latency: {self.total_latency_ms:.0f}ms")

        issues = []
        if self.loops_detected > 0:
            issues.append(f"Loops: {self.loops_detected}")
        if self.dead_ends > 0:
            issues.append(f"Dead ends: {self.dead_ends}")
        if self.failed_tool_calls > 0:
            issues.append(f"Failed: {self.failed_tool_calls}")

        if issues:
            lines.append(", ".join(issues))

        return "\n".join(lines)


@dataclass(frozen=True)
class FailurePoint:
    """A detected failure in the trajectory.

    Attributes:
        step_index: Index of the failing step (0-based).
        failure_type: Type of failure detected.
        description: Human-readable description.
        severity: Severity level (warning or error).

    Example:
        >>> failure = FailurePoint(
        ...     step_index=3,
        ...     failure_type="tool_error",
        ...     description="API call timed out",
        ...     severity="error",
        ... )
    """

    step_index: int
    failure_type: Literal["tool_error", "loop", "dead_end", "timeout"]
    description: str
    severity: Literal["warning", "error"] = "warning"


class TrajectoryAnalyzer:
    """Analyze agent execution trajectories.

    Provides methods to extract statistics, detect patterns like loops
    and dead ends, and identify failure points.

    Attributes:
        similarity_threshold: Threshold for detecting similar content.

    Example:
        >>> analyzer = TrajectoryAnalyzer()
        >>> summary = analyzer.analyze(response)
        >>> failures = analyzer.find_failures(response)
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """Initialize the analyzer.

        Args:
            similarity_threshold: Threshold for considering content similar.
        """
        self.similarity_threshold = similarity_threshold

    def analyze(self, response: AgentResponse) -> TrajectorySummary:
        """Generate summary statistics for a trajectory.

        Args:
            response: Agent response with trajectory to analyze.

        Returns:
            TrajectorySummary with statistics.
        """
        steps = response.steps

        # Count step types
        step_types: dict[str, int] = {}
        for step in steps:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1

        # Count tool calls and failures
        tool_calls: dict[str, int] = {}
        failed_tool_calls = 0
        for step in steps:
            if step.tool_call:
                name = step.tool_call.name
                tool_calls[name] = tool_calls.get(name, 0) + 1
                if not step.tool_call.success:
                    failed_tool_calls += 1

        # Detect patterns
        loops = self.detect_loops(response)
        dead_ends = self.detect_dead_ends(response)

        return TrajectorySummary(
            total_steps=len(steps),
            tool_calls=tool_calls,
            total_latency_ms=response.total_latency_ms,
            step_types=step_types,
            loops_detected=len(loops),
            dead_ends=len(dead_ends),
            failed_tool_calls=failed_tool_calls,
        )

    def find_failures(self, response: AgentResponse) -> list[FailurePoint]:
        """Identify failure points in trajectory.

        Args:
            response: Agent response with trajectory to analyze.

        Returns:
            List of detected failure points.
        """
        failures: list[FailurePoint] = []
        steps = response.steps

        # Check for tool errors
        for i, step in enumerate(steps):
            if step.tool_call and not step.tool_call.success:
                failures.append(
                    FailurePoint(
                        step_index=i,
                        failure_type="tool_error",
                        description=f"Tool '{step.tool_call.name}' failed: {step.tool_call.error or 'unknown error'}",
                        severity="error",
                    )
                )

        # Check for loops
        loops = self.detect_loops(response)
        for start_idx, end_idx in loops:
            failures.append(
                FailurePoint(
                    step_index=start_idx,
                    failure_type="loop",
                    description=f"Reasoning loop detected between steps {start_idx + 1} and {end_idx + 1}",
                    severity="warning",
                )
            )

        # Check for dead ends
        dead_ends = self.detect_dead_ends(response)
        for idx in dead_ends:
            failures.append(
                FailurePoint(
                    step_index=idx,
                    failure_type="dead_end",
                    description=f"Step {idx + 1} appears to be a dead end",
                    severity="warning",
                )
            )

        # Sort by step index
        failures.sort(key=lambda f: f.step_index)
        return failures

    def detect_loops(self, response: AgentResponse) -> list[tuple[int, int]]:
        """Detect reasoning loops in trajectory.

        A loop is detected when:
        - Same tool is called with same/similar input consecutively
        - Similar thought content appears multiple times

        Args:
            response: Agent response with trajectory.

        Returns:
            List of (start_index, end_index) tuples for detected loops.
        """
        steps = response.steps
        loops: list[tuple[int, int]] = []

        if len(steps) < 2:
            return loops

        # Check for consecutive similar tool calls
        prev_signature: str | None = None
        prev_idx: int | None = None

        for i, step in enumerate(steps):
            if step.tool_call:
                signature = f"{step.tool_call.name}:{sorted(step.tool_call.input.items())}"
                if signature == prev_signature and prev_idx is not None:
                    loops.append((prev_idx, i))
                prev_signature = signature
                prev_idx = i
            else:
                prev_signature = None
                prev_idx = None

        # Check for similar thought content (non-consecutive)
        thought_indices: list[tuple[int, str]] = []
        for i, step in enumerate(steps):
            if step.step_type == "thought" and step.content:
                thought_indices.append((i, step.content.lower().strip()))

        for i, (idx1, content1) in enumerate(thought_indices):
            for idx2, content2 in thought_indices[i + 1 :]:
                if idx2 - idx1 > 1:  # Not consecutive
                    similarity = SequenceMatcher(None, content1, content2).ratio()
                    if similarity >= self.similarity_threshold:
                        loops.append((idx1, idx2))

        return loops

    def detect_dead_ends(self, response: AgentResponse) -> list[int]:
        """Detect dead-end steps in trajectory.

        A dead-end is:
        - Failed tool call with no retry
        - Last step before a completely different approach

        Args:
            response: Agent response with trajectory.

        Returns:
            List of step indices that are dead ends.
        """
        steps = response.steps
        dead_ends: list[int] = []

        if len(steps) < 2:
            return dead_ends

        for i, step in enumerate(steps):
            # Failed tool call not followed by similar retry
            tc = step.tool_call
            if tc and not tc.success:
                # Check if next steps retry with same tool
                has_retry = False
                for j in range(i + 1, min(i + 3, len(steps))):
                    next_tc = steps[j].tool_call
                    if next_tc and next_tc.name == tc.name:
                        has_retry = True
                        break
                if not has_retry:
                    dead_ends.append(i)

        return dead_ends
