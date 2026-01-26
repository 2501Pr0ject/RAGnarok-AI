"""Multi-step reasoning evaluation for agents.

This module provides evaluators for assessing the quality of
multi-step reasoning in agent trajectories.

Example:
    >>> from ragnarok_ai.agents import AgentResponse, AgentStep
    >>> from ragnarok_ai.agents.evaluators import (
    ...     ReasoningCoherenceEvaluator,
    ...     GoalProgressEvaluator,
    ...     ReasoningEfficiencyEvaluator,
    ... )
    >>>
    >>> # LLM-based coherence evaluation
    >>> coherence = ReasoningCoherenceEvaluator(llm)
    >>> score = await coherence.evaluate(response)
    >>>
    >>> # Pure function efficiency evaluation
    >>> efficiency = ReasoningEfficiencyEvaluator()
    >>> result = efficiency.evaluate(response, optimal_steps=3)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.agents.types import AgentResponse, AgentStep
    from ragnarok_ai.core.protocols import LLMProtocol


# =============================================================================
# Types
# =============================================================================


@dataclass(frozen=True)
class StepCoherence:
    """Coherence evaluation for a single step transition.

    Attributes:
        from_step: Index of source step.
        to_step: Index of target step.
        is_coherent: Whether transition is logically coherent.
        reasoning: LLM explanation for the decision.

    Example:
        >>> coherence = StepCoherence(
        ...     from_step=0,
        ...     to_step=1,
        ...     is_coherent=True,
        ...     reasoning="Step 2 naturally follows from the search results.",
        ... )
    """

    from_step: int
    to_step: int
    is_coherent: bool
    reasoning: str


@dataclass
class CoherenceResult:
    """Result of reasoning coherence evaluation.

    Attributes:
        score: Proportion of coherent transitions (0.0-1.0).
        transitions: Per-transition coherence evaluations.
        summary: Human-readable summary.

    Example:
        >>> result.score
        0.75
        >>> len(result.transitions)
        4
    """

    score: float
    transitions: list[StepCoherence] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class StepProgress:
    """Progress evaluation for a single step.

    Attributes:
        step_index: Index of the step (0-based).
        progress_score: Progress toward goal (0.0-1.0).
        reasoning: LLM explanation.

    Example:
        >>> progress = StepProgress(
        ...     step_index=2,
        ...     progress_score=0.6,
        ...     reasoning="Found relevant information, 60% complete.",
        ... )
    """

    step_index: int
    progress_score: float
    reasoning: str


@dataclass
class GoalProgressResult:
    """Result of goal progress evaluation.

    Attributes:
        final_score: Progress at the last step (0.0-1.0).
        is_monotonic: Whether progress always increased.
        steps: Per-step progress evaluations.
        summary: Human-readable summary.

    Example:
        >>> result.final_score
        1.0
        >>> result.is_monotonic
        True
    """

    final_score: float
    is_monotonic: bool
    steps: list[StepProgress] = field(default_factory=list)
    summary: str = ""


@dataclass
class EfficiencyResult:
    """Result of reasoning efficiency evaluation.

    Attributes:
        score: Efficiency score (0.0-1.0).
        actual_steps: Number of steps taken.
        optimal_steps: Expected minimum steps (if provided).
        redundant_steps: Number of redundant/similar steps detected.
        loop_count: Number of loops detected.
        summary: Human-readable summary.

    Example:
        >>> result = EfficiencyResult(
        ...     score=0.6,
        ...     actual_steps=5,
        ...     optimal_steps=3,
        ...     redundant_steps=1,
        ...     loop_count=1,
        ...     summary="Agent took 5 steps instead of optimal 3.",
        ... )
    """

    score: float
    actual_steps: int
    optimal_steps: int | None = None
    redundant_steps: int = 0
    loop_count: int = 0
    summary: str = ""


# =============================================================================
# Prompts
# =============================================================================

COHERENCE_PROMPT = """Analyze if the transition between two consecutive reasoning steps is logically coherent.

Step {from_idx} ({from_type}): {from_content}

Step {to_idx} ({to_type}): {to_content}

A coherent transition means Step {to_idx} naturally follows from Step {from_idx}.
Consider: Does the agent's reasoning flow logically? Is there a clear connection?

Return JSON only:
{{"is_coherent": true or false, "reasoning": "brief explanation"}}"""

PROGRESS_PROMPT = """Evaluate progress toward completing a task.

Task: {task}

Step {idx}/{total} ({step_type}): {content}

On a scale of 0-100, how much progress has been made toward completing the task?
- 0 = no progress at all
- 50 = halfway done
- 100 = task fully completed

Return JSON only:
{{"progress": 0-100, "reasoning": "brief explanation"}}"""


# =============================================================================
# Evaluators
# =============================================================================


class ReasoningCoherenceEvaluator:
    """Evaluate logical flow between reasoning steps.

    Uses an LLM to assess whether each step transition is logically
    coherent. A coherent trajectory has steps that naturally follow
    from each other.

    Attributes:
        llm: The LLM provider for evaluation.

    Example:
        >>> evaluator = ReasoningCoherenceEvaluator(llm)
        >>> score = await evaluator.evaluate(response)
        >>> print(f"Coherence: {score:.1%}")
        Coherence: 80.0%
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize the evaluator.

        Args:
            llm: LLM provider implementing LLMProtocol.
        """
        self.llm = llm

    async def evaluate(self, response: AgentResponse) -> float:
        """Evaluate coherence and return score.

        Args:
            response: Agent response with steps to evaluate.

        Returns:
            Coherence score between 0.0 and 1.0.
        """
        result = await self.evaluate_detailed(response)
        return result.score

    async def evaluate_detailed(self, response: AgentResponse) -> CoherenceResult:
        """Evaluate coherence with detailed transition analysis.

        Args:
            response: Agent response with steps to evaluate.

        Returns:
            CoherenceResult with score and per-transition details.
        """
        steps = response.steps

        # Handle edge cases
        if len(steps) <= 1:
            return CoherenceResult(
                score=1.0,
                transitions=[],
                summary="Single step or empty trajectory is trivially coherent.",
            )

        # Evaluate each transition
        transitions: list[StepCoherence] = []
        for i in range(len(steps) - 1):
            coherence = await self._evaluate_transition(steps[i], steps[i + 1], i)
            transitions.append(coherence)

        # Calculate score
        coherent_count = sum(1 for t in transitions if t.is_coherent)
        score = coherent_count / len(transitions)

        # Generate summary
        if score == 1.0:
            summary = "All step transitions are logically coherent."
        elif score == 0.0:
            summary = "No step transitions are logically coherent."
        else:
            summary = f"{coherent_count}/{len(transitions)} transitions are coherent."

        return CoherenceResult(
            score=score,
            transitions=transitions,
            summary=summary,
        )

    async def _evaluate_transition(
        self,
        from_step: AgentStep,
        to_step: AgentStep,
        from_idx: int,
    ) -> StepCoherence:
        """Evaluate a single step transition.

        Args:
            from_step: Source step.
            to_step: Target step.
            from_idx: Index of source step.

        Returns:
            StepCoherence evaluation result.
        """
        prompt = COHERENCE_PROMPT.format(
            from_idx=from_idx + 1,
            to_idx=from_idx + 2,
            from_type=from_step.step_type,
            from_content=from_step.content[:500],
            to_type=to_step.step_type,
            to_content=to_step.content[:500],
        )

        try:
            llm_response = await self.llm.generate(prompt)
            result = self._parse_json(llm_response)

            return StepCoherence(
                from_step=from_idx,
                to_step=from_idx + 1,
                is_coherent=bool(result.get("is_coherent", False)),
                reasoning=str(result.get("reasoning", "No reasoning provided.")),
            )
        except Exception:
            # Default to coherent on parse failure
            return StepCoherence(
                from_step=from_idx,
                to_step=from_idx + 1,
                is_coherent=True,
                reasoning="Evaluation failed, defaulting to coherent.",
            )

    def _parse_json(self, text: str) -> dict[str, object]:
        """Parse JSON from LLM response."""
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {}


class GoalProgressEvaluator:
    """Evaluate progress toward the goal at each step.

    Uses an LLM to assess how much progress has been made at each
    step toward completing the given task.

    Attributes:
        llm: The LLM provider for evaluation.

    Example:
        >>> evaluator = GoalProgressEvaluator(llm)
        >>> result = await evaluator.evaluate_detailed(response, task="Find the capital of France")
        >>> print(f"Final progress: {result.final_score:.1%}")
        >>> print(f"Monotonic: {result.is_monotonic}")
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize the evaluator.

        Args:
            llm: LLM provider implementing LLMProtocol.
        """
        self.llm = llm

    async def evaluate(self, response: AgentResponse, task: str) -> float:
        """Evaluate goal progress and return final score.

        Args:
            response: Agent response with steps to evaluate.
            task: The task/goal the agent was trying to accomplish.

        Returns:
            Final progress score between 0.0 and 1.0.
        """
        result = await self.evaluate_detailed(response, task)
        return result.final_score

    async def evaluate_detailed(
        self,
        response: AgentResponse,
        task: str,
    ) -> GoalProgressResult:
        """Evaluate goal progress with per-step details.

        Args:
            response: Agent response with steps to evaluate.
            task: The task/goal the agent was trying to accomplish.

        Returns:
            GoalProgressResult with final score and per-step progress.
        """
        steps = response.steps

        # Handle empty case
        if not steps:
            return GoalProgressResult(
                final_score=0.0,
                is_monotonic=True,
                steps=[],
                summary="No steps to evaluate.",
            )

        # Evaluate progress at each step
        step_progress: list[StepProgress] = []
        for i, step in enumerate(steps):
            progress = await self._evaluate_step_progress(step, task, i, len(steps))
            step_progress.append(progress)

        # Calculate final score and monotonicity
        final_score = step_progress[-1].progress_score if step_progress else 0.0

        is_monotonic = True
        for i in range(1, len(step_progress)):
            if step_progress[i].progress_score < step_progress[i - 1].progress_score:
                is_monotonic = False
                break

        # Generate summary
        if final_score >= 0.9:
            summary = "Task completed successfully."
        elif final_score >= 0.5:
            summary = f"Partial progress ({final_score:.0%}) toward goal."
        else:
            summary = f"Limited progress ({final_score:.0%}) toward goal."

        if not is_monotonic:
            summary += " Progress was not monotonic (some steps reduced progress)."

        return GoalProgressResult(
            final_score=final_score,
            is_monotonic=is_monotonic,
            steps=step_progress,
            summary=summary,
        )

    async def _evaluate_step_progress(
        self,
        step: AgentStep,
        task: str,
        idx: int,
        total: int,
    ) -> StepProgress:
        """Evaluate progress at a single step.

        Args:
            step: The step to evaluate.
            task: The task being accomplished.
            idx: Step index (0-based).
            total: Total number of steps.

        Returns:
            StepProgress evaluation result.
        """
        prompt = PROGRESS_PROMPT.format(
            task=task,
            idx=idx + 1,
            total=total,
            step_type=step.step_type,
            content=step.content[:500],
        )

        try:
            llm_response = await self.llm.generate(prompt)
            result = self._parse_json(llm_response)

            raw_progress = result.get("progress", 0)
            try:
                progress_val = float(str(raw_progress)) if raw_progress is not None else 0.0
            except (ValueError, TypeError):
                progress_val = 0.0
            progress = max(0.0, min(1.0, progress_val / 100.0))

            return StepProgress(
                step_index=idx,
                progress_score=progress,
                reasoning=str(result.get("reasoning", "No reasoning provided.")),
            )
        except Exception:
            # Default to 0 progress on failure
            return StepProgress(
                step_index=idx,
                progress_score=0.0,
                reasoning="Evaluation failed.",
            )

    def _parse_json(self, text: str) -> dict[str, object]:
        """Parse JSON from LLM response."""
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {}


class ReasoningEfficiencyEvaluator:
    """Evaluate reasoning efficiency without LLM.

    Detects inefficiencies like redundant steps, loops, and
    excessive step counts. This is a pure-function evaluator
    that doesn't require an LLM.

    Example:
        >>> evaluator = ReasoningEfficiencyEvaluator()
        >>> result = evaluator.evaluate(response, optimal_steps=3)
        >>> print(f"Efficiency: {result.score:.1%}")
        >>> print(f"Loops detected: {result.loop_count}")
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        """Initialize the evaluator.

        Args:
            similarity_threshold: Threshold for considering steps similar (0.0-1.0).
        """
        self.similarity_threshold = similarity_threshold

    def evaluate(
        self,
        response: AgentResponse,
        optimal_steps: int | None = None,
    ) -> EfficiencyResult:
        """Evaluate reasoning efficiency.

        Args:
            response: Agent response with steps to evaluate.
            optimal_steps: Expected minimum steps (optional).

        Returns:
            EfficiencyResult with score and detected inefficiencies.
        """
        steps = response.steps
        actual_steps = len(steps)

        # Handle empty case
        if actual_steps == 0:
            return EfficiencyResult(
                score=1.0,
                actual_steps=0,
                optimal_steps=optimal_steps,
                redundant_steps=0,
                loop_count=0,
                summary="No steps to evaluate.",
            )

        # Detect redundant steps (similar content)
        redundant_steps = self._detect_redundant_steps(steps)

        # Detect loops (repeated tool calls)
        loop_count = self._detect_loops(steps)

        # Calculate score
        score = self._calculate_score(
            actual_steps=actual_steps,
            optimal_steps=optimal_steps,
            redundant_steps=redundant_steps,
            loop_count=loop_count,
        )

        # Generate summary
        summary = self._generate_summary(
            actual_steps=actual_steps,
            optimal_steps=optimal_steps,
            redundant_steps=redundant_steps,
            loop_count=loop_count,
            score=score,
        )

        return EfficiencyResult(
            score=score,
            actual_steps=actual_steps,
            optimal_steps=optimal_steps,
            redundant_steps=redundant_steps,
            loop_count=loop_count,
            summary=summary,
        )

    def _detect_redundant_steps(self, steps: list[AgentStep]) -> int:
        """Detect steps with similar content.

        Args:
            steps: List of agent steps.

        Returns:
            Number of redundant steps detected.
        """
        if len(steps) < 2:
            return 0

        redundant = 0
        seen_contents: list[str] = []

        for step in steps:
            content = step.content.strip().lower()
            if not content:
                continue

            for seen in seen_contents:
                similarity = SequenceMatcher(None, content, seen).ratio()
                if similarity >= self.similarity_threshold:
                    redundant += 1
                    break
            else:
                seen_contents.append(content)

        return redundant

    def _detect_loops(self, steps: list[AgentStep]) -> int:
        """Detect repeated consecutive tool calls.

        Args:
            steps: List of agent steps.

        Returns:
            Number of loops detected.
        """
        if len(steps) < 2:
            return 0

        loops = 0
        prev_tool_signature: str | None = None

        for step in steps:
            if step.tool_call:
                # Create signature from tool name and input
                signature = f"{step.tool_call.name}:{sorted(step.tool_call.input.items())}"

                if signature == prev_tool_signature:
                    loops += 1

                prev_tool_signature = signature
            else:
                prev_tool_signature = None

        return loops

    def _calculate_score(
        self,
        actual_steps: int,
        optimal_steps: int | None,
        redundant_steps: int,
        loop_count: int,
    ) -> float:
        """Calculate efficiency score.

        Args:
            actual_steps: Number of steps taken.
            optimal_steps: Expected minimum steps.
            redundant_steps: Number of redundant steps.
            loop_count: Number of loops.

        Returns:
            Efficiency score between 0.0 and 1.0.
        """
        if actual_steps == 0:
            return 1.0

        # Base score from optimal comparison (if provided)
        base_score = (
            min(1.0, optimal_steps / actual_steps)
            if optimal_steps is not None and optimal_steps > 0
            else 1.0
        )

        # Penalties for inefficiencies
        redundant_penalty = 0.1 * redundant_steps
        loop_penalty = 0.15 * loop_count

        score = base_score - redundant_penalty - loop_penalty
        return max(0.0, min(1.0, score))

    def _generate_summary(
        self,
        actual_steps: int,
        optimal_steps: int | None,
        redundant_steps: int,
        loop_count: int,
        score: float,
    ) -> str:
        """Generate human-readable summary.

        Args:
            actual_steps: Number of steps taken.
            optimal_steps: Expected minimum steps.
            redundant_steps: Number of redundant steps.
            loop_count: Number of loops.
            score: Calculated efficiency score.

        Returns:
            Summary string.
        """
        parts = []

        if optimal_steps is not None:
            if actual_steps <= optimal_steps:
                parts.append(f"Efficient: {actual_steps} steps (optimal: {optimal_steps}).")
            else:
                parts.append(f"Took {actual_steps} steps (optimal: {optimal_steps}).")

        if redundant_steps > 0:
            parts.append(f"Detected {redundant_steps} redundant step(s).")

        if loop_count > 0:
            parts.append(f"Detected {loop_count} loop(s).")

        if not parts:
            if score >= 0.9:
                parts.append("Reasoning was efficient.")
            else:
                parts.append(f"Efficiency score: {score:.0%}.")

        return " ".join(parts)
