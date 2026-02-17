"""Unit tests for reasoning evaluators."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragnarok_ai.agents import AgentResponse, AgentStep, ToolCall
from ragnarok_ai.agents.evaluators import (
    CoherenceResult,
    EfficiencyResult,
    GoalProgressEvaluator,
    GoalProgressResult,
    ReasoningCoherenceEvaluator,
    ReasoningEfficiencyEvaluator,
    StepCoherence,
    StepProgress,
)

# =============================================================================
# StepCoherence Tests
# =============================================================================


class TestStepCoherence:
    """Tests for StepCoherence dataclass."""

    def test_creation(self) -> None:
        """Create a step coherence evaluation."""
        coherence = StepCoherence(
            from_step=0,
            to_step=1,
            is_coherent=True,
            reasoning="Logical flow.",
        )
        assert coherence.from_step == 0
        assert coherence.to_step == 1
        assert coherence.is_coherent is True
        assert coherence.reasoning == "Logical flow."

    def test_frozen(self) -> None:
        """StepCoherence is immutable."""
        coherence = StepCoherence(
            from_step=0,
            to_step=1,
            is_coherent=True,
            reasoning="test",
        )
        with pytest.raises(AttributeError):
            coherence.is_coherent = False  # type: ignore[misc]


# =============================================================================
# CoherenceResult Tests
# =============================================================================


class TestCoherenceResult:
    """Tests for CoherenceResult dataclass."""

    def test_creation(self) -> None:
        """Create a coherence result."""
        result = CoherenceResult(
            score=0.75,
            transitions=[
                StepCoherence(0, 1, True, "ok"),
                StepCoherence(1, 2, True, "ok"),
                StepCoherence(2, 3, True, "ok"),
                StepCoherence(3, 4, False, "jump"),
            ],
            summary="3/4 transitions coherent.",
        )
        assert result.score == 0.75
        assert len(result.transitions) == 4


# =============================================================================
# StepProgress Tests
# =============================================================================


class TestStepProgress:
    """Tests for StepProgress dataclass."""

    def test_creation(self) -> None:
        """Create a step progress evaluation."""
        progress = StepProgress(
            step_index=2,
            progress_score=0.6,
            reasoning="60% complete.",
        )
        assert progress.step_index == 2
        assert progress.progress_score == 0.6

    def test_frozen(self) -> None:
        """StepProgress is immutable."""
        progress = StepProgress(0, 0.5, "test")
        with pytest.raises(AttributeError):
            progress.progress_score = 0.9  # type: ignore[misc]


# =============================================================================
# GoalProgressResult Tests
# =============================================================================


class TestGoalProgressResult:
    """Tests for GoalProgressResult dataclass."""

    def test_creation(self) -> None:
        """Create a goal progress result."""
        result = GoalProgressResult(
            final_score=1.0,
            is_monotonic=True,
            steps=[
                StepProgress(0, 0.3, "started"),
                StepProgress(1, 0.6, "halfway"),
                StepProgress(2, 1.0, "done"),
            ],
            summary="Task completed.",
        )
        assert result.final_score == 1.0
        assert result.is_monotonic is True
        assert len(result.steps) == 3


# =============================================================================
# EfficiencyResult Tests
# =============================================================================


class TestEfficiencyResult:
    """Tests for EfficiencyResult dataclass."""

    def test_creation(self) -> None:
        """Create an efficiency result."""
        result = EfficiencyResult(
            score=0.6,
            actual_steps=5,
            optimal_steps=3,
            redundant_steps=1,
            loop_count=1,
            summary="5 steps, optimal 3.",
        )
        assert result.score == 0.6
        assert result.actual_steps == 5
        assert result.optimal_steps == 3
        assert result.redundant_steps == 1
        assert result.loop_count == 1


# =============================================================================
# ReasoningCoherenceEvaluator Tests
# =============================================================================


class TestReasoningCoherenceEvaluator:
    """Tests for ReasoningCoherenceEvaluator."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create a mock LLM."""
        llm = AsyncMock()
        llm.is_local = True
        return llm

    @pytest.mark.asyncio
    async def test_empty_steps(self, mock_llm: AsyncMock) -> None:
        """Empty response is trivially coherent."""
        evaluator = ReasoningCoherenceEvaluator(mock_llm)
        response = AgentResponse(answer="done", steps=[])

        result = await evaluator.evaluate_detailed(response)

        assert result.score == 1.0
        assert result.transitions == []

    @pytest.mark.asyncio
    async def test_single_step(self, mock_llm: AsyncMock) -> None:
        """Single step is trivially coherent."""
        evaluator = ReasoningCoherenceEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[AgentStep(step_type="thought", content="thinking")],
        )

        result = await evaluator.evaluate_detailed(response)

        assert result.score == 1.0
        assert result.transitions == []

    @pytest.mark.asyncio
    async def test_all_coherent(self, mock_llm: AsyncMock) -> None:
        """All transitions are coherent."""
        mock_llm.generate.return_value = '{"is_coherent": true, "reasoning": "ok"}'

        evaluator = ReasoningCoherenceEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="I need to search"),
                AgentStep(step_type="action", content="Searching..."),
                AgentStep(step_type="observation", content="Found results"),
            ],
        )

        result = await evaluator.evaluate_detailed(response)

        assert result.score == 1.0
        assert len(result.transitions) == 2
        assert all(t.is_coherent for t in result.transitions)

    @pytest.mark.asyncio
    async def test_some_incoherent(self, mock_llm: AsyncMock) -> None:
        """Some transitions are incoherent."""
        mock_llm.generate.side_effect = [
            '{"is_coherent": true, "reasoning": "ok"}',
            '{"is_coherent": false, "reasoning": "jump in logic"}',
        ]

        evaluator = ReasoningCoherenceEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="step 1"),
                AgentStep(step_type="thought", content="step 2"),
                AgentStep(step_type="thought", content="step 3"),
            ],
        )

        result = await evaluator.evaluate_detailed(response)

        assert result.score == 0.5
        assert result.transitions[0].is_coherent is True
        assert result.transitions[1].is_coherent is False

    @pytest.mark.asyncio
    async def test_evaluate_returns_score(self, mock_llm: AsyncMock) -> None:
        """evaluate() returns just the score."""
        mock_llm.generate.return_value = '{"is_coherent": true, "reasoning": "ok"}'

        evaluator = ReasoningCoherenceEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="step 1"),
                AgentStep(step_type="thought", content="step 2"),
            ],
        )

        score = await evaluator.evaluate(response)

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_malformed_json_defaults_incoherent(self, mock_llm: AsyncMock) -> None:
        """Malformed LLM response defaults to incoherent."""
        mock_llm.generate.return_value = "not valid json"

        evaluator = ReasoningCoherenceEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="step 1"),
                AgentStep(step_type="thought", content="step 2"),
            ],
        )

        result = await evaluator.evaluate_detailed(response)

        assert result.score == 0.0
        assert result.transitions[0].is_coherent is False


# =============================================================================
# GoalProgressEvaluator Tests
# =============================================================================


class TestGoalProgressEvaluator:
    """Tests for GoalProgressEvaluator."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create a mock LLM."""
        llm = AsyncMock()
        llm.is_local = True
        return llm

    @pytest.mark.asyncio
    async def test_empty_steps(self, mock_llm: AsyncMock) -> None:
        """Empty response has 0 progress."""
        evaluator = GoalProgressEvaluator(mock_llm)
        response = AgentResponse(answer="", steps=[])

        result = await evaluator.evaluate_detailed(response, task="Find X")

        assert result.final_score == 0.0
        assert result.is_monotonic is True
        assert result.steps == []

    @pytest.mark.asyncio
    async def test_monotonic_progress(self, mock_llm: AsyncMock) -> None:
        """Progress increases monotonically."""
        mock_llm.generate.side_effect = [
            '{"progress": 30, "reasoning": "started"}',
            '{"progress": 60, "reasoning": "halfway"}',
            '{"progress": 100, "reasoning": "done"}',
        ]

        evaluator = GoalProgressEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="step 1"),
                AgentStep(step_type="action", content="step 2"),
                AgentStep(step_type="final_answer", content="step 3"),
            ],
        )

        result = await evaluator.evaluate_detailed(response, task="Complete task")

        assert result.final_score == 1.0
        assert result.is_monotonic is True
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_non_monotonic_progress(self, mock_llm: AsyncMock) -> None:
        """Progress decreases at some point."""
        mock_llm.generate.side_effect = [
            '{"progress": 50, "reasoning": "good start"}',
            '{"progress": 30, "reasoning": "wrong path"}',
            '{"progress": 80, "reasoning": "recovered"}',
        ]

        evaluator = GoalProgressEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="step 1"),
                AgentStep(step_type="thought", content="step 2"),
                AgentStep(step_type="thought", content="step 3"),
            ],
        )

        result = await evaluator.evaluate_detailed(response, task="Task")

        assert result.final_score == 0.8
        assert result.is_monotonic is False

    @pytest.mark.asyncio
    async def test_evaluate_returns_final_score(self, mock_llm: AsyncMock) -> None:
        """evaluate() returns just the final score."""
        mock_llm.generate.return_value = '{"progress": 75, "reasoning": "mostly done"}'

        evaluator = GoalProgressEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[AgentStep(step_type="thought", content="thinking")],
        )

        score = await evaluator.evaluate(response, task="Task")

        assert score == 0.75

    @pytest.mark.asyncio
    async def test_progress_clamped(self, mock_llm: AsyncMock) -> None:
        """Progress is clamped to [0, 1]."""
        mock_llm.generate.return_value = '{"progress": 150, "reasoning": "over"}'

        evaluator = GoalProgressEvaluator(mock_llm)
        response = AgentResponse(
            answer="done",
            steps=[AgentStep(step_type="thought", content="step")],
        )

        result = await evaluator.evaluate_detailed(response, task="Task")

        assert result.final_score == 1.0


# =============================================================================
# ReasoningEfficiencyEvaluator Tests
# =============================================================================


class TestReasoningEfficiencyEvaluator:
    """Tests for ReasoningEfficiencyEvaluator."""

    def test_empty_steps(self) -> None:
        """Empty response is efficient."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(answer="", steps=[])

        result = evaluator.evaluate(response)

        assert result.score == 1.0
        assert result.actual_steps == 0

    def test_optimal_steps_exact(self) -> None:
        """Exact optimal steps gives score 1.0."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="think"),
                AgentStep(step_type="action", content="act"),
                AgentStep(step_type="final_answer", content="answer"),
            ],
        )

        result = evaluator.evaluate(response, optimal_steps=3)

        assert result.score == 1.0
        assert result.actual_steps == 3
        assert result.optimal_steps == 3

    def test_optimal_steps_exceeded(self) -> None:
        """More steps than optimal reduces score."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="a"),
                AgentStep(step_type="thought", content="b"),
                AgentStep(step_type="thought", content="c"),
                AgentStep(step_type="thought", content="d"),
                AgentStep(step_type="thought", content="e"),
                AgentStep(step_type="thought", content="f"),
            ],
        )

        result = evaluator.evaluate(response, optimal_steps=3)

        # 3/6 = 0.5 base score
        assert result.score == 0.5
        assert result.actual_steps == 6

    def test_redundant_step_detection(self) -> None:
        """Detect similar content steps."""
        evaluator = ReasoningEfficiencyEvaluator(similarity_threshold=0.8)
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="I need to search for information"),
                AgentStep(step_type="thought", content="I need to search for information about this"),
                AgentStep(step_type="action", content="Searching..."),
            ],
        )

        result = evaluator.evaluate(response)

        assert result.redundant_steps >= 1

    def test_loop_detection(self) -> None:
        """Detect repeated consecutive tool calls."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="search",
                    tool_call=ToolCall(name="search", input={"q": "test"}, output="r1"),
                ),
                AgentStep(
                    step_type="action",
                    content="search again",
                    tool_call=ToolCall(name="search", input={"q": "test"}, output="r2"),
                ),
                AgentStep(
                    step_type="action",
                    content="search once more",
                    tool_call=ToolCall(name="search", input={"q": "test"}, output="r3"),
                ),
            ],
        )

        result = evaluator.evaluate(response)

        assert result.loop_count == 2  # 2 repeated calls after the first

    def test_no_loops_different_tools(self) -> None:
        """Different tool calls are not loops."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="search",
                    tool_call=ToolCall(name="search", input={"q": "test"}, output="r1"),
                ),
                AgentStep(
                    step_type="action",
                    content="calculate",
                    tool_call=ToolCall(name="calculate", input={"x": 1}, output="1"),
                ),
            ],
        )

        result = evaluator.evaluate(response)

        assert result.loop_count == 0

    def test_no_loops_different_inputs(self) -> None:
        """Same tool with different inputs is not a loop."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="search 1",
                    tool_call=ToolCall(name="search", input={"q": "cats"}, output="r1"),
                ),
                AgentStep(
                    step_type="action",
                    content="search 2",
                    tool_call=ToolCall(name="search", input={"q": "dogs"}, output="r2"),
                ),
            ],
        )

        result = evaluator.evaluate(response)

        assert result.loop_count == 0

    def test_penalties_reduce_score(self) -> None:
        """Redundant steps and loops reduce score."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="search",
                    tool_call=ToolCall(name="search", input={"q": "x"}, output="r1"),
                ),
                AgentStep(
                    step_type="action",
                    content="search",
                    tool_call=ToolCall(name="search", input={"q": "x"}, output="r2"),
                ),
            ],
        )

        result = evaluator.evaluate(response)

        # 1 loop detected, penalty of 0.15
        assert result.score < 1.0
        assert result.loop_count == 1

    def test_summary_generation(self) -> None:
        """Summary is generated correctly."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="a"),
                AgentStep(step_type="thought", content="b"),
            ],
        )

        result = evaluator.evaluate(response, optimal_steps=2)

        assert "Efficient" in result.summary or "2 steps" in result.summary

    def test_no_optimal_steps(self) -> None:
        """Without optimal_steps, base score is 1.0."""
        evaluator = ReasoningEfficiencyEvaluator()
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="analyzing the problem requirements"),
                AgentStep(step_type="action", content="searching for relevant information"),
                AgentStep(step_type="final_answer", content="the answer is 42"),
            ],
        )

        result = evaluator.evaluate(response)

        assert result.score == 1.0
        assert result.optimal_steps is None
