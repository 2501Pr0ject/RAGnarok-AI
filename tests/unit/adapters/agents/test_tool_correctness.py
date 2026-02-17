"""Unit tests for tool-use correctness metrics."""

from __future__ import annotations

import pytest

from ragnarok_ai.agents import AgentResponse, AgentStep, ToolCall
from ragnarok_ai.agents.evaluators import (
    ExpectedToolCall,
    ToolCallEvaluation,
    ToolUseMetrics,
    arg_presence_rate,
    evaluate_tool_use,
    tool_error_rate,
    tool_success_rate,
)

# ============================================================================
# ExpectedToolCall Tests
# ============================================================================


class TestExpectedToolCall:
    """Tests for ExpectedToolCall dataclass."""

    def test_basic_creation(self) -> None:
        """Create expected tool with just name."""
        expected = ExpectedToolCall(name="search")
        assert expected.name == "search"
        assert expected.required_args == frozenset()
        assert expected.optional_args == frozenset()

    def test_with_required_args(self) -> None:
        """Create expected tool with required args."""
        expected = ExpectedToolCall(
            name="get_weather",
            required_args=frozenset({"city", "date"}),
        )
        assert expected.required_args == frozenset({"city", "date"})

    def test_with_optional_args(self) -> None:
        """Create expected tool with optional args."""
        expected = ExpectedToolCall(
            name="search",
            required_args=frozenset({"query"}),
            optional_args=frozenset({"limit", "offset"}),
        )
        assert expected.optional_args == frozenset({"limit", "offset"})

    def test_frozen_immutability(self) -> None:
        """ExpectedToolCall is immutable."""
        expected = ExpectedToolCall(name="test")
        with pytest.raises(AttributeError):
            expected.name = "different"  # type: ignore[misc]


# ============================================================================
# ToolCallEvaluation Tests
# ============================================================================


class TestToolCallEvaluation:
    """Tests for ToolCallEvaluation dataclass."""

    def test_score_all_correct(self) -> None:
        """Score is 1.0 when all metrics are perfect."""
        tc = ToolCall(name="search", input={"q": "test"}, output="found")
        evaluation = ToolCallEvaluation(
            tool_call=tc,
            name_correct=True,
            args_present=1.0,
            success=True,
        )
        assert evaluation.score == 1.0

    def test_score_all_wrong(self) -> None:
        """Score is 0.0 when all metrics fail."""
        tc = ToolCall(name="wrong", input={}, output="", error="failed")
        evaluation = ToolCallEvaluation(
            tool_call=tc,
            name_correct=False,
            args_present=0.0,
            success=False,
        )
        assert evaluation.score == 0.0

    def test_score_partial_args(self) -> None:
        """Score reflects partial arg presence."""
        tc = ToolCall(name="search", input={"q": "test"}, output="ok")
        evaluation = ToolCallEvaluation(
            tool_call=tc,
            name_correct=True,
            args_present=0.5,
            success=True,
        )
        # 0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 1.0 = 0.4 + 0.15 + 0.3 = 0.85
        assert evaluation.score == pytest.approx(0.85)

    def test_score_failed_call(self) -> None:
        """Score penalizes failed calls."""
        tc = ToolCall(name="search", input={"q": "test"}, output="", error="timeout")
        evaluation = ToolCallEvaluation(
            tool_call=tc,
            name_correct=True,
            args_present=1.0,
            success=False,
        )
        # 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 0.0 = 0.4 + 0.3 + 0.0 = 0.7
        assert evaluation.score == pytest.approx(0.7)

    def test_frozen_immutability(self) -> None:
        """ToolCallEvaluation is immutable."""
        tc = ToolCall(name="test", input={}, output="ok")
        evaluation = ToolCallEvaluation(
            tool_call=tc,
            name_correct=True,
            args_present=1.0,
            success=True,
        )
        with pytest.raises(AttributeError):
            evaluation.name_correct = False  # type: ignore[misc]


# ============================================================================
# ToolUseMetrics Tests
# ============================================================================


class TestToolUseMetrics:
    """Tests for ToolUseMetrics dataclass."""

    def test_basic_metrics(self) -> None:
        """Basic metrics creation."""
        metrics = ToolUseMetrics(
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            success_rate=0.8,
            arg_correctness=0.9,
        )
        assert metrics.total_calls == 10
        assert metrics.success_rate == 0.8

    def test_selection_metrics_none_by_default(self) -> None:
        """Selection metrics are None without expected tools."""
        metrics = ToolUseMetrics(
            total_calls=5,
            successful_calls=5,
            failed_calls=0,
            success_rate=1.0,
            arg_correctness=1.0,
        )
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1 is None

    def test_selection_metrics_with_expected(self) -> None:
        """Selection metrics populated with expected tools."""
        metrics = ToolUseMetrics(
            total_calls=3,
            successful_calls=3,
            failed_calls=0,
            success_rate=1.0,
            arg_correctness=1.0,
            precision=0.67,
            recall=1.0,
            f1=0.8,
            correct_calls=2,
            unnecessary_calls=1,
            missing_calls=0,
        )
        assert metrics.precision == pytest.approx(0.67)
        assert metrics.recall == 1.0
        assert metrics.f1 == pytest.approx(0.8)

    def test_summary_without_selection(self) -> None:
        """Summary without selection metrics."""
        metrics = ToolUseMetrics(
            total_calls=5,
            successful_calls=4,
            failed_calls=1,
            success_rate=0.8,
            arg_correctness=0.9,
        )
        summary = metrics.summary()
        assert "Total calls: 5" in summary
        assert "Success rate: 80.0%" in summary
        assert "Precision" not in summary

    def test_summary_with_selection(self) -> None:
        """Summary with selection metrics."""
        metrics = ToolUseMetrics(
            total_calls=3,
            successful_calls=3,
            failed_calls=0,
            success_rate=1.0,
            arg_correctness=1.0,
            precision=1.0,
            recall=0.5,
            f1=0.667,
            correct_calls=1,
            unnecessary_calls=2,
            missing_calls=1,
        )
        summary = metrics.summary()
        assert "Precision: 100.0%" in summary
        assert "Recall: 50.0%" in summary
        assert "Correct: 1" in summary
        assert "Missing: 1" in summary


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestToolSuccessRate:
    """Tests for tool_success_rate helper."""

    def test_all_successful(self) -> None:
        """100% success rate when all calls succeed."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call 1",
                    tool_call=ToolCall(name="t1", input={}, output="ok"),
                ),
                AgentStep(
                    step_type="action",
                    content="call 2",
                    tool_call=ToolCall(name="t2", input={}, output="ok"),
                ),
            ],
        )
        assert tool_success_rate(response) == 1.0

    def test_all_failed(self) -> None:
        """0% success rate when all calls fail."""
        response = AgentResponse(
            answer="failed",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call 1",
                    tool_call=ToolCall(name="t1", input={}, output="", error="e1"),
                ),
                AgentStep(
                    step_type="action",
                    content="call 2",
                    tool_call=ToolCall(name="t2", input={}, output="", error="e2"),
                ),
            ],
        )
        assert tool_success_rate(response) == 0.0

    def test_mixed_results(self) -> None:
        """Correct rate for mixed success/failure."""
        response = AgentResponse(
            answer="partial",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call 1",
                    tool_call=ToolCall(name="t1", input={}, output="ok"),
                ),
                AgentStep(
                    step_type="action",
                    content="call 2",
                    tool_call=ToolCall(name="t2", input={}, output="", error="err"),
                ),
            ],
        )
        assert tool_success_rate(response) == 0.5

    def test_no_tool_calls(self) -> None:
        """Returns 1.0 when no tool calls."""
        response = AgentResponse(
            answer="simple",
            steps=[AgentStep(step_type="thought", content="thinking")],
        )
        assert tool_success_rate(response) == 1.0


class TestToolErrorRate:
    """Tests for tool_error_rate helper."""

    def test_complement_of_success_rate(self) -> None:
        """Error rate is 1 - success rate."""
        response = AgentResponse(
            answer="partial",
            steps=[
                AgentStep(
                    step_type="action",
                    content="ok",
                    tool_call=ToolCall(name="t1", input={}, output="ok"),
                ),
                AgentStep(
                    step_type="action",
                    content="fail",
                    tool_call=ToolCall(name="t2", input={}, output="", error="e"),
                ),
                AgentStep(
                    step_type="action",
                    content="fail2",
                    tool_call=ToolCall(name="t3", input={}, output="", error="e"),
                ),
            ],
        )
        # 1 success out of 3
        assert tool_error_rate(response) == pytest.approx(2 / 3)


class TestArgPresenceRate:
    """Tests for arg_presence_rate helper."""

    def test_all_args_present(self) -> None:
        """Returns 1.0 when all required args present."""
        expected = ExpectedToolCall(
            name="search",
            required_args=frozenset({"query", "limit"}),
        )
        tc = ToolCall(
            name="search",
            input={"query": "test", "limit": 10, "extra": "ignored"},
            output="ok",
        )
        assert arg_presence_rate(tc, expected) == 1.0

    def test_no_args_present(self) -> None:
        """Returns 0.0 when no required args present."""
        expected = ExpectedToolCall(
            name="search",
            required_args=frozenset({"query", "limit"}),
        )
        tc = ToolCall(name="search", input={"other": "value"}, output="ok")
        assert arg_presence_rate(tc, expected) == 0.0

    def test_partial_args_present(self) -> None:
        """Returns proportion of required args present."""
        expected = ExpectedToolCall(
            name="search",
            required_args=frozenset({"query", "limit", "offset"}),
        )
        tc = ToolCall(name="search", input={"query": "test"}, output="ok")
        assert arg_presence_rate(tc, expected) == pytest.approx(1 / 3)

    def test_no_required_args(self) -> None:
        """Returns 1.0 when no required args defined."""
        expected = ExpectedToolCall(name="simple_tool")
        tc = ToolCall(name="simple_tool", input={}, output="ok")
        assert arg_presence_rate(tc, expected) == 1.0


# ============================================================================
# evaluate_tool_use Tests
# ============================================================================


class TestEvaluateToolUse:
    """Tests for evaluate_tool_use function."""

    def test_no_tool_calls(self) -> None:
        """Empty response has default metrics."""
        response = AgentResponse(answer="simple", steps=[])
        metrics = evaluate_tool_use(response)

        assert metrics.total_calls == 0
        assert metrics.success_rate == 1.0
        assert metrics.arg_correctness == 1.0
        assert metrics.precision is None
        assert metrics.evaluations == []

    def test_no_tool_calls_with_expected(self) -> None:
        """Empty response with expected tools has 0 recall."""
        response = AgentResponse(answer="simple", steps=[])
        expected = [ExpectedToolCall(name="required_tool")]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.missing_calls == 1

    def test_all_successful(self) -> None:
        """All successful calls."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call",
                    tool_call=ToolCall(name="t1", input={"a": 1}, output="ok"),
                ),
            ],
        )
        metrics = evaluate_tool_use(response)

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.success_rate == 1.0

    def test_all_failed(self) -> None:
        """All failed calls."""
        response = AgentResponse(
            answer="failed",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call",
                    tool_call=ToolCall(name="t1", input={}, output="", error="err"),
                ),
            ],
        )
        metrics = evaluate_tool_use(response)

        assert metrics.success_rate == 0.0
        assert metrics.failed_calls == 1

    def test_mixed_results(self) -> None:
        """Mix of successful and failed calls."""
        response = AgentResponse(
            answer="partial",
            steps=[
                AgentStep(
                    step_type="action",
                    content="ok",
                    tool_call=ToolCall(name="t1", input={}, output="ok"),
                ),
                AgentStep(
                    step_type="action",
                    content="fail",
                    tool_call=ToolCall(name="t2", input={}, output="", error="e"),
                ),
            ],
        )
        metrics = evaluate_tool_use(response)

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.success_rate == 0.5

    def test_with_expected_tools_all_match(self) -> None:
        """All actual tools match expected."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="search",
                    tool_call=ToolCall(name="search", input={"q": "test"}, output="ok"),
                ),
            ],
        )
        expected = [ExpectedToolCall(name="search", required_args=frozenset({"q"}))]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.correct_calls == 1
        assert metrics.unnecessary_calls == 0
        assert metrics.missing_calls == 0

    def test_with_expected_tools_extra_calls(self) -> None:
        """More actual calls than expected."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call 1",
                    tool_call=ToolCall(name="search", input={}, output="ok"),
                ),
                AgentStep(
                    step_type="action",
                    content="call 2",
                    tool_call=ToolCall(name="extra", input={}, output="ok"),
                ),
            ],
        )
        expected = [ExpectedToolCall(name="search")]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        # 1 correct out of 2 actual
        assert metrics.precision == 0.5
        # 1 correct out of 1 expected
        assert metrics.recall == 1.0
        assert metrics.f1 == pytest.approx(2 / 3)
        assert metrics.unnecessary_calls == 1

    def test_with_expected_tools_missing_calls(self) -> None:
        """Fewer actual calls than expected."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call",
                    tool_call=ToolCall(name="search", input={}, output="ok"),
                ),
            ],
        )
        expected = [
            ExpectedToolCall(name="search"),
            ExpectedToolCall(name="calculate"),
        ]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        # 1 correct out of 1 actual
        assert metrics.precision == 1.0
        # 1 correct out of 2 expected
        assert metrics.recall == 0.5
        assert metrics.f1 == pytest.approx(2 / 3)
        assert metrics.missing_calls == 1

    def test_arg_validation(self) -> None:
        """Argument validation affects arg_correctness."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call",
                    tool_call=ToolCall(
                        name="search",
                        input={"query": "test"},  # Missing 'limit'
                        output="ok",
                    ),
                ),
            ],
        )
        expected = [
            ExpectedToolCall(
                name="search",
                required_args=frozenset({"query", "limit"}),
            ),
        ]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        # Only 1 of 2 required args present
        assert metrics.arg_correctness == 0.5
        assert len(metrics.evaluations) == 1
        assert metrics.evaluations[0].args_present == 0.5

    def test_evaluations_populated(self) -> None:
        """Per-call evaluations are populated."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="c1",
                    tool_call=ToolCall(name="t1", input={}, output="ok"),
                ),
                AgentStep(
                    step_type="action",
                    content="c2",
                    tool_call=ToolCall(name="t2", input={}, output="", error="e"),
                ),
            ],
        )
        metrics = evaluate_tool_use(response)

        assert len(metrics.evaluations) == 2
        assert metrics.evaluations[0].success is True
        assert metrics.evaluations[1].success is False

    def test_duplicate_expected_tools(self) -> None:
        """Each expected tool can only match once."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="c1",
                    tool_call=ToolCall(name="search", input={}, output="r1"),
                ),
                AgentStep(
                    step_type="action",
                    content="c2",
                    tool_call=ToolCall(name="search", input={}, output="r2"),
                ),
            ],
        )
        # Only one search expected
        expected = [ExpectedToolCall(name="search")]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        # First search matches, second is unnecessary
        assert metrics.correct_calls == 1
        assert metrics.unnecessary_calls == 1
        assert metrics.precision == 0.5

    def test_multiple_expected_same_name(self) -> None:
        """Multiple expected tools with same name."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="c1",
                    tool_call=ToolCall(name="search", input={}, output="r1"),
                ),
                AgentStep(
                    step_type="action",
                    content="c2",
                    tool_call=ToolCall(name="search", input={}, output="r2"),
                ),
            ],
        )
        # Two searches expected
        expected = [
            ExpectedToolCall(name="search"),
            ExpectedToolCall(name="search"),
        ]
        metrics = evaluate_tool_use(response, expected_tools=expected)

        # Both searches match
        assert metrics.correct_calls == 2
        assert metrics.unnecessary_calls == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_empty_expected_tools_list(self) -> None:
        """Empty expected_tools list (different from None)."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="call",
                    tool_call=ToolCall(name="t1", input={}, output="ok"),
                ),
            ],
        )
        metrics = evaluate_tool_use(response, expected_tools=[])

        # With empty expected list, precision should handle division
        assert metrics.precision == 0.0  # 0 correct / 1 actual (no expected to match)
        assert metrics.recall == 1.0  # 0 correct / 0 expected (vacuously true)
        assert metrics.unnecessary_calls == 1

    def test_error_recovery_pattern(self) -> None:
        """Agent retries after error."""
        response = AgentResponse(
            answer="finally done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="first try",
                    tool_call=ToolCall(name="api", input={}, output="", error="timeout"),
                ),
                AgentStep(
                    step_type="action",
                    content="retry",
                    tool_call=ToolCall(name="api", input={}, output="success"),
                ),
            ],
        )
        metrics = evaluate_tool_use(response)

        assert metrics.total_calls == 2
        assert metrics.success_rate == 0.5
        # Both evaluations captured
        assert metrics.evaluations[0].success is False
        assert metrics.evaluations[1].success is True
