"""Unit tests for agent types and protocols."""

from __future__ import annotations

import pytest

from ragnarok_ai.agents import AgentProtocol, AgentResponse, AgentStep, ToolCall

# ============================================================================
# ToolCall Tests
# ============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_success_property_true(self) -> None:
        """success is True when no error."""
        tool_call = ToolCall(
            name="search",
            input={"query": "test"},
            output="results",
        )
        assert tool_call.success is True

    def test_success_property_false(self) -> None:
        """success is False when error present."""
        tool_call = ToolCall(
            name="search",
            input={"query": "test"},
            output="",
            error="Connection timeout",
        )
        assert tool_call.success is False

    def test_frozen_immutability(self) -> None:
        """ToolCall is immutable (frozen=True)."""
        tool_call = ToolCall(
            name="search",
            input={"query": "test"},
            output="results",
        )
        with pytest.raises(AttributeError):
            tool_call.name = "different"  # type: ignore[misc]

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        tool_call = ToolCall(
            name="tool",
            input={},
            output="out",
        )
        assert tool_call.error is None
        assert tool_call.latency_ms == 0.0

    def test_with_latency(self) -> None:
        """latency_ms is recorded correctly."""
        tool_call = ToolCall(
            name="slow_tool",
            input={"x": 1},
            output="done",
            latency_ms=1500.5,
        )
        assert tool_call.latency_ms == 1500.5


# ============================================================================
# AgentStep Tests
# ============================================================================


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_thought_step(self) -> None:
        """Create a thought step."""
        step = AgentStep(
            step_type="thought",
            content="I need to analyze this problem.",
        )
        assert step.step_type == "thought"
        assert step.content == "I need to analyze this problem."
        assert step.tool_call is None

    def test_action_step_with_tool(self) -> None:
        """Create an action step with tool call."""
        tool_call = ToolCall(
            name="calculator",
            input={"expression": "2+2"},
            output="4",
        )
        step = AgentStep(
            step_type="action",
            content="Calculating 2+2",
            tool_call=tool_call,
        )
        assert step.step_type == "action"
        assert step.tool_call is not None
        assert step.tool_call.name == "calculator"

    def test_observation_step(self) -> None:
        """Create an observation step."""
        step = AgentStep(
            step_type="observation",
            content="The search returned 5 results.",
        )
        assert step.step_type == "observation"

    def test_final_answer_step(self) -> None:
        """Create a final answer step."""
        step = AgentStep(
            step_type="final_answer",
            content="The answer is 42.",
        )
        assert step.step_type == "final_answer"

    def test_frozen_immutability(self) -> None:
        """AgentStep is immutable (frozen=True)."""
        step = AgentStep(
            step_type="thought",
            content="test",
        )
        with pytest.raises(AttributeError):
            step.content = "modified"  # type: ignore[misc]

    def test_with_metadata(self) -> None:
        """Step can have custom metadata."""
        step = AgentStep(
            step_type="action",
            content="test",
            metadata={"token_count": 150, "model": "gpt-4"},
        )
        assert step.metadata["token_count"] == 150
        assert step.metadata["model"] == "gpt-4"

    def test_with_latency(self) -> None:
        """Step records latency."""
        step = AgentStep(
            step_type="thought",
            content="thinking...",
            latency_ms=50.5,
        )
        assert step.latency_ms == 50.5


# ============================================================================
# AgentResponse Tests
# ============================================================================


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    @pytest.fixture
    def sample_response(self) -> AgentResponse:
        """Create a sample agent response with multiple steps."""
        return AgentResponse(
            answer="Paris is the capital of France.",
            steps=[
                AgentStep(step_type="thought", content="Let me think about this."),
                AgentStep(
                    step_type="action",
                    content="Searching...",
                    tool_call=ToolCall(
                        name="search",
                        input={"query": "capital of France"},
                        output="Paris",
                        latency_ms=100.0,
                    ),
                ),
                AgentStep(step_type="observation", content="Found: Paris"),
                AgentStep(step_type="final_answer", content="Paris"),
            ],
            total_latency_ms=250.0,
        )

    def test_tool_calls_property(self, sample_response: AgentResponse) -> None:
        """tool_calls extracts all tool calls from steps."""
        tool_calls = sample_response.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"

    def test_num_steps(self, sample_response: AgentResponse) -> None:
        """num_steps returns correct count."""
        assert sample_response.num_steps == 4

    def test_num_tool_calls(self, sample_response: AgentResponse) -> None:
        """num_tool_calls returns correct count."""
        assert sample_response.num_tool_calls == 1

    def test_thoughts_extraction(self, sample_response: AgentResponse) -> None:
        """thoughts extracts all thought step contents."""
        thoughts = sample_response.thoughts
        assert len(thoughts) == 1
        assert thoughts[0] == "Let me think about this."

    def test_reasoning_trace(self, sample_response: AgentResponse) -> None:
        """reasoning_trace formats execution flow."""
        trace = sample_response.reasoning_trace
        assert "[1] THOUGHT" in trace
        assert "[2] ACTION" in trace
        assert "search" in trace
        assert "[3] OBSERVATION" in trace
        assert "[4] FINAL_ANSWER" in trace

    def test_empty_response(self) -> None:
        """Empty response has sensible defaults."""
        response = AgentResponse(answer="Simple answer")
        assert response.num_steps == 0
        assert response.tool_calls == []
        assert response.thoughts == []
        assert response.total_latency_ms == 0.0

    def test_to_dict(self, sample_response: AgentResponse) -> None:
        """to_dict returns JSON-serializable dict."""
        d = sample_response.to_dict()
        assert d["answer"] == "Paris is the capital of France."
        assert len(d["steps"]) == 4
        assert d["steps"][0]["step_type"] == "thought"
        assert d["steps"][1]["tool_call"]["name"] == "search"
        assert d["steps"][1]["tool_call"]["input"] == {"query": "capital of France"}
        assert d["total_latency_ms"] == 250.0

    def test_to_dict_without_tool_calls(self) -> None:
        """to_dict handles steps without tool calls."""
        response = AgentResponse(
            answer="test",
            steps=[AgentStep(step_type="thought", content="thinking")],
        )
        d = response.to_dict()
        assert d["steps"][0]["tool_call"] is None

    def test_multiple_tool_calls(self) -> None:
        """Response with multiple tool calls."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(
                    step_type="action",
                    content="First tool",
                    tool_call=ToolCall(name="tool1", input={}, output="r1"),
                ),
                AgentStep(step_type="observation", content="Got r1"),
                AgentStep(
                    step_type="action",
                    content="Second tool",
                    tool_call=ToolCall(name="tool2", input={}, output="r2"),
                ),
                AgentStep(step_type="observation", content="Got r2"),
            ],
        )
        assert response.num_tool_calls == 2
        assert response.tool_calls[0].name == "tool1"
        assert response.tool_calls[1].name == "tool2"

    def test_with_metadata(self) -> None:
        """Response can have custom metadata."""
        response = AgentResponse(
            answer="test",
            metadata={"model": "gpt-4", "temperature": 0.7},
        )
        assert response.metadata["model"] == "gpt-4"


# ============================================================================
# AgentProtocol Tests
# ============================================================================


class TestAgentProtocol:
    """Tests for AgentProtocol."""

    def test_duck_typing_works(self) -> None:
        """Class implementing query() matches protocol."""

        class MyAgent:
            async def query(self, question: str) -> AgentResponse:
                return AgentResponse(answer=f"Answer to: {question}")

        agent = MyAgent()
        # Duck typing check via isinstance
        assert isinstance(agent, AgentProtocol)

    def test_isinstance_check_fails_without_query(self) -> None:
        """Class without query() doesn't match protocol."""

        class NotAnAgent:
            async def run(self, _question: str) -> str:
                return "not an agent"

        not_agent = NotAnAgent()
        assert not isinstance(not_agent, AgentProtocol)

    def test_isinstance_check_fails_wrong_signature(self) -> None:
        """Class with wrong signature doesn't match protocol."""

        class WrongAgent:
            async def query(self) -> str:  # Missing question parameter
                return "wrong"

        # Note: runtime_checkable only checks method existence, not signature
        # This is a limitation of Python's Protocol
        wrong = WrongAgent()
        # This will actually pass isinstance check (runtime_checkable limitation)
        # Full signature checking requires static type checkers (mypy)
        assert isinstance(wrong, AgentProtocol)

    @pytest.mark.asyncio
    async def test_protocol_usage(self) -> None:
        """Protocol can be used as type hint."""

        class SimpleAgent:
            async def query(self, _question: str) -> AgentResponse:
                return AgentResponse(
                    answer="42",
                    steps=[
                        AgentStep(step_type="final_answer", content="42"),
                    ],
                )

        async def use_agent(agent: AgentProtocol) -> str:
            response = await agent.query("What is the answer?")
            return response.answer

        agent = SimpleAgent()
        result = await use_agent(agent)
        assert result == "42"
