"""Unit tests for ReAct and CoT adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragnarok_ai.adapters.agents import ChainOfThoughtAdapter, ReActAdapter, ReActParser

# =============================================================================
# ReActParser Tests
# =============================================================================


class TestReActParser:
    """Tests for ReActParser."""

    def test_parse_simple_trace(self) -> None:
        """Parse a simple ReAct trace."""
        parser = ReActParser()
        text = """Thought: I need to search for information
Action: search
Action Input: {"query": "test"}
Observation: Found some results
Final Answer: The answer is 42"""

        steps = parser.parse(text)

        assert len(steps) == 4
        assert steps[0].step_type == "thought"
        assert "search for information" in steps[0].content
        assert steps[1].step_type == "action"
        assert steps[1].tool_call is not None
        assert steps[1].tool_call.name == "search"
        assert steps[2].step_type == "observation"
        assert steps[3].step_type == "final_answer"
        assert "42" in steps[3].content

    def test_parse_multiple_actions(self) -> None:
        """Parse trace with multiple action cycles."""
        parser = ReActParser()
        text = """Thought: First I need to search
Action: search
Action Input: {"q": "first"}
Observation: Result 1
Thought: Now I need more info
Action: search
Action Input: {"q": "second"}
Observation: Result 2
Final Answer: Combined answer"""

        steps = parser.parse(text)

        # 2 thoughts + 2 actions + 2 observations + 1 final = 7
        assert len(steps) == 7
        action_steps = [s for s in steps if s.step_type == "action"]
        assert len(action_steps) == 2
        assert action_steps[0].tool_call.input == {"q": "first"}
        assert action_steps[1].tool_call.input == {"q": "second"}

    def test_parse_with_valid_json_input(self) -> None:
        """Parse action with valid JSON input."""
        parser = ReActParser()
        text = """Action: api_call
Action Input: {"endpoint": "/users", "method": "GET", "params": {"id": 123}}
Observation: Success"""

        steps = parser.parse(text)

        assert len(steps) == 2
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.input["endpoint"] == "/users"
        assert steps[0].tool_call.input["method"] == "GET"
        assert steps[0].tool_call.input["params"] == {"id": 123}

    def test_parse_with_single_quote_json(self) -> None:
        """Parse action with Python-style single quote dict."""
        parser = ReActParser()
        text = """Action: search
Action Input: {'query': 'test value', 'limit': 10}
Observation: Done"""

        steps = parser.parse(text)

        assert len(steps) == 2
        assert steps[0].tool_call is not None
        # Should handle single quotes
        input_dict = steps[0].tool_call.input
        assert "query" in input_dict or "raw" in input_dict

    def test_parse_with_simple_string_input(self) -> None:
        """Parse action with plain string input."""
        parser = ReActParser()
        text = """Action: search
Action Input: just a simple query string
Observation: Results"""

        steps = parser.parse(text)

        assert len(steps) == 2
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.input.get("raw") == "just a simple query string"

    def test_parse_malformed_json_input(self) -> None:
        """Parse action with malformed/incomplete JSON."""
        parser = ReActParser()
        text = """Action: search
Action Input: {query: test, missing quotes
Observation: Still works"""

        steps = parser.parse(text)

        assert len(steps) == 2
        assert steps[0].tool_call is not None
        # Should fallback gracefully
        assert "raw" in steps[0].tool_call.input or "query" in steps[0].tool_call.input

    def test_parse_multiline_observation(self) -> None:
        """Parse observation that spans multiple lines."""
        parser = ReActParser()
        text = """Thought: Searching
Action: search
Action Input: {"q": "test"}
Observation: This is a long observation
that spans multiple lines
with various content
including numbers 123 and symbols @#$
Final Answer: Done"""

        steps = parser.parse(text)

        observation_steps = [s for s in steps if s.step_type == "observation"]
        assert len(observation_steps) == 1
        obs_content = observation_steps[0].content
        assert "multiple lines" in obs_content
        assert "123" in obs_content

    def test_parse_multiline_json_input(self) -> None:
        """Parse action input JSON that spans multiple lines."""
        parser = ReActParser()
        text = """Action: create
Action Input: {
    "name": "test",
    "value": 123
}
Observation: Created"""

        steps = parser.parse(text)

        assert len(steps) == 2
        assert steps[0].tool_call is not None
        # Should combine multiline JSON
        input_dict = steps[0].tool_call.input
        assert input_dict.get("name") == "test" or "raw" in input_dict

    def test_custom_prefixes(self) -> None:
        """Parse with custom prefixes."""
        parser = ReActParser(
            thought_prefix="Thinking:",
            action_prefix="Do:",
            action_input_prefix="With:",
            observation_prefix="Result:",
        )
        text = """Thinking: I should search
Do: lookup
With: {"term": "test"}
Result: Found it"""

        steps = parser.parse(text)

        assert len(steps) == 3
        assert steps[0].step_type == "thought"
        assert steps[1].step_type == "action"
        assert steps[1].tool_call.name == "lookup"
        assert steps[2].step_type == "observation"

    def test_parse_to_response(self) -> None:
        """parse_to_response wraps in AgentResponse."""
        parser = ReActParser()
        text = """Thought: Thinking
Action: do
Action Input: {}
Observation: Done
Final Answer: The result is 42"""

        response = parser.parse_to_response(text)

        assert response.answer == "The result is 42"
        assert len(response.steps) == 4

    def test_parse_to_response_with_explicit_answer(self) -> None:
        """parse_to_response uses explicit answer if provided."""
        parser = ReActParser()
        text = """Thought: Thinking
Final Answer: Wrong answer"""

        response = parser.parse_to_response(text, answer="Correct answer")

        assert response.answer == "Correct answer"

    def test_parse_empty_input(self) -> None:
        """Parse empty input returns empty list."""
        parser = ReActParser()

        assert parser.parse("") == []
        assert parser.parse("   ") == []
        assert parser.parse("\n\n") == []

    def test_parse_no_final_answer(self) -> None:
        """Parse trace without Final Answer extracts from observation."""
        parser = ReActParser()
        text = """Thought: Let me search
Action: search
Action Input: {"q": "test"}
Observation: The answer is Paris"""

        response = parser.parse_to_response(text)

        assert "Paris" in response.answer

    def test_tool_call_output_updated(self) -> None:
        """Observation updates the corresponding tool call output."""
        parser = ReActParser()
        text = """Action: search
Action Input: {"q": "test"}
Observation: Search results here"""

        steps = parser.parse(text)

        assert len(steps) == 2
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.output == "Search results here"

    def test_parse_key_value_input(self) -> None:
        """Parse action input with key=value format."""
        parser = ReActParser()
        text = """Action: search
Action Input: query=test, limit=10
Observation: Done"""

        steps = parser.parse(text)

        assert len(steps) == 2
        input_dict = steps[0].tool_call.input
        # Should extract key-value pairs
        assert "query" in input_dict or "raw" in input_dict


# =============================================================================
# ReActAdapter Tests
# =============================================================================


class TestReActAdapter:
    """Tests for ReActAdapter."""

    @pytest.mark.asyncio
    async def test_async_agent(self) -> None:
        """Wrap an async agent."""

        async def async_agent(question: str) -> str:
            return f"""Thought: Processing {question}
Action: answer
Action Input: {{"response": "42"}}
Observation: Answered
Final Answer: 42"""

        adapter = ReActAdapter(async_agent)
        response = await adapter.query("What is the answer?")

        assert response.answer == "42"
        assert len(response.steps) > 0
        assert response.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_sync_agent(self) -> None:
        """Wrap a sync agent."""

        def sync_agent(question: str) -> str:
            return f"""Thought: Got {question}
Final Answer: Sync response"""

        adapter = ReActAdapter(sync_agent)
        response = await adapter.query("Test?")

        assert response.answer == "Sync response"

    @pytest.mark.asyncio
    async def test_custom_parser(self) -> None:
        """Use custom parser with adapter."""

        async def agent(_: str) -> str:
            return """Thinking: Custom format
Do: action
With: {}
Result: Done"""

        custom_parser = ReActParser(
            thought_prefix="Thinking:",
            action_prefix="Do:",
            action_input_prefix="With:",
            observation_prefix="Result:",
        )

        adapter = ReActAdapter(agent, parser=custom_parser)
        response = await adapter.query("Test")

        assert len(response.steps) == 3

    @pytest.mark.asyncio
    async def test_custom_answer_extractor(self) -> None:
        """Use custom answer extractor."""

        async def agent(_: str) -> str:
            return """Some output
ANSWER: Custom extracted"""

        def extract(output: str) -> str:
            for line in output.split("\n"):
                if line.startswith("ANSWER:"):
                    return line[7:].strip()
            return ""

        adapter = ReActAdapter(agent, extract_answer=extract)
        response = await adapter.query("Test")

        assert response.answer == "Custom extracted"

    @pytest.mark.asyncio
    async def test_metadata_populated(self) -> None:
        """Response metadata is populated."""

        async def agent(_: str) -> str:
            return "Final Answer: Done"

        adapter = ReActAdapter(agent)
        response = await adapter.query("Test")

        assert response.metadata["adapter"] == "react"
        assert "raw_output_length" in response.metadata


# =============================================================================
# ChainOfThoughtAdapter Tests
# =============================================================================


class TestChainOfThoughtAdapter:
    """Tests for ChainOfThoughtAdapter."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        llm = AsyncMock()
        llm.is_local = True
        return llm

    @pytest.mark.asyncio
    async def test_basic_cot(self, mock_llm: AsyncMock) -> None:
        """Basic CoT reasoning."""
        mock_llm.generate.return_value = """First, I need to understand the problem.

Then, I'll break it down into parts.

Finally, the answer is 42."""

        adapter = ChainOfThoughtAdapter(mock_llm)
        response = await adapter.query("What is the answer?")

        assert "42" in response.answer
        assert len(response.steps) >= 2

    @pytest.mark.asyncio
    async def test_step_separation(self, mock_llm: AsyncMock) -> None:
        """Steps are separated correctly."""
        mock_llm.generate.return_value = """Step 1: Analyze the input.

Step 2: Process the data.

Step 3: Therefore, the answer is X."""

        adapter = ChainOfThoughtAdapter(mock_llm)
        response = await adapter.query("Process this")

        thought_steps = [s for s in response.steps if s.step_type == "thought"]
        assert len(thought_steps) >= 2

    @pytest.mark.asyncio
    async def test_custom_cot_prompt(self, mock_llm: AsyncMock) -> None:
        """Custom CoT prompt is used."""
        mock_llm.generate.return_value = "The answer is 5."

        adapter = ChainOfThoughtAdapter(
            mock_llm,
            cot_prompt="Think carefully step by step:",
        )
        await adapter.query("2+3=?")

        # Check that prompt was built correctly
        call_args = mock_llm.generate.call_args[0][0]
        assert "Think carefully step by step:" in call_args

    @pytest.mark.asyncio
    async def test_custom_step_separator(self, mock_llm: AsyncMock) -> None:
        """Custom step separator works."""
        mock_llm.generate.return_value = "First---Second---Third"

        adapter = ChainOfThoughtAdapter(mock_llm, step_separator="---")
        response = await adapter.query("Test")

        assert len(response.steps) == 3

    @pytest.mark.asyncio
    async def test_answer_prefix(self, mock_llm: AsyncMock) -> None:
        """Answer prefix extracts explicit answer."""
        mock_llm.generate.return_value = """Let me think about this.

I'll consider the options.

ANSWER: The final result is 100."""

        adapter = ChainOfThoughtAdapter(mock_llm, answer_prefix="ANSWER:")
        response = await adapter.query("Calculate")

        assert "100" in response.answer

    @pytest.mark.asyncio
    async def test_metadata_populated(self, mock_llm: AsyncMock) -> None:
        """Response metadata is populated."""
        mock_llm.generate.return_value = "Simple answer"

        adapter = ChainOfThoughtAdapter(mock_llm)
        response = await adapter.query("Test")

        assert response.metadata["adapter"] == "chain_of_thought"
        assert "cot_prompt" in response.metadata
        assert response.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_numbered_steps(self, mock_llm: AsyncMock) -> None:
        """Parse numbered step format."""
        mock_llm.generate.return_value = """1. First step of reasoning
2. Second step with more detail
3. Third step concluding
4. The answer is 42"""

        adapter = ChainOfThoughtAdapter(mock_llm, step_separator="\n\n")
        response = await adapter.query("Test")

        # Should detect numbered format
        assert len(response.steps) >= 1

    @pytest.mark.asyncio
    async def test_empty_response(self, mock_llm: AsyncMock) -> None:
        """Handle empty LLM response."""
        mock_llm.generate.return_value = ""

        adapter = ChainOfThoughtAdapter(mock_llm)
        response = await adapter.query("Test")

        assert response.answer == ""
        assert response.steps == []

    @pytest.mark.asyncio
    async def test_conclusion_detection(self, mock_llm: AsyncMock) -> None:
        """Detect conclusion markers."""
        mock_llm.generate.return_value = """First I analyze.

Then I process.

Therefore, the answer is 7."""

        adapter = ChainOfThoughtAdapter(mock_llm)
        response = await adapter.query("Test")

        # Should detect "therefore" as conclusion
        final_steps = [s for s in response.steps if s.step_type == "final_answer"]
        assert len(final_steps) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for adapters."""

    @pytest.mark.asyncio
    async def test_react_adapter_implements_protocol(self) -> None:
        """ReActAdapter response works with AgentProtocol."""

        async def agent(_: str) -> str:
            return "Final Answer: Test"

        adapter = ReActAdapter(agent)

        # Check protocol compatibility
        assert hasattr(adapter, "query")
        response = await adapter.query("Test")
        assert hasattr(response, "answer")
        assert hasattr(response, "steps")

    @pytest.mark.asyncio
    async def test_cot_adapter_with_evaluator(self) -> None:
        """CoT adapter works with efficiency evaluator."""
        from ragnarok_ai.agents.evaluators import ReasoningEfficiencyEvaluator

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """Step 1.

Step 2.

Step 3."""

        adapter = ChainOfThoughtAdapter(mock_llm)
        response = await adapter.query("Test")

        evaluator = ReasoningEfficiencyEvaluator()
        result = evaluator.evaluate(response, optimal_steps=3)

        assert result.actual_steps == len(response.steps)
