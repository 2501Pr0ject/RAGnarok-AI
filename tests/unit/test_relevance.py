"""Tests for Relevance evaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.evaluators.relevance import (
    RelevanceEvaluator,
    RelevanceResult,
)


class TestRelevanceEvaluatorInit:
    """Tests for RelevanceEvaluator initialization."""

    def test_init_with_llm(self) -> None:
        """Evaluator stores the LLM reference."""
        mock_llm = AsyncMock()
        evaluator = RelevanceEvaluator(llm=mock_llm)

        assert evaluator.llm is mock_llm


class TestRelevanceEvaluatorEvaluate:
    """Tests for RelevanceEvaluator.evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_fully_relevant(self) -> None:
        """Returns 1.0 when answer fully addresses the question."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """{
            "score": 1.0,
            "reasoning": "The answer directly addresses the question about the capital of France.",
            "aspects_covered": ["capital of France"],
            "aspects_missing": []
        }"""

        evaluator = RelevanceEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital of France.",
            query="What is the capital of France?",
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_not_relevant(self) -> None:
        """Returns 0.0 when answer does not address the question."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """{
            "score": 0.0,
            "reasoning": "The answer talks about weather, not the capital.",
            "aspects_covered": [],
            "aspects_missing": ["capital of France"]
        }"""

        evaluator = RelevanceEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="The weather is nice today.",
            query="What is the capital of France?",
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_partial_relevance(self) -> None:
        """Returns partial score when some aspects are covered."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """{
            "score": 0.5,
            "reasoning": "The answer mentions Paris but doesn't fully explain it's the capital.",
            "aspects_covered": ["Paris mentioned"],
            "aspects_missing": ["explanation of capital status"]
        }"""

        evaluator = RelevanceEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is a beautiful city.",
            query="What is the capital of France and why is it significant?",
        )

        assert score == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self) -> None:
        """Returns 0.0 for empty response."""
        mock_llm = AsyncMock()

        evaluator = RelevanceEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="",
            query="What is the capital of France?",
        )

        assert score == 0.0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_empty_query(self) -> None:
        """Returns 1.0 for empty query."""
        mock_llm = AsyncMock()

        evaluator = RelevanceEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital.",
            query="",
        )

        assert score == 1.0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_query_required(self) -> None:
        """Raises EvaluationError when query is None."""
        mock_llm = AsyncMock()

        evaluator = RelevanceEvaluator(llm=mock_llm)

        with pytest.raises(EvaluationError, match="Query is required"):
            await evaluator.evaluate(
                response="Paris is the capital.",
                query=None,
            )

    @pytest.mark.asyncio
    async def test_evaluate_context_parameter_ignored(self) -> None:
        """Context parameter is accepted but not used."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """{
            "score": 1.0,
            "reasoning": "Fully relevant.",
            "aspects_covered": ["capital"],
            "aspects_missing": []
        }"""

        evaluator = RelevanceEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital.",
            context="This context is not used.",
            query="What is the capital?",
        )

        assert score == 1.0
        # Context should not appear in the prompt
        call_args = mock_llm.generate.call_args[0][0]
        assert "This context is not used" not in call_args


class TestRelevanceEvaluatorEvaluateDetailed:
    """Tests for RelevanceEvaluator.evaluate_detailed method."""

    @pytest.mark.asyncio
    async def test_evaluate_detailed_returns_result(self) -> None:
        """Returns RelevanceResult with aspects and reasoning."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """{
            "score": 0.8,
            "reasoning": "The answer addresses most aspects of the question.",
            "aspects_covered": ["main topic", "key details"],
            "aspects_missing": ["additional context"]
        }"""

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="A detailed response here.",
            query="Tell me about the main topic with details and context.",
        )

        assert isinstance(result, RelevanceResult)
        assert result.score == 0.8
        assert "addresses most aspects" in result.reasoning
        assert result.aspects_covered == ["main topic", "key details"]
        assert result.aspects_missing == ["additional context"]

    @pytest.mark.asyncio
    async def test_evaluate_detailed_empty_response(self) -> None:
        """Empty response returns appropriate result."""
        mock_llm = AsyncMock()

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="   ",
            query="What is the capital?",
        )

        assert result.score == 0.0
        assert result.aspects_covered == []
        assert result.aspects_missing == ["entire question"]
        assert "Empty response" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_empty_query(self) -> None:
        """Empty query returns appropriate result."""
        mock_llm = AsyncMock()

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="Some answer here.",
            query="   ",
        )

        assert result.score == 1.0
        assert result.aspects_covered == []
        assert result.aspects_missing == []
        assert "No question provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_all_aspects_covered(self) -> None:
        """All aspects covered returns perfect score."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """{
            "score": 1.0,
            "reasoning": "All aspects of the question are fully addressed.",
            "aspects_covered": ["aspect 1", "aspect 2", "aspect 3"],
            "aspects_missing": []
        }"""

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="Complete answer covering everything.",
            query="Multi-part question?",
        )

        assert result.score == 1.0
        assert len(result.aspects_covered) == 3
        assert len(result.aspects_missing) == 0


class TestRelevanceEvaluatorJsonParsing:
    """Tests for JSON parsing logic."""

    @pytest.mark.asyncio
    async def test_parse_clean_json(self) -> None:
        """Parses clean JSON correctly."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            '{"score": 0.75, "reasoning": "Good", "aspects_covered": ["a"], "aspects_missing": []}'
        )

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        assert result.score == 0.75

    @pytest.mark.asyncio
    async def test_parse_json_with_surrounding_text(self) -> None:
        """Parses JSON embedded in text."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = 'Here is my analysis: {"score": 0.9, "reasoning": "Great answer", "aspects_covered": [], "aspects_missing": []} Thanks!'

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_parse_json_missing_optional_fields(self) -> None:
        """Handles missing optional fields gracefully."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"score": 0.5, "reasoning": "OK"}'

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        assert result.score == 0.5
        assert result.aspects_covered == []
        assert result.aspects_missing == []

    @pytest.mark.asyncio
    async def test_parse_invalid_json(self) -> None:
        """Returns default values for invalid JSON."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "This is not valid JSON at all"

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        # Default score is 0.0 when parsing fails
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_clamp_score_above_one(self) -> None:
        """Clamps score above 1.0 to 1.0."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"score": 1.5, "reasoning": "Invalid score"}'

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_clamp_score_below_zero(self) -> None:
        """Clamps score below 0.0 to 0.0."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"score": -0.5, "reasoning": "Invalid score"}'

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_aspects_as_non_list_handled(self) -> None:
        """Handles non-list aspects gracefully."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            '{"score": 0.7, "reasoning": "OK", "aspects_covered": "not a list", "aspects_missing": 123}'
        )

        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Query")

        assert result.score == 0.7
        assert result.aspects_covered == []
        assert result.aspects_missing == []


class TestRelevanceEvaluatorErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_llm_error_raises_evaluation_error(self) -> None:
        """LLM error is wrapped in EvaluationError."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM connection failed")

        evaluator = RelevanceEvaluator(llm=mock_llm)

        with pytest.raises(EvaluationError, match="Failed to evaluate relevance"):
            await evaluator.evaluate_detailed("Response", "Query")


class TestRelevanceEvaluatorProtocolCompliance:
    """Tests for EvaluatorProtocol compliance."""

    def test_implements_protocol(self) -> None:
        """RelevanceEvaluator implements EvaluatorProtocol."""
        from ragnarok_ai.core.protocols import EvaluatorProtocol

        mock_llm = AsyncMock()
        evaluator = RelevanceEvaluator(llm=mock_llm)

        assert isinstance(evaluator, EvaluatorProtocol)

    def test_has_evaluate_method(self) -> None:
        """RelevanceEvaluator has evaluate method."""
        mock_llm = AsyncMock()
        evaluator = RelevanceEvaluator(llm=mock_llm)

        assert hasattr(evaluator, "evaluate")
        assert callable(evaluator.evaluate)


class TestRelevanceResultModel:
    """Tests for RelevanceResult model."""

    def test_create_relevance_result(self) -> None:
        """RelevanceResult can be created with valid data."""
        result = RelevanceResult(
            score=0.85,
            reasoning="The answer addresses the question well.",
            aspects_covered=["main topic"],
            aspects_missing=["details"],
        )

        assert result.score == 0.85
        assert result.reasoning == "The answer addresses the question well."
        assert result.aspects_covered == ["main topic"]
        assert result.aspects_missing == ["details"]

    def test_create_with_defaults(self) -> None:
        """RelevanceResult uses defaults for optional fields."""
        result = RelevanceResult(
            score=0.5,
            reasoning="OK",
        )

        assert result.aspects_covered == []
        assert result.aspects_missing == []

    def test_score_validation(self) -> None:
        """Score must be between 0 and 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RelevanceResult(score=1.5, reasoning="Invalid")

        with pytest.raises(ValidationError):
            RelevanceResult(score=-0.1, reasoning="Invalid")

    def test_relevance_result_is_frozen(self) -> None:
        """RelevanceResult is immutable."""
        from pydantic import ValidationError

        result = RelevanceResult(score=1.0, reasoning="OK")

        with pytest.raises(ValidationError):
            result.score = 0.5
