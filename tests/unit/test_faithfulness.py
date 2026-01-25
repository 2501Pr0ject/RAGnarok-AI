"""Tests for Faithfulness evaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.evaluators.faithfulness import (
    ClaimVerification,
    FaithfulnessEvaluator,
    FaithfulnessResult,
)


class TestFaithfulnessEvaluatorInit:
    """Tests for FaithfulnessEvaluator initialization."""

    def test_init_with_llm(self) -> None:
        """Evaluator stores the LLM reference."""
        mock_llm = AsyncMock()
        evaluator = FaithfulnessEvaluator(llm=mock_llm)

        assert evaluator.llm is mock_llm


class TestFaithfulnessEvaluatorEvaluate:
    """Tests for FaithfulnessEvaluator.evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_all_claims_supported(self) -> None:
        """Returns 1.0 when all claims are supported."""
        mock_llm = AsyncMock()
        # First call: extract claims
        mock_llm.generate.side_effect = [
            '["Paris is the capital of France"]',
            '{"supported": true, "reasoning": "The context states Paris is the capital."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_no_claims_supported(self) -> None:
        """Returns 0.0 when no claims are supported."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Berlin is the capital of France"]',
            '{"supported": false, "reasoning": "The context says Paris is the capital, not Berlin."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Berlin is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_partial_support(self) -> None:
        """Returns partial score when some claims are supported."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital of France", "France has 100 million people"]',
            '{"supported": true, "reasoning": "Context confirms Paris is the capital."}',
            '{"supported": false, "reasoning": "Population not mentioned in context."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital of France. France has 100 million people.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self) -> None:
        """Returns 1.0 for empty response (no claims to verify)."""
        mock_llm = AsyncMock()

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="",
            context="Some context here.",
        )

        assert score == 1.0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_empty_context(self) -> None:
        """Returns 0.0 for empty context (nothing to verify against)."""
        mock_llm = AsyncMock()

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital.",
            context="",
        )

        assert score == 0.0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_no_claims_extracted(self) -> None:
        """Returns 1.0 when no claims can be extracted."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "[]"

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Hello!",
            context="Some context.",
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_query_parameter_ignored(self) -> None:
        """Query parameter is accepted but not used."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital"]',
            '{"supported": true, "reasoning": "Confirmed."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        score = await evaluator.evaluate(
            response="Paris is the capital.",
            context="Its capital is Paris.",
            query="What is the capital?",
        )

        assert score == 1.0


class TestFaithfulnessEvaluatorEvaluateDetailed:
    """Tests for FaithfulnessEvaluator.evaluate_detailed method."""

    @pytest.mark.asyncio
    async def test_evaluate_detailed_returns_result(self) -> None:
        """Returns FaithfulnessResult with claims and reasoning."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital of France"]',
            '{"supported": true, "reasoning": "The context confirms this."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="Paris is the capital of France.",
            context="France is a country. Its capital is Paris.",
        )

        assert isinstance(result, FaithfulnessResult)
        assert result.score == 1.0
        assert len(result.claims) == 1
        assert result.claims[0].claim == "Paris is the capital of France"
        assert result.claims[0].supported is True
        assert "confirms" in result.claims[0].reasoning
        assert "All claims" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_multiple_claims(self) -> None:
        """Handles multiple claims correctly."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim 1", "Claim 2", "Claim 3"]',
            '{"supported": true, "reasoning": "Reason 1"}',
            '{"supported": false, "reasoning": "Reason 2"}',
            '{"supported": true, "reasoning": "Reason 3"}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="Response with multiple claims.",
            context="Some context.",
        )

        assert result.score == pytest.approx(2 / 3)
        assert len(result.claims) == 3
        assert result.claims[0].supported is True
        assert result.claims[1].supported is False
        assert result.claims[2].supported is True
        assert "2 out of 3" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_empty_response(self) -> None:
        """Empty response returns appropriate result."""
        mock_llm = AsyncMock()

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="   ",
            context="Some context.",
        )

        assert result.score == 1.0
        assert result.claims == []
        assert "Empty response" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_no_claims(self) -> None:
        """No claims returns appropriate result."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "[]"

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed(
            response="Hello there!",
            context="Some context.",
        )

        assert result.score == 1.0
        assert result.claims == []
        assert "No verifiable claims" in result.reasoning


class TestFaithfulnessEvaluatorClaimExtraction:
    """Tests for claim extraction logic."""

    @pytest.mark.asyncio
    async def test_extract_claims_json_array(self) -> None:
        """Extracts claims from valid JSON array."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim A", "Claim B"]',
            '{"supported": true, "reasoning": "OK"}',
            '{"supported": true, "reasoning": "OK"}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Context")

        assert len(result.claims) == 2
        assert result.claims[0].claim == "Claim A"
        assert result.claims[1].claim == "Claim B"

    @pytest.mark.asyncio
    async def test_extract_claims_with_surrounding_text(self) -> None:
        """Extracts claims from JSON array embedded in text."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            'Here are the claims: ["Claim X"] and that\'s it.',
            '{"supported": true, "reasoning": "Found."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Context")

        assert len(result.claims) == 1
        assert result.claims[0].claim == "Claim X"

    @pytest.mark.asyncio
    async def test_extract_claims_invalid_json(self) -> None:
        """Raises EvaluationError on invalid JSON."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        evaluator = FaithfulnessEvaluator(llm=mock_llm)

        with pytest.raises(EvaluationError, match="Failed to extract claims"):
            await evaluator.evaluate_detailed("Response", "Context")


class TestFaithfulnessEvaluatorClaimVerification:
    """Tests for claim verification logic."""

    @pytest.mark.asyncio
    async def test_verify_claim_supported(self) -> None:
        """Correctly identifies supported claim."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["The sky is blue"]',
            '{"supported": true, "reasoning": "Context mentions blue sky."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("The sky is blue.", "The sky is blue.")

        assert result.claims[0].supported is True

    @pytest.mark.asyncio
    async def test_verify_claim_not_supported(self) -> None:
        """Correctly identifies unsupported claim."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["The sky is green"]',
            '{"supported": false, "reasoning": "Context says sky is blue."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("The sky is green.", "The sky is blue.")

        assert result.claims[0].supported is False

    @pytest.mark.asyncio
    async def test_verify_claim_json_with_text(self) -> None:
        """Parses verification result with surrounding text."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim"]',
            'Based on analysis: {"supported": true, "reasoning": "Found it."}',
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_detailed("Response", "Context")

        assert result.claims[0].supported is True

    @pytest.mark.asyncio
    async def test_verify_claim_llm_error(self) -> None:
        """Raises EvaluationError on verification failure."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim"]',
            Exception("Verification failed"),
        ]

        evaluator = FaithfulnessEvaluator(llm=mock_llm)

        with pytest.raises(EvaluationError, match="Failed to verify claim"):
            await evaluator.evaluate_detailed("Response", "Context")


class TestFaithfulnessEvaluatorProtocolCompliance:
    """Tests for EvaluatorProtocol compliance."""

    def test_implements_protocol(self) -> None:
        """FaithfulnessEvaluator implements EvaluatorProtocol."""
        from ragnarok_ai.core.protocols import EvaluatorProtocol

        mock_llm = AsyncMock()
        evaluator = FaithfulnessEvaluator(llm=mock_llm)

        assert isinstance(evaluator, EvaluatorProtocol)

    def test_has_evaluate_method(self) -> None:
        """FaithfulnessEvaluator has evaluate method."""
        mock_llm = AsyncMock()
        evaluator = FaithfulnessEvaluator(llm=mock_llm)

        assert hasattr(evaluator, "evaluate")
        assert callable(evaluator.evaluate)


class TestClaimVerificationModel:
    """Tests for ClaimVerification model."""

    def test_create_claim_verification(self) -> None:
        """ClaimVerification can be created with valid data."""
        verification = ClaimVerification(
            claim="Test claim",
            supported=True,
            reasoning="Test reasoning",
        )

        assert verification.claim == "Test claim"
        assert verification.supported is True
        assert verification.reasoning == "Test reasoning"

    def test_claim_verification_is_frozen(self) -> None:
        """ClaimVerification is immutable."""
        from pydantic import ValidationError

        verification = ClaimVerification(
            claim="Test",
            supported=True,
            reasoning="Reason",
        )

        with pytest.raises(ValidationError):
            verification.claim = "New claim"


class TestFaithfulnessResultModel:
    """Tests for FaithfulnessResult model."""

    def test_create_faithfulness_result(self) -> None:
        """FaithfulnessResult can be created with valid data."""
        result = FaithfulnessResult(
            score=0.75,
            claims=[],
            reasoning="Test reasoning",
        )

        assert result.score == 0.75
        assert result.claims == []
        assert result.reasoning == "Test reasoning"

    def test_score_validation(self) -> None:
        """Score must be between 0 and 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FaithfulnessResult(score=1.5, claims=[], reasoning="Invalid")

        with pytest.raises(ValidationError):
            FaithfulnessResult(score=-0.1, claims=[], reasoning="Invalid")

    def test_faithfulness_result_is_frozen(self) -> None:
        """FaithfulnessResult is immutable."""
        from pydantic import ValidationError

        result = FaithfulnessResult(score=1.0, claims=[], reasoning="OK")

        with pytest.raises(ValidationError):
            result.score = 0.5
