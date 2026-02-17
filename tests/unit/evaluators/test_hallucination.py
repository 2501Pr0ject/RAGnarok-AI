"""Tests for Hallucination detector."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.evaluators.hallucination import (
    Hallucination,
    HallucinationDetector,
    HallucinationResult,
)


class TestHallucinationDetectorInit:
    """Tests for HallucinationDetector initialization."""

    def test_init_with_llm(self) -> None:
        """Detector stores the LLM reference."""
        mock_llm = AsyncMock()
        detector = HallucinationDetector(llm=mock_llm)

        assert detector.llm is mock_llm


class TestHallucinationDetectorEvaluate:
    """Tests for HallucinationDetector.evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_no_hallucinations(self) -> None:
        """Returns 0.0 when no hallucinations are detected."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital of France"]',
            '{"is_hallucination": false, "reason": "Supported by context."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="Paris is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_all_hallucinations(self) -> None:
        """Returns 1.0 when all claims are hallucinations."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris was founded in 500 BC"]',
            '{"is_hallucination": true, "reason": "Not mentioned in context."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="Paris was founded in 500 BC.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_partial_hallucinations(self) -> None:
        """Returns partial score when some claims are hallucinations."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital of France", "Paris was founded in 500 BC"]',
            '{"is_hallucination": false, "reason": "Supported by context."}',
            '{"is_hallucination": true, "reason": "Not mentioned in context."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="Paris is the capital of France. Paris was founded in 500 BC.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self) -> None:
        """Returns 0.0 for empty response (no claims to hallucinate)."""
        mock_llm = AsyncMock()

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="",
            context="Some context here.",
        )

        assert score == 0.0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_empty_context(self) -> None:
        """Returns 1.0 for empty context (all claims are potential hallucinations)."""
        mock_llm = AsyncMock()

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="Paris is the capital.",
            context="",
        )

        assert score == 1.0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_no_claims_extracted(self) -> None:
        """Returns 0.0 when no claims can be extracted."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "[]"

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="Hello!",
            context="Some context.",
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_query_parameter_ignored(self) -> None:
        """Query parameter is accepted but not used."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital"]',
            '{"is_hallucination": false, "reason": "Confirmed."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        score = await detector.evaluate(
            response="Paris is the capital.",
            context="Its capital is Paris.",
            query="What is the capital?",
        )

        assert score == 0.0


class TestHallucinationDetectorEvaluateDetailed:
    """Tests for HallucinationDetector.evaluate_detailed method."""

    @pytest.mark.asyncio
    async def test_evaluate_detailed_returns_result(self) -> None:
        """Returns HallucinationResult with hallucinations list and reasoning."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Paris is the capital", "Founded in 500 BC"]',
            '{"is_hallucination": false, "reason": "Confirmed."}',
            '{"is_hallucination": true, "reason": "Not in context."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed(
            response="Paris is the capital. Founded in 500 BC.",
            context="France is a country. Its capital is Paris.",
        )

        assert isinstance(result, HallucinationResult)
        assert result.score == 0.5
        assert result.total_claims == 2
        assert len(result.hallucinations) == 1
        assert result.hallucinations[0].claim == "Founded in 500 BC"
        assert "Not in context" in result.hallucinations[0].reason
        assert "1 out of 2" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_no_hallucinations(self) -> None:
        """No hallucinations returns appropriate result."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim 1", "Claim 2"]',
            '{"is_hallucination": false, "reason": "OK"}',
            '{"is_hallucination": false, "reason": "OK"}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed(
            response="Response with claims.",
            context="Supporting context.",
        )

        assert result.score == 0.0
        assert result.hallucinations == []
        assert result.total_claims == 2
        assert "No hallucinations" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_all_hallucinations(self) -> None:
        """All hallucinations returns appropriate result."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["False claim 1", "False claim 2"]',
            '{"is_hallucination": true, "reason": "Made up"}',
            '{"is_hallucination": true, "reason": "Fabricated"}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed(
            response="False claims here.",
            context="Unrelated context.",
        )

        assert result.score == 1.0
        assert len(result.hallucinations) == 2
        assert result.total_claims == 2
        assert "All claims" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_empty_response(self) -> None:
        """Empty response returns appropriate result."""
        mock_llm = AsyncMock()

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed(
            response="   ",
            context="Some context.",
        )

        assert result.score == 0.0
        assert result.hallucinations == []
        assert result.total_claims == 0
        assert "Empty response" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_empty_context(self) -> None:
        """Empty context returns appropriate result."""
        mock_llm = AsyncMock()

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed(
            response="Some response.",
            context="   ",
        )

        assert result.score == 1.0
        assert result.hallucinations == []
        assert result.total_claims == 0
        assert "No context provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_detailed_no_claims(self) -> None:
        """No claims returns appropriate result."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "[]"

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed(
            response="Hello there!",
            context="Some context.",
        )

        assert result.score == 0.0
        assert result.hallucinations == []
        assert result.total_claims == 0
        assert "No verifiable claims" in result.reasoning


class TestHallucinationDetectorClaimExtraction:
    """Tests for claim extraction logic."""

    @pytest.mark.asyncio
    async def test_extract_claims_json_array(self) -> None:
        """Extracts claims from valid JSON array."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim A", "Claim B"]',
            '{"is_hallucination": false, "reason": "OK"}',
            '{"is_hallucination": false, "reason": "OK"}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed("Response", "Context")

        assert result.total_claims == 2

    @pytest.mark.asyncio
    async def test_extract_claims_with_surrounding_text(self) -> None:
        """Extracts claims from JSON array embedded in text."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            'Here are the claims: ["Claim X"] and that\'s it.',
            '{"is_hallucination": false, "reason": "Found."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed("Response", "Context")

        assert result.total_claims == 1

    @pytest.mark.asyncio
    async def test_extract_claims_llm_error(self) -> None:
        """Raises EvaluationError on LLM failure."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        detector = HallucinationDetector(llm=mock_llm)

        with pytest.raises(EvaluationError, match="Failed to extract claims"):
            await detector.evaluate_detailed("Response", "Context")


class TestHallucinationDetectorHallucinationCheck:
    """Tests for hallucination checking logic."""

    @pytest.mark.asyncio
    async def test_check_hallucination_detected(self) -> None:
        """Correctly identifies hallucination."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Made up fact"]',
            '{"is_hallucination": true, "reason": "Not in context."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed("Made up fact.", "Real context.")

        assert len(result.hallucinations) == 1
        assert result.hallucinations[0].claim == "Made up fact"

    @pytest.mark.asyncio
    async def test_check_hallucination_not_detected(self) -> None:
        """Correctly identifies supported claim."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["True fact"]',
            '{"is_hallucination": false, "reason": "Supported."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed("True fact.", "Context with true fact.")

        assert len(result.hallucinations) == 0

    @pytest.mark.asyncio
    async def test_check_hallucination_json_with_text(self) -> None:
        """Parses hallucination check result with surrounding text."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim"]',
            'Based on analysis: {"is_hallucination": true, "reason": "Fabricated."}',
        ]

        detector = HallucinationDetector(llm=mock_llm)
        result = await detector.evaluate_detailed("Response", "Context")

        assert len(result.hallucinations) == 1

    @pytest.mark.asyncio
    async def test_check_hallucination_llm_error(self) -> None:
        """Raises EvaluationError on verification failure."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '["Claim"]',
            Exception("Verification failed"),
        ]

        detector = HallucinationDetector(llm=mock_llm)

        with pytest.raises(EvaluationError, match="Failed to check hallucination"):
            await detector.evaluate_detailed("Response", "Context")


class TestHallucinationDetectorProtocolCompliance:
    """Tests for EvaluatorProtocol compliance."""

    def test_implements_protocol(self) -> None:
        """HallucinationDetector implements EvaluatorProtocol."""
        from ragnarok_ai.core.protocols import EvaluatorProtocol

        mock_llm = AsyncMock()
        detector = HallucinationDetector(llm=mock_llm)

        assert isinstance(detector, EvaluatorProtocol)

    def test_has_evaluate_method(self) -> None:
        """HallucinationDetector has evaluate method."""
        mock_llm = AsyncMock()
        detector = HallucinationDetector(llm=mock_llm)

        assert hasattr(detector, "evaluate")
        assert callable(detector.evaluate)


class TestHallucinationModel:
    """Tests for Hallucination model."""

    def test_create_hallucination(self) -> None:
        """Hallucination can be created with valid data."""
        hallucination = Hallucination(
            claim="Made up claim",
            reason="Not in context",
        )

        assert hallucination.claim == "Made up claim"
        assert hallucination.reason == "Not in context"

    def test_hallucination_is_frozen(self) -> None:
        """Hallucination is immutable."""
        from pydantic import ValidationError

        hallucination = Hallucination(
            claim="Test",
            reason="Reason",
        )

        with pytest.raises(ValidationError):
            hallucination.claim = "New claim"


class TestHallucinationResultModel:
    """Tests for HallucinationResult model."""

    def test_create_hallucination_result(self) -> None:
        """HallucinationResult can be created with valid data."""
        result = HallucinationResult(
            score=0.25,
            hallucinations=[],
            total_claims=4,
            reasoning="Test reasoning",
        )

        assert result.score == 0.25
        assert result.hallucinations == []
        assert result.total_claims == 4
        assert result.reasoning == "Test reasoning"

    def test_create_with_defaults(self) -> None:
        """HallucinationResult uses defaults for optional fields."""
        result = HallucinationResult(
            score=0.5,
            reasoning="OK",
        )

        assert result.hallucinations == []
        assert result.total_claims == 0

    def test_score_validation(self) -> None:
        """Score must be between 0 and 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HallucinationResult(score=1.5, reasoning="Invalid")

        with pytest.raises(ValidationError):
            HallucinationResult(score=-0.1, reasoning="Invalid")

    def test_total_claims_validation(self) -> None:
        """Total claims must be non-negative."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HallucinationResult(score=0.5, total_claims=-1, reasoning="Invalid")

    def test_hallucination_result_is_frozen(self) -> None:
        """HallucinationResult is immutable."""
        from pydantic import ValidationError

        result = HallucinationResult(score=0.0, reasoning="OK")

        with pytest.raises(ValidationError):
            result.score = 0.5
