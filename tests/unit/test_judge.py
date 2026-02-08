"""Tests for LLM-as-Judge evaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnarok_ai.evaluators.judge import (
    JudgeResult,
    JudgeResults,
    LLMJudge,
    parse_judge_response,
)

# =============================================================================
# Parser Tests (No LLM required)
# =============================================================================


class TestParseJudgeResponse:
    """Tests for parse_judge_response function."""

    def test_parse_complete_response(self):
        """Should parse a well-formatted response."""
        response = """Score: 5
Verdict: PASS
Explanation: The answer is completely faithful to the context. All claims are directly supported."""

        result = parse_judge_response(response, "faithfulness")

        assert result.verdict == "PASS"
        assert result.raw_score == 5
        assert result.score == 1.0  # (5-1)/4
        assert result.criteria == "faithfulness"
        assert "completely faithful" in result.explanation

    def test_parse_fail_response(self):
        """Should parse a failing response."""
        response = """Score: 1
Verdict: FAIL
Explanation: The answer contradicts the context entirely."""

        result = parse_judge_response(response, "hallucination")

        assert result.verdict == "FAIL"
        assert result.raw_score == 1
        assert result.score == 0.0  # (1-1)/4
        assert result.criteria == "hallucination"

    def test_parse_partial_response(self):
        """Should parse a partial response."""
        response = """Score: 3
Verdict: PARTIAL
Explanation: Some claims are supported, others are not."""

        result = parse_judge_response(response, "relevance")

        assert result.verdict == "PARTIAL"
        assert result.raw_score == 3
        assert result.score == 0.5  # (3-1)/4

    def test_infer_verdict_from_score(self):
        """Should infer verdict when not explicitly provided."""
        response = """Score: 4
Explanation: Good answer."""

        result = parse_judge_response(response, "completeness")

        assert result.verdict == "PASS"  # score >= 4 -> PASS
        assert result.raw_score == 4

    def test_infer_fail_from_low_score(self):
        """Should infer FAIL for low scores."""
        response = """Score: 2
Explanation: Poor answer."""

        result = parse_judge_response(response, "faithfulness")

        assert result.verdict == "FAIL"  # score <= 2 -> FAIL

    def test_infer_partial_from_middle_score(self):
        """Should infer PARTIAL for middle scores."""
        response = """Score: 3
Explanation: Mixed results."""

        result = parse_judge_response(response, "relevance")

        assert result.verdict == "PARTIAL"  # 2 < score < 4 -> PARTIAL

    def test_default_score_when_missing(self):
        """Should use default score 3 when not found."""
        response = """The answer seems okay but has some issues.
Verdict: PARTIAL
Explanation: Mixed quality."""

        result = parse_judge_response(response, "completeness")

        assert result.raw_score == 3  # default
        assert result.verdict == "PARTIAL"

    def test_clamp_invalid_score(self):
        """Should clamp scores to valid 1-5 range."""
        response = """Score: 7
Explanation: Way too high."""

        result = parse_judge_response(response, "faithfulness")

        assert result.raw_score == 5  # clamped to max

    def test_case_insensitive_verdict(self):
        """Should handle case-insensitive verdict matching."""
        response = """Score: 4
Verdict: pass
Explanation: All good."""

        result = parse_judge_response(response, "relevance")

        assert result.verdict == "PASS"

    def test_strip_trailing_artifacts(self):
        """Should clean up explanation trailing artifacts."""
        response = """Score: 4
Verdict: PASS
Explanation: Good answer with proper context.

###Some trailing prompt artifact"""

        result = parse_judge_response(response, "faithfulness")

        assert "###" not in result.explanation
        assert result.explanation == "Good answer with proper context."


# =============================================================================
# JudgeResults Tests
# =============================================================================


class TestJudgeResults:
    """Tests for JudgeResults dataclass."""

    def test_overall_score_calculation(self):
        """Should calculate average score."""
        results = JudgeResults(
            faithfulness=JudgeResult("PASS", 1.0, "Good", "faithfulness", 5),
            relevance=JudgeResult("PASS", 0.75, "Good", "relevance", 4),
            hallucination=JudgeResult("PASS", 1.0, "None", "hallucination", 5),
            completeness=JudgeResult("PARTIAL", 0.5, "Okay", "completeness", 3),
        )

        assert results.overall_score == pytest.approx(0.8125)  # (1+0.75+1+0.5)/4

    def test_overall_verdict_all_pass(self):
        """Should return PASS when all pass."""
        results = JudgeResults(
            faithfulness=JudgeResult("PASS", 1.0, "", "faithfulness", 5),
            relevance=JudgeResult("PASS", 1.0, "", "relevance", 5),
            hallucination=JudgeResult("PASS", 1.0, "", "hallucination", 5),
            completeness=JudgeResult("PASS", 1.0, "", "completeness", 5),
        )

        assert results.overall_verdict == "PASS"

    def test_overall_verdict_one_fail(self):
        """Should return FAIL if any criterion fails."""
        results = JudgeResults(
            faithfulness=JudgeResult("FAIL", 0.0, "", "faithfulness", 1),
            relevance=JudgeResult("PASS", 1.0, "", "relevance", 5),
            hallucination=JudgeResult("PASS", 1.0, "", "hallucination", 5),
            completeness=JudgeResult("PASS", 1.0, "", "completeness", 5),
        )

        assert results.overall_verdict == "FAIL"

    def test_overall_verdict_one_partial(self):
        """Should return PARTIAL if any criterion is partial (and none fail)."""
        results = JudgeResults(
            faithfulness=JudgeResult("PASS", 1.0, "", "faithfulness", 5),
            relevance=JudgeResult("PARTIAL", 0.5, "", "relevance", 3),
            hallucination=JudgeResult("PASS", 1.0, "", "hallucination", 5),
            completeness=JudgeResult("PASS", 1.0, "", "completeness", 5),
        )

        assert results.overall_verdict == "PARTIAL"

    def test_to_dict(self):
        """Should convert to dictionary."""
        results = JudgeResults(
            faithfulness=JudgeResult("PASS", 1.0, "Good", "faithfulness", 5),
            relevance=JudgeResult("PASS", 0.75, "Okay", "relevance", 4),
            hallucination=JudgeResult("PASS", 1.0, "None", "hallucination", 5),
            completeness=JudgeResult("PARTIAL", 0.5, "Partial", "completeness", 3),
        )

        d = results.to_dict()

        assert d["faithfulness"]["verdict"] == "PASS"
        assert d["relevance"]["raw_score"] == 4
        assert d["overall"]["verdict"] == "PARTIAL"
        assert d["overall"]["score"] == pytest.approx(0.8125)


# =============================================================================
# LLMJudge Tests (Mocked)
# =============================================================================


class TestLLMJudge:
    """Tests for LLMJudge class with mocked LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate = AsyncMock()
        return llm

    @pytest.fixture
    def judge(self, mock_llm):
        """Create judge with mock LLM."""
        return LLMJudge(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness(self, judge, mock_llm):
        """Should evaluate faithfulness."""
        mock_llm.generate.return_value = """Score: 5
Verdict: PASS
Explanation: All claims are supported by the context."""

        result = await judge.evaluate_faithfulness(
            context="Paris is the capital of France.",
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
        )

        assert result.verdict == "PASS"
        assert result.score == 1.0
        assert result.criteria == "faithfulness"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_relevance(self, judge, mock_llm):
        """Should evaluate relevance."""
        mock_llm.generate.return_value = """Score: 4
Verdict: PASS
Explanation: Answer addresses the question."""

        result = await judge.evaluate_relevance(
            question="What is Python?",
            answer="Python is a programming language.",
        )

        assert result.verdict == "PASS"
        assert result.criteria == "relevance"

    @pytest.mark.asyncio
    async def test_detect_hallucination(self, judge, mock_llm):
        """Should detect hallucination."""
        mock_llm.generate.return_value = """Score: 2
Verdict: FAIL
Explanation: The claim about NASA is not in the context."""

        result = await judge.detect_hallucination(
            context="Python was created in 1991.",
            answer="Python was created in 1991 for NASA.",
        )

        assert result.verdict == "FAIL"
        assert result.criteria == "hallucination"

    @pytest.mark.asyncio
    async def test_evaluate_completeness(self, judge, mock_llm):
        """Should evaluate completeness."""
        mock_llm.generate.return_value = """Score: 3
Verdict: PARTIAL
Explanation: Covers main point but misses details."""

        result = await judge.evaluate_completeness(
            question="What is Python and who created it?",
            answer="Python is a programming language.",
            context="Python is a programming language created by Guido van Rossum in 1991.",
        )

        assert result.verdict == "PARTIAL"
        assert result.criteria == "completeness"

    @pytest.mark.asyncio
    async def test_evaluate_all(self, judge, mock_llm):
        """Should run all evaluations."""
        mock_llm.generate.side_effect = [
            "Score: 5\nVerdict: PASS\nExplanation: Faithful",
            "Score: 4\nVerdict: PASS\nExplanation: Relevant",
            "Score: 5\nVerdict: PASS\nExplanation: No hallucination",
            "Score: 4\nVerdict: PASS\nExplanation: Complete",
        ]

        results = await judge.evaluate_all(
            context="Test context",
            question="Test question",
            answer="Test answer",
        )

        assert isinstance(results, JudgeResults)
        assert results.faithfulness.verdict == "PASS"
        assert results.relevance.verdict == "PASS"
        assert results.hallucination.verdict == "PASS"
        assert results.completeness.verdict == "PASS"
        assert results.overall_verdict == "PASS"
        assert mock_llm.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, judge, mock_llm):
        """Should evaluate batch of items."""
        mock_llm.generate.return_value = "Score: 4\nVerdict: PASS\nExplanation: Good"

        items = [
            {"context": "C1", "question": "Q1", "answer": "A1"},
            {"context": "C2", "question": "Q2", "answer": "A2"},
        ]

        results = await judge.evaluate_batch(items)

        assert len(results) == 2
        assert all(isinstance(r, JudgeResults) for r in results)


# =============================================================================
# Integration Tests (Require Ollama)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_judge_with_prometheus():
    """Integration test with actual Prometheus 2 model.

    Run with: pytest -m integration
    Requires: ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
    """
    judge = LLMJudge()  # Uses default Prometheus 2 Q5_K_M

    result = await judge.evaluate_faithfulness(
        context="Paris est la capitale de la France. Elle compte environ 2 millions d'habitants.",
        question="Quelle est la capitale de la France ?",
        answer="Paris est la capitale de la France.",
    )

    # Prometheus 2 Q5_K_M tends to be conservative in scoring
    assert result.verdict in ["PASS", "PARTIAL"]
    assert result.score >= 0.5  # At least 3/5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_judge_detects_hallucination():
    """Integration test for hallucination detection.

    Run with: pytest -m integration
    Requires: ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
    """
    judge = LLMJudge()  # Uses default Prometheus 2 Q5_K_M

    result = await judge.detect_hallucination(
        context="Python est un langage de programmation créé en 1991.",
        answer="Python a été créé en 1991 par Guido van Rossum pour la NASA.",
    )

    # "Guido van Rossum" and "pour la NASA" are hallucinated (not in context)
    # Prometheus 2 should detect this, but may be conservative
    assert result.verdict in ["FAIL", "PARTIAL"]
    assert result.score <= 0.75  # Should not be rated highly
