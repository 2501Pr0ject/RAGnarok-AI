"""Tests for LLM-as-Judge evaluator."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragnarok_ai.core.exceptions import (
    EvaluationError,
    JudgeConnectionError,
    JudgeDependencyError,
    JudgeModelNotFoundError,
    LLMConnectionError,
)
from ragnarok_ai.evaluators.judge import (
    COMPLETENESS_PROMPT,
    FAITHFULNESS_PROMPT,
    HALLUCINATION_PROMPT,
    PROMETHEUS_ABS_SYSTEM_PROMPT,
    PROMETHEUS_RUBRICS,
    RELEVANCE_PROMPT,
    JudgeResult,
    JudgeResults,
    LLMJudge,
    build_prometheus_prompt,
    parse_judge_response,
    parse_prometheus_response,
    verdict_from_raw_score,
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

    def test_missing_score_raises(self):
        """A response without 'Score: N' is an explicit error, never a default score.

        Regression: up to 1.12.1 a missing score silently became 3.
        """
        response = """The answer seems okay but has some issues.
Verdict: PARTIAL
Explanation: Mixed quality."""

        with pytest.raises(EvaluationError, match="Unparseable completeness judgment"):
            parse_judge_response(response, "completeness")

    def test_clamp_invalid_score(self):
        """Should clamp scores to valid 1-5 range."""
        response = """Score: 7
Explanation: Way too high."""

        result = parse_judge_response(response, "faithfulness")

        assert result.raw_score == 5  # clamped to max

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("Score: 2\nVerdict: PASS\nExplanation: contradictory", "FAIL"),
            ("Score: 5\nVerdict: FAIL\nExplanation: contradictory", "PASS"),
            ("Score: 3\nVerdict: PASS\nExplanation: contradictory", "PARTIAL"),
            ("Score: 4\nVerdict: partial\nExplanation: contradictory", "PASS"),
        ],
    )
    def test_verdict_written_by_model_is_ignored(self, response, expected):
        """Regression: the verdict always follows the score, even when the model writes another one."""
        result = parse_judge_response(response, "faithfulness")

        assert result.verdict == expected
        assert result.verdict == verdict_from_raw_score(result.raw_score)

    def test_case_insensitive_verdict(self):
        """A lowercase verdict line does not disturb parsing; the verdict still comes from the score."""
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


class TestLLMJudgeMedicalMode:
    """Tests for LLMJudge with medical_mode enabled."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate = AsyncMock()
        return llm

    @pytest.fixture
    def medical_judge(self, mock_llm):
        """Create judge with medical mode enabled."""
        return LLMJudge(llm=mock_llm, medical_mode=True)

    @pytest.mark.asyncio
    async def test_faithfulness_with_medical_mode(self, medical_judge, mock_llm):
        """Test that medical mode normalizes abbreviations in faithfulness eval."""
        mock_llm.generate.return_value = """Score: 5
Verdict: PASS
Explanation: All claims supported."""

        result = await medical_judge.evaluate_faithfulness(
            context="Patient has CHF",
            question="What condition?",
            answer="Patient has congestive heart failure",
        )

        assert result.verdict == "PASS"
        # Verify the normalizer was used (both texts should be normalized)
        assert medical_judge._normalizer is not None

    @pytest.mark.asyncio
    async def test_hallucination_with_medical_mode(self, medical_judge, mock_llm):
        """Test that medical mode normalizes abbreviations in hallucination detection."""
        mock_llm.generate.return_value = """Score: 5
Verdict: PASS
Explanation: No hallucinations."""

        result = await medical_judge.detect_hallucination(
            context="Patient diagnosed with MI",
            answer="Patient has myocardial infarction",
        )

        assert result.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_completeness_with_medical_mode(self, medical_judge, mock_llm):
        """Test that medical mode normalizes abbreviations in completeness eval."""
        mock_llm.generate.return_value = """Score: 4
Verdict: PASS
Explanation: Complete answer."""

        result = await medical_judge.evaluate_completeness(
            question="What does patient have?",
            answer="Patient has congestive heart failure",
            context="Patient has CHF",
        )

        assert result.verdict == "PASS"


class TestLLMJudgeModelDiscovery:
    """Tests for model discovery and fallback logic."""

    def test_get_available_models_success(self):
        """Test _get_available_models with successful response."""
        from unittest.mock import patch

        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "mistral"},
            ]
        }

        with patch.object(httpx, "get", return_value=mock_response):
            judge = LLMJudge.__new__(LLMJudge)
            models = judge._get_available_models("http://localhost:11434")

        assert "llama3.2" in models
        assert "mistral" in models

    def test_get_available_models_unreachable_server_raises(self):
        """An unreachable server is a connection error, not an empty model list."""
        from unittest.mock import patch

        import httpx

        with (
            patch.object(httpx, "get", side_effect=httpx.ConnectError("Connection refused")),
            pytest.raises(JudgeConnectionError, match="Cannot reach Ollama at http://localhost:11434"),
        ):
            LLMJudge.__new__(LLMJudge)._get_available_models("http://localhost:11434")

    def test_get_available_models_non_200_raises(self):
        """An error status from the server is a connection error too."""
        from unittest.mock import patch

        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "internal error"

        with (
            patch.object(httpx, "get", return_value=mock_response),
            pytest.raises(JudgeConnectionError, match="answered 500"),
        ):
            LLMJudge.__new__(LLMJudge)._get_available_models("http://localhost:11434")

    @pytest.mark.parametrize(
        ("requested", "available", "expected"),
        [
            ("mistral:7b", ["mistral:7b", "llama3.2:latest"], "mistral:7b"),
            ("mistral", ["mistral:latest"], "mistral:latest"),
            (LLMJudge.DEFAULT_MODEL, ["prometheus2:latest", "mistral:7b"], "prometheus2:latest"),
            ("prometheus2", [LLMJudge.DEFAULT_MODEL], LLMJudge.DEFAULT_MODEL),
        ],
    )
    def test_resolve_model_accepted_resolutions(self, requested, available, expected):
        """Exact name, ':latest' normalization and Prometheus aliases resolve."""
        assert LLMJudge._resolve_model(requested, available, "http://x") == expected

    @pytest.mark.parametrize(
        ("requested", "available"),
        [
            ("prometheus2", ["mistral:7b", "llama3.2:latest"]),  # explicit model absent, generic models installed
            (LLMJudge.DEFAULT_MODEL, ["mistral:7b"]),  # default absent
            ("mistral", ["mistral-nemo:latest"]),  # substring is not a match
            ("llama3", ["llama3.2:latest"]),
            ("mistral:7b", []),
        ],
    )
    def test_resolve_model_never_substitutes_another_model(self, requested, available):
        """Regression: no generic fallback, an absent model is an error with the pull command."""
        with pytest.raises(JudgeModelNotFoundError, match=re.escape(f"ollama pull {requested}")):
            LLMJudge._resolve_model(requested, available, "http://x")

    def test_constructor_does_not_fall_back_to_an_installed_generic_model(self):
        """Regression: LLMJudge() with only mistral installed used mistral silently; it now raises."""
        with (
            patch.object(LLMJudge, "_get_available_models", return_value=["mistral:7b"]),
            pytest.raises(JudgeModelNotFoundError, match=re.escape("ollama pull hf.co/RichardErkhov")),
        ):
            LLMJudge()

    def test_constructor_normalizes_latest_tag(self):
        with patch.object(LLMJudge, "_get_available_models", return_value=["mistral:latest"]):
            judge = LLMJudge(model="mistral")

        assert judge.model == "mistral:latest"
        assert judge.llm.model == "mistral:latest"

    def test_constructor_resolves_prometheus_alias_with_a_log(self, caplog):
        with (
            patch.object(LLMJudge, "_get_available_models", return_value=["prometheus2:latest"]),
            caplog.at_level("INFO", logger="ragnarok_ai.evaluators.judge"),
        ):
            judge = LLMJudge()

        assert judge.model == "prometheus2:latest"
        assert judge._prometheus is True
        assert "using the installed Prometheus 2 model 'prometheus2:latest'" in caplog.text


class TestLLMJudgeInit:
    """Tests for LLMJudge initialization."""

    def test_init_with_custom_llm(self):
        """Test initialization with custom LLM."""
        mock_llm = MagicMock()
        mock_llm.model = "custom-model"

        judge = LLMJudge(llm=mock_llm)

        assert judge.llm == mock_llm
        assert judge.model == "custom-model"

    def test_init_with_custom_llm_no_model_attr(self):
        """Test initialization with custom LLM without model attribute."""
        mock_llm = MagicMock(spec=[])  # No attributes

        judge = LLMJudge(llm=mock_llm)

        assert judge.model == "custom"

    def test_medical_mode_disabled(self):
        """Test that medical mode is disabled by default."""
        mock_llm = MagicMock()
        judge = LLMJudge(llm=mock_llm)

        assert judge.medical_mode is False
        assert judge._normalizer is None

    def test_medical_mode_enabled(self):
        """Test that medical mode initializes normalizer."""
        mock_llm = MagicMock()
        judge = LLMJudge(llm=mock_llm, medical_mode=True)

        assert judge.medical_mode is True
        assert judge._normalizer is not None


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


# =============================================================================
# Prometheus 2 native format
# =============================================================================

# Verbatim endings of real Prometheus 2 7B outputs under the 1.12.1 prompts:
# both were silently parsed as a score of 3.
PROMETHEUS_OUTPUT_WITHOUT_RESULT_1 = (
    "Overall, the answer is mostly accurate and aligns well with the context.\n\nSo the score is 4. [PASS]"
)
PROMETHEUS_OUTPUT_WITHOUT_RESULT_2 = (
    "Therefore, the answer is mostly relevant and addresses the main point with minor gaps. \n[PASS]"
)


class TestParsePrometheusResponse:
    """Tests for the '[RESULT] N' parser."""

    def test_parses_result(self):
        result = parse_prometheus_response("The answer is well supported. [RESULT] 4", "faithfulness")

        assert result.raw_score == 4
        assert result.score == 0.75
        assert result.verdict == "PASS"
        assert result.explanation == "The answer is well supported."
        assert result.criteria == "faithfulness"

    def test_parses_parenthesized_result(self):
        result = parse_prometheus_response("Unsupported claims. [RESULT] (2)", "faithfulness")

        assert result.raw_score == 2
        assert result.verdict == "FAIL"

    def test_uses_last_result(self):
        result = parse_prometheus_response("Quoting [RESULT] 1 from the rubric. Final: [RESULT] 5", "relevance")

        assert result.raw_score == 5

    @pytest.mark.parametrize(
        "response",
        [PROMETHEUS_OUTPUT_WITHOUT_RESULT_1, PROMETHEUS_OUTPUT_WITHOUT_RESULT_2, "Score: 4", "[RESULT] 7", ""],
    )
    def test_unparseable_response_raises(self, response):
        """Never a default score (regression: 1.12.1 turned these into 3)."""
        with pytest.raises(EvaluationError, match="no '\\[RESULT\\] N'"):
            parse_prometheus_response(response, "faithfulness")


class TestBuildPrometheusPrompt:
    """Tests for the prometheus-eval absolute-grading template."""

    def test_prompt_follows_official_template(self):
        prompt = build_prometheus_prompt("faithfulness", "Context:\nX\n\nQuestion:\nY", "Z")

        assert prompt.startswith(PROMETHEUS_ABS_SYSTEM_PROMPT)
        assert "###The instruction to evaluate:\nContext:\nX\n\nQuestion:\nY" in prompt
        assert "###Response to evaluate:\nZ" in prompt
        assert "[Does the answer contain only information supported by the context?]" in prompt
        assert "Score 5: Completely faithful - all claims supported by context" in prompt
        assert '"(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"' in prompt
        assert prompt.endswith("###Feedback: ")

    @pytest.mark.parametrize(
        ("criteria", "legacy_prompt"),
        [
            ("faithfulness", FAITHFULNESS_PROMPT),
            ("hallucination", HALLUCINATION_PROMPT),
            ("relevance", RELEVANCE_PROMPT),
            ("completeness", COMPLETENESS_PROMPT),
        ],
    )
    def test_rubric_matches_legacy_prompt(self, criteria, legacy_prompt):
        """Both prompt formats grade against the same question and the same score descriptions."""
        rubric = PROMETHEUS_RUBRICS[criteria]

        assert f"] {rubric['criteria']}\n" in legacy_prompt
        for n in range(1, 6):
            assert f"Score {n}: {rubric[f'score{n}_description']}\n" in legacy_prompt


class TestLLMJudgePromptFormat:
    """The judge picks the native template for Prometheus, 'Score: N' otherwise."""

    @staticmethod
    def make_judge(model: str, response: str) -> tuple[LLMJudge, MagicMock]:
        llm = MagicMock()
        llm.model = model
        llm.generate = AsyncMock(return_value=response)
        return LLMJudge(llm=llm), llm

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("evaluate_faithfulness", {"context": "C", "question": "Q", "answer": "A"}),
            ("evaluate_relevance", {"question": "Q", "answer": "A"}),
            ("detect_hallucination", {"context": "C", "answer": "A"}),
            ("evaluate_completeness", {"question": "Q", "answer": "A", "context": "C"}),
        ],
    )
    async def test_prometheus_uses_native_template(self, method, kwargs):
        judge, llm = self.make_judge(
            "hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M", "Fine. [RESULT] 5"
        )

        result = await getattr(judge, method)(**kwargs)

        prompt = llm.generate.call_args.args[0]
        assert prompt.startswith(PROMETHEUS_ABS_SYSTEM_PROMPT)
        assert "Score: [1-5]" not in prompt
        assert result.raw_score == 5

    @pytest.mark.asyncio
    async def test_prometheus_unparseable_output_raises(self):
        judge, _ = self.make_judge("prometheus2", PROMETHEUS_OUTPUT_WITHOUT_RESULT_1)

        with pytest.raises(EvaluationError):
            await judge.evaluate_faithfulness(context="C", question="Q", answer="A")

    @pytest.mark.asyncio
    async def test_other_judges_keep_score_format(self):
        judge, llm = self.make_judge("mistral:7b", "Score: 4\nVerdict: PASS\nExplanation: ok")

        result = await judge.evaluate_faithfulness(context="C", question="Q", answer="A")

        assert "Score: [1-5]" in llm.generate.call_args.args[0]
        assert result.raw_score == 4

    @pytest.mark.asyncio
    async def test_other_judges_unparseable_output_raises(self):
        judge, _ = self.make_judge("mistral:7b", "I think it is fine.")

        with pytest.raises(EvaluationError):
            await judge.evaluate_relevance(question="Q", answer="A")


class TestLLMJudgeConfigurationErrors:
    """Regression: the three configuration failures are distinct and actionable."""

    def test_missing_httpx_names_the_extra(self):
        """Without httpx, the constructor says which extra to install (fresh interpreter)."""
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['httpx'] = None\n"
            "from ragnarok_ai import LLMJudge\n"
            "try:\n"
            "    LLMJudge()\n"
            "except ImportError as e:\n"
            "    print('IMPORT_ERROR', isinstance(e, RuntimeError), type(e).__name__, e)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

        assert "IMPORT_ERROR True JudgeDependencyError" in out.stdout
        assert "pip install 'ragnarok-ai[ollama]'" in out.stdout

    def test_unreachable_server_is_a_connection_error(self):
        """Ollama down: JudgeConnectionError naming the URL, never 'model not installed'."""
        import httpx

        with (
            patch.object(httpx, "get", side_effect=httpx.ConnectError("Connection refused")),
            pytest.raises(JudgeConnectionError, match="Cannot reach Ollama at http://x:1") as info,
        ):
            LLMJudge(model="prometheus2", base_url="http://x:1")

        assert isinstance(info.value, RuntimeError)  # what earlier versions raised
        assert isinstance(info.value, LLMConnectionError)

    def test_missing_dependency_is_also_a_runtime_error(self):
        err = JudgeDependencyError(extra="ollama", module="httpx")

        assert isinstance(err, ImportError)
        assert isinstance(err, RuntimeError)
        assert "pip install 'ragnarok-ai[ollama]'" in str(err)

    def test_missing_model_gives_the_pull_command(self):
        """Server up, model absent, nothing to fall back on: the error carries 'ollama pull'."""
        with (
            patch.object(LLMJudge, "_get_available_models", return_value=["qwen2.5:7b-instruct"]),
            pytest.raises(JudgeModelNotFoundError, match="ollama pull prometheus2") as info,
        ):
            LLMJudge(model="prometheus2")

        assert info.value.model == "prometheus2"
        assert info.value.available == ["qwen2.5:7b-instruct"]
        assert isinstance(info.value, RuntimeError)  # what earlier versions raised
