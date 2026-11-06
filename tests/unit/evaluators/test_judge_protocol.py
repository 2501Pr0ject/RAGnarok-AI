"""LLMJudge as a JudgeProtocol: criteria, configuration, traces, parse errors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnarok_ai.core import JudgeProtocol, LLMResponse
from ragnarok_ai.core.exceptions import EvaluationError, JudgeParseError
from ragnarok_ai.evaluators.judge import LLMJudge, parse_judge_response, parse_prometheus_response
from ragnarok_ai.judges import OrdinalBands, OrdinalScore


def make_llm(model: str = "mistral:7b", response: str = "Score: 4\nExplanation: ok") -> MagicMock:
    llm = MagicMock(spec=["generate", "embed", "model", "options", "num_ctx"])
    llm.model = model
    llm.options = {"temperature": 0.0, "seed": 42, "num_ctx": 8192}
    llm.num_ctx = 8192
    llm.generate = AsyncMock(return_value=response)
    return llm


class DetailedLLM:
    """A client with generate_detailed() returning the generic LLMResponse."""

    def __init__(self, text: str, prompt_tokens: int | None = 120, num_ctx: int | None = 8192) -> None:
        self.model = "mistral:7b"
        self.options = {"seed": 42}
        self.num_ctx = num_ctx
        self.text = text
        self.prompt_tokens = prompt_tokens

    async def generate(self, prompt: str) -> str:
        return (await self.generate_detailed(prompt)).text

    async def generate_detailed(self, prompt: str) -> LLMResponse:  # noqa: ARG002
        return LLMResponse(
            self.text, prompt_tokens=self.prompt_tokens, completion_tokens=7, model=self.model, latency_ms=12.5
        )

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


class TestProtocolConformance:
    def test_llmjudge_is_a_judge(self):
        judge = LLMJudge(llm=make_llm())
        assert isinstance(judge, JudgeProtocol)
        assert judge.criteria == frozenset({"faithfulness", "relevance", "hallucination", "completeness"})

    @pytest.mark.asyncio
    async def test_unknown_criterion_raises_value_error(self):
        judge = LLMJudge(llm=make_llm())
        with pytest.raises(ValueError, match="Unknown criterion 'toxicity'"):
            await judge.judge("toxicity", question="q", context="c", answer="a")
        with pytest.raises(ValueError, match="Unknown criterion 'toxicity'"):
            judge.config("toxicity")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "kwargs", "criterion"),
        [
            ("evaluate_faithfulness", {"context": "C", "question": "Q", "answer": "A"}, "faithfulness"),
            ("evaluate_relevance", {"question": "Q", "answer": "A"}, "relevance"),
            ("detect_hallucination", {"context": "C", "answer": "A"}, "hallucination"),
            ("evaluate_completeness", {"question": "Q", "answer": "A", "context": "C"}, "completeness"),
        ],
    )
    async def test_named_methods_are_wrappers_of_judge(self, method, kwargs, criterion):
        """Same prompt whether the named method or judge() is called."""
        llm_a, llm_b = make_llm(), make_llm()
        via_method = await getattr(LLMJudge(llm=llm_a), method)(**kwargs)
        via_judge = await LLMJudge(llm=llm_b).judge(
            criterion, question=kwargs.get("question", ""), context=kwargs.get("context", ""), answer="A"
        )

        assert llm_a.generate.call_args.args[0] == llm_b.generate.call_args.args[0]
        assert via_method.to_dict() == via_judge.to_dict()
        assert via_method.criteria == criterion


class TestConfig:
    def test_fields_describe_the_run(self):
        judge = LLMJudge(llm=make_llm("prometheus2:latest"))
        config = judge.config("faithfulness")

        assert config.judge_class == "ragnarok_ai.evaluators.judge.LLMJudge"
        assert config.judge_kind == "llm"
        assert config.model == "prometheus2:latest"
        assert config.model_requested is None  # client passed by the user
        assert config.backend is None
        assert config.options == (("num_ctx", 8192), ("seed", 42), ("temperature", 0.0))
        assert config.prompt_format == "prometheus"
        assert len(config.prompt_template_sha256 or "") == 64
        assert config.criterion == "faithfulness"
        assert config.unit == "answer" and config.aggregation is None
        assert config.verdict_policy == OrdinalBands(pass_at=4, fail_at=2)
        assert config.verdict_policy.pass_threshold_on_score() == 0.75
        assert config.input_transforms == ()
        assert config.ragnarok_ai_version

    def test_template_hash_depends_on_format_and_criterion_not_on_the_sample(self):
        prometheus = LLMJudge(llm=make_llm("prometheus2:latest"))
        score_n = LLMJudge(llm=make_llm("mistral:7b"))
        h = {
            (fmt, c): j.config(c).prompt_template_sha256
            for fmt, j in (("p", prometheus), ("s", score_n))
            for c in j.criteria
        }
        assert len(set(h.values())) == 8  # 2 formats x 4 criteria, all distinct
        # stable across instances
        assert (
            LLMJudge(llm=make_llm("prometheus2:latest")).config("relevance").prompt_template_sha256
            == h[("p", "relevance")]
        )

    def test_ollama_client_built_by_the_judge_is_described(self):
        from unittest.mock import patch

        with patch.object(LLMJudge, "_get_available_models", return_value=["mistral:latest"]):
            judge = LLMJudge(model="mistral")
        config = judge.config("relevance")

        assert config.backend == "ollama"
        assert config.model == "mistral:latest"  # resolved tag
        assert config.model_requested == "mistral"  # what was asked
        assert config.options == (("num_ctx", 8192), ("seed", 42), ("temperature", 0.0))

    def test_medical_mode_is_an_input_transform_except_for_relevance(self):
        plain = LLMJudge(llm=make_llm())
        medical = LLMJudge(llm=make_llm(), medical_mode=True)

        for criterion in ("faithfulness", "hallucination", "completeness"):
            transforms = medical.config(criterion).input_transforms
            assert len(transforms) == 1 and transforms[0].startswith("medical_normalizer:")
            assert not plain.config(criterion).comparable_with(medical.config(criterion))
        assert medical.config("relevance").input_transforms == ()
        assert plain.config("relevance").comparable_with(medical.config("relevance"))

    def test_disambiguation_model_is_part_of_the_transforms(self):
        disambiguator = MagicMock(spec=["generate", "embed", "model"])
        disambiguator.model = "qwen2.5:0.5b"
        judge = LLMJudge(llm=make_llm(), medical_mode=True, disambiguation_llm=disambiguator)

        transforms = judge.config("faithfulness").input_transforms
        assert len(transforms) == 2 and transforms[1] == "disambiguation:qwen2.5:0.5b"


class TestTrace:
    @pytest.mark.asyncio
    async def test_trace_from_a_detailed_client(self):
        judge = LLMJudge(llm=DetailedLLM("Score: 5\nExplanation: ok"))
        result = await judge.evaluate_relevance(question="Q", answer="A")

        assert result.trace is not None
        assert (result.trace.prompt_tokens, result.trace.completion_tokens) == (120, 7)
        assert result.trace.context_truncated is False
        assert result.trace.latency_ms == 12.5  # reported by the client
        assert result.trace.raw_output is None  # not kept by default

    @pytest.mark.asyncio
    async def test_truncation_is_reported_in_the_trace(self):
        judge = LLMJudge(llm=DetailedLLM("Score: 5\nExplanation: ok", prompt_tokens=4099))
        result = await judge.evaluate_relevance(question="Q", answer="A")

        assert result.trace is not None and result.trace.context_truncated is True
        assert judge.truncation_warnings == 1

    @pytest.mark.asyncio
    async def test_trace_from_a_plain_client(self):
        judge = LLMJudge(llm=make_llm(), keep_raw_output=True)
        result = await judge.evaluate_relevance(question="Q", answer="A")

        assert result.trace is not None
        assert result.trace.prompt_tokens is None and result.trace.context_truncated is None
        assert result.trace.latency_ms is not None and result.trace.latency_ms >= 0.0
        assert result.trace.raw_output == "Score: 4\nExplanation: ok"

    @pytest.mark.asyncio
    async def test_unknown_counters_leave_truncation_unknown(self):
        judge = LLMJudge(llm=DetailedLLM("Score: 5\nExplanation: ok", prompt_tokens=None))
        result = await judge.evaluate_relevance(question="Q", answer="A")

        assert result.trace is not None and result.trace.context_truncated is None


class TestResultSerialization:
    @pytest.mark.asyncio
    async def test_to_dict_keeps_the_five_fields_unless_detailed(self):
        result = await LLMJudge(llm=make_llm()).evaluate_relevance(question="Q", answer="A")

        assert set(result.to_dict()) == {"verdict", "score", "raw_score", "explanation", "criteria"}
        detailed = result.to_dict(detailed=True)
        assert detailed["output"] == {"value": 4, "min": 1, "max": 5, "kind": "ordinal", "type": "OrdinalScore"}
        assert detailed["config"]["criterion"] == "relevance"
        assert detailed["trace"]["type"] == "CallTrace"
        assert detailed["units"] is None


class TestJudgeParseError:
    @pytest.mark.parametrize(
        ("parser", "text"),
        [(parse_judge_response, "no score here"), (parse_prometheus_response, "no marker here")],
    )
    def test_is_an_evaluation_error_with_context(self, parser, text):
        with pytest.raises(EvaluationError) as info:
            parser(text, "relevance")

        assert isinstance(info.value, JudgeParseError)
        assert info.value.criterion == "relevance"
        assert info.value.raw_output == text

    def test_parsers_fill_the_native_output(self):
        assert parse_judge_response("Score: 3\nExplanation: x", "relevance").output == OrdinalScore(3, 1, 5)
        assert parse_prometheus_response("fine [RESULT] 2", "relevance").output == OrdinalScore(2, 1, 5)
