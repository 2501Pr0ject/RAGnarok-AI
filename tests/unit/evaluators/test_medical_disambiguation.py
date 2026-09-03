"""Tests for SLM-based medical abbreviation disambiguation."""

from __future__ import annotations

import pytest

from ragnarok_ai.evaluators.medical.abbreviations import AmbiguousEntry
from ragnarok_ai.evaluators.medical.disambiguation import SLMDisambiguator
from ragnarok_ai.evaluators.medical.medical_normalizer import MedicalAbbreviationNormalizer

MS_CANDIDATES = [
    AmbiguousEntry("multiple sclerosis", ["neuro", "brain", "lesion"], 1),
    AmbiguousEntry("mitral stenosis", ["valve", "heart", "murmur"], 0),
    AmbiguousEntry("morphine sulfate", ["pain", "dose", "iv"], 0),
]


class FakeLLM:
    """Scripted LLMProtocol implementation for tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


class FailingLLM:
    """LLMProtocol implementation whose generate always raises."""

    async def generate(self, prompt: str) -> str:  # noqa: ARG002
        msg = "connection refused"
        raise RuntimeError(msg)

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


class TestSLMDisambiguator:
    """Test suite for SLMDisambiguator."""

    @pytest.mark.asyncio
    async def test_resolves_chosen_candidate(self) -> None:
        """The returned expansion matches the number the model answered."""
        llm = FakeLLM(["2"])
        disambiguator = SLMDisambiguator(llm)

        result = await disambiguator.resolve("MS", MS_CANDIDATES, "severe MS with elevated gradients")

        assert result == "mitral stenosis"
        # The prompt is a closed set: every candidate is offered, numbered
        assert "1. multiple sclerosis" in llm.calls[0]
        assert "2. mitral stenosis" in llm.calls[0]

    @pytest.mark.asyncio
    async def test_verbose_answer_is_parsed(self) -> None:
        """A chatty model answer still resolves via its first number."""
        llm = FakeLLM(["The answer is 3."])
        disambiguator = SLMDisambiguator(llm)

        result = await disambiguator.resolve("MS", MS_CANDIDATES, "MS 4mg IV for pain")

        assert result == "morphine sulfate"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("response", ["0", "7", "no idea", ""])
    async def test_abstains_on_unusable_answer(self, response: str) -> None:
        """Zero, out-of-range, or non-numeric answers mean abstention."""
        disambiguator = SLMDisambiguator(FakeLLM([response]))

        assert await disambiguator.resolve("MS", MS_CANDIDATES, "MS noted") is None

    @pytest.mark.asyncio
    async def test_abstains_on_llm_error(self) -> None:
        """An LLM failure abstains instead of raising."""
        disambiguator = SLMDisambiguator(FailingLLM())

        assert await disambiguator.resolve("MS", MS_CANDIDATES, "MS noted") is None

    @pytest.mark.asyncio
    async def test_abstains_on_empty_candidates(self) -> None:
        """No candidates means nothing to classify."""
        llm = FakeLLM(["1"])
        disambiguator = SLMDisambiguator(llm)

        assert await disambiguator.resolve("MS", [], "MS noted") is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_decisions_are_cached(self) -> None:
        """Same abbreviation + context does not call the LLM twice."""
        llm = FakeLLM(["2"])
        disambiguator = SLMDisambiguator(llm)

        first = await disambiguator.resolve("MS", MS_CANDIDATES, "severe MS with gradients")
        second = await disambiguator.resolve("MS", MS_CANDIDATES, "severe MS with gradients")

        assert first == second == "mitral stenosis"
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_abstentions_are_cached(self) -> None:
        """A cached 'cannot tell' also avoids repeat calls."""
        llm = FakeLLM(["0"])
        disambiguator = SLMDisambiguator(llm)

        assert await disambiguator.resolve("MS", MS_CANDIDATES, "MS noted") is None
        assert await disambiguator.resolve("MS", MS_CANDIDATES, "MS noted") is None
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_different_context_is_a_cache_miss(self) -> None:
        """A different context window triggers a fresh classification."""
        llm = FakeLLM(["2", "1"])
        disambiguator = SLMDisambiguator(llm)

        first = await disambiguator.resolve("MS", MS_CANDIDATES, "severe MS with gradients")
        second = await disambiguator.resolve("MS", MS_CANDIDATES, "MS with white matter lesions")

        assert first == "mitral stenosis"
        assert second == "multiple sclerosis"
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_cache_is_bounded(self) -> None:
        """The cache evicts oldest entries beyond cache_size."""
        llm = FakeLLM(["1"])
        disambiguator = SLMDisambiguator(llm, cache_size=2)

        await disambiguator.resolve("MS", MS_CANDIDATES, "context a")
        await disambiguator.resolve("MS", MS_CANDIDATES, "context b")
        await disambiguator.resolve("MS", MS_CANDIDATES, "context c")

        assert len(disambiguator._cache) == 2


class TestNormalizerEscalation:
    """Test suite for normalize_text_async with a disambiguator."""

    @pytest.mark.asyncio
    async def test_sparse_context_escalates_to_slm(self) -> None:
        """Zero keyword hits hand the choice to the SLM, tagged [slm]."""
        llm = FakeLLM(["2"])  # mitral stenosis
        normalizer = MedicalAbbreviationNormalizer(disambiguator=SLMDisambiguator(llm))

        # No MS context keyword present — keyword scorer has no signal
        normalized, expansions = await normalizer.normalize_text_async("Patient presenting with severe MS today")

        assert "mitral stenosis" in normalized
        assert "MS → mitral stenosis [slm]" in expansions
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_keyword_hit_does_not_call_slm(self) -> None:
        """A confident keyword resolution never escalates."""
        llm = FakeLLM(["2"])
        normalizer = MedicalAbbreviationNormalizer(disambiguator=SLMDisambiguator(llm))

        normalized, expansions = await normalizer.normalize_text_async("Brain MRI shows MS lesion in white matter")

        assert "multiple sclerosis" in normalized
        assert "MS → multiple sclerosis" in expansions
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_slm_abstention_falls_back_to_priority(self) -> None:
        """When the SLM cannot tell, the priority default still applies."""
        llm = FakeLLM(["0"])
        normalizer = MedicalAbbreviationNormalizer(disambiguator=SLMDisambiguator(llm))

        normalized, expansions = await normalizer.normalize_text_async("Patient presenting with severe MS today")

        # Highest-priority meaning, same as without a disambiguator
        assert "multiple sclerosis" in normalized
        assert "MS → multiple sclerosis" in expansions
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_unambiguous_abbreviations_do_not_escalate(self) -> None:
        """Plain dictionary entries never involve the SLM."""
        llm = FakeLLM(["1"])
        normalizer = MedicalAbbreviationNormalizer(disambiguator=SLMDisambiguator(llm))

        normalized, _expansions = await normalizer.normalize_text_async("Patient has CHF and MI")

        assert "congestive heart failure" in normalized
        assert "myocardial infarction" in normalized
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_async_without_disambiguator_matches_sync(self) -> None:
        """normalize_text_async is a strict superset: without a strategy it equals normalize_text."""
        normalizer = MedicalAbbreviationNormalizer()

        text = "Patient with CHF, s/p CABG, presenting with MS today"
        assert await normalizer.normalize_text_async(text) == normalizer.normalize_text(text)


class TestResolveConfidence:
    """Test suite for _resolve_with_confidence."""

    def test_unambiguous_is_confident(self) -> None:
        normalizer = MedicalAbbreviationNormalizer()

        assert normalizer._resolve_with_confidence("CHF", "") == ("congestive heart failure", True)

    def test_keyword_hit_is_confident(self) -> None:
        normalizer = MedicalAbbreviationNormalizer()

        full_form, confident = normalizer._resolve_with_confidence("MS", "brain lesion on MRI")

        assert full_form == "multiple sclerosis"
        assert confident is True

    def test_zero_hits_is_not_confident(self) -> None:
        normalizer = MedicalAbbreviationNormalizer()

        full_form, confident = normalizer._resolve_with_confidence("MS", "patient presenting today")

        assert full_form == "multiple sclerosis"  # priority fallback
        assert confident is False

    def test_unknown_abbreviation(self) -> None:
        normalizer = MedicalAbbreviationNormalizer()

        assert normalizer._resolve_with_confidence("XYZ", "anything") == (None, True)


class TestEvaluatorWiring:
    """The disambiguation_llm parameter reaches the normalizer."""

    @pytest.mark.asyncio
    async def test_faithfulness_evaluator_uses_slm(self) -> None:
        from ragnarok_ai.evaluators.faithfulness import FaithfulnessEvaluator

        judge_llm = FakeLLM(
            [
                '["Patient has mitral stenosis"]',
                '{"supported": true, "reasoning": "Matches context"}',
            ]
        )
        slm = FakeLLM(["2"])  # mitral stenosis
        evaluator = FaithfulnessEvaluator(judge_llm, medical_mode=True, disambiguation_llm=slm)

        result = await evaluator.evaluate_detailed(
            response="Patient has mitral stenosis",
            context="Patient presenting with severe MS today",
        )

        assert result.score == 1.0
        assert len(slm.calls) == 1

    def test_judge_wires_disambiguator(self) -> None:
        from ragnarok_ai.evaluators.judge import LLMJudge

        judge = LLMJudge(llm=FakeLLM(["5"]), medical_mode=True, disambiguation_llm=FakeLLM(["1"]))

        assert judge._normalizer is not None
        assert judge._normalizer._disambiguator is not None
