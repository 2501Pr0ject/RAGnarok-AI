"""SentenceLevel: segmentation, per-sentence calls, skipped units, aggregation, configuration."""

from __future__ import annotations

import pytest

from ragnarok_ai.core import JudgeProtocol
from ragnarok_ai.core.exceptions import NoUnitsError
from ragnarok_ai.evaluators.judge import JudgeResult
from ragnarok_ai.judges import (
    AggregationPolicy,
    JudgeConfig,
    OrdinalBands,
    OrdinalScore,
    SentenceLevel,
    judge_many,
)
from ragnarok_ai.text import SEGMENTER_ID


class InnerJudge:
    """Answer-level judge: the score is read from a marker in the sentence, 5 by default."""

    criteria = frozenset({"faithfulness", "relevance"})

    def __init__(self) -> None:
        self.answers: list[str] = []

    def config(self, criterion: str) -> JudgeConfig:
        return JudgeConfig(
            judge_class="tests.InnerJudge",
            judge_kind="custom",
            model="inner",
            criterion=criterion,
            verdict_policy=OrdinalBands(4, 2),
            ragnarok_ai_version="test",
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        self.answers.append(answer)
        raw = 5
        for marker, value in (("(bad)", 1), ("(meh)", 3), ("(ok)", 4)):
            if marker in answer:
                raw = value
        return JudgeResult(
            verdict="PASS" if raw >= 4 else ("FAIL" if raw <= 2 else "PARTIAL"),
            score=(raw - 1) / 4,
            explanation=f"raw {raw}",
            criteria=criterion,
            raw_score=raw,
            output=OrdinalScore(raw, 1, 5),
        )


ANSWER = "The policy is no (Passage 3). Sure!\nThis claim is wrong (bad). This one is fine."


class TestProtocolAndConfig:
    def test_is_a_judge_with_the_inner_criteria(self):
        judge = SentenceLevel(InnerJudge())
        assert isinstance(judge, JudgeProtocol)
        assert judge.criteria == frozenset({"faithfulness", "relevance"})

    def test_config_is_the_inner_one_at_sentence_level(self):
        judge = SentenceLevel(InnerJudge(), aggregation=AggregationPolicy("mean", "from_score"), lang="en")
        config = judge.config("faithfulness")
        inner = InnerJudge().config("faithfulness")

        assert config.unit == "sentence"
        assert config.aggregation == AggregationPolicy("mean", "from_score")
        assert SentenceLevel(InnerJudge()).config("faithfulness").aggregation == AggregationPolicy(
            "fraction_pass", "worst"
        )
        assert config.segmenter == SEGMENTER_ID
        assert config.judge_class == "ragnarok_ai.judges.sentence_level.SentenceLevel[tests.InnerJudge]"
        assert (config.model, config.verdict_policy, config.criterion) == (
            inner.model,
            inner.verdict_policy,
            "faithfulness",
        )
        assert not config.comparable_with(inner)  # unit and aggregation differ
        assert not config.comparable_with(SentenceLevel(InnerJudge()).config("faithfulness"))  # aggregation differs

    @pytest.mark.asyncio
    async def test_unknown_criterion(self):
        with pytest.raises(ValueError, match="Unknown criterion 'toxicity'"):
            await SentenceLevel(InnerJudge()).judge("toxicity", question="q", context="c", answer="A sentence here.")


class TestJudging:
    @pytest.mark.asyncio
    async def test_each_sentence_is_judged_skipped_units_are_kept_not_sent(self):
        inner = InnerJudge()
        result = await SentenceLevel(inner).judge("faithfulness", question="q", context="c", answer=ANSWER)

        assert inner.answers == ["The policy is no (Passage 3).", "This claim is wrong (bad).", "This one is fine."]
        assert result.units is not None and len(result.units) == 4
        texts = [u.text for u in result.units]
        assert texts == ["The policy is no (Passage 3).", "Sure!", "This claim is wrong (bad).", "This one is fine."]
        assert [u.skipped for u in result.units] == [None, "short_fragment", None, None]
        for u in result.units:
            assert ANSWER[u.span[0] : u.span[1]] == u.text
        judged = [u for u in result.units if u.skipped is None]
        assert [u.verdict for u in judged] == ["PASS", "FAIL", "PASS"]
        assert judged[1].output == OrdinalScore(1, 1, 5)

    @pytest.mark.asyncio
    async def test_default_aggregation_is_fraction_pass_and_worst(self):
        result = await SentenceLevel(InnerJudge()).judge("faithfulness", question="q", context="c", answer=ANSWER)

        assert result.score == pytest.approx(2 / 3)  # 2 PASS sentences out of 3 judged
        assert result.verdict == "FAIL"  # worst
        assert result.raw_score is None  # no single native score for an aggregate
        assert result.output is None and result.trace is None
        assert result.config is not None and result.config.unit == "sentence"
        assert result.explanation.splitlines()[0].startswith("[0-29] PASS:")

    @pytest.mark.asyncio
    async def test_fraction_pass_with_from_score(self):
        judge = SentenceLevel(InnerJudge(), aggregation=AggregationPolicy("fraction_pass", "from_score"))
        result = await judge.judge("faithfulness", question="q", context="c", answer=ANSWER)

        assert result.score == pytest.approx(2 / 3)
        assert result.verdict == "PARTIAL"  # OrdinalBands(4, 2): 0.667 is between 0.25 and 0.75
        assert result.raw_score is None

    @pytest.mark.asyncio
    async def test_min_aggregation_remains_available(self):
        judge = SentenceLevel(InnerJudge(), aggregation=AggregationPolicy("min", "worst"))
        result = await judge.judge("faithfulness", question="q", context="c", answer=ANSWER)

        assert (result.score, result.verdict) == (0.0, "FAIL")

    @pytest.mark.asyncio
    async def test_all_units_skipped_is_a_no_units_failure(self):
        judge = SentenceLevel(InnerJudge())
        with pytest.raises(NoUnitsError, match="short_fragment"):
            await judge.judge("faithfulness", question="q", context="c", answer="Sure!\nThanks!")

        attempts = await judge_many(
            judge, "faithfulness", [type("S", (), {"question": "q", "context": "c", "answer": "Sure!"})()]
        )
        assert attempts[0].error is not None and attempts[0].error.kind == "no_units"

    @pytest.mark.asyncio
    async def test_single_sentence_answer_matches_the_inner_judge(self):
        """One sentence under the min aggregation: the wrapper's score and verdict are the inner judge's."""
        inner = InnerJudge()
        direct = await inner.judge("relevance", question="q", context="c", answer="A single sentence (meh).")
        by_min = SentenceLevel(InnerJudge(), aggregation=AggregationPolicy("min", "worst"))
        wrapped = await by_min.judge("relevance", question="q", context="c", answer="A single sentence (meh).")
        assert (wrapped.score, wrapped.verdict) == (direct.score, direct.verdict)
        assert wrapped.raw_score is None and direct.raw_score == 3
        # default fraction_pass: a single PARTIAL sentence scores 0.0 (no PASS sentence), verdict unchanged
        default = await SentenceLevel(InnerJudge()).judge(
            "relevance", question="q", context="c", answer="A single sentence (meh)."
        )
        assert (default.score, default.verdict) == (0.0, "PARTIAL")
