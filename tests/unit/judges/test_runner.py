"""Tests for judge_many: order, identifiers, failures and configuration errors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from ragnarok_ai.calibration import CalibrationSample
from ragnarok_ai.core.exceptions import (
    EvaluationError,
    JudgeDependencyError,
    JudgeModelNotFoundError,
    JudgeParseError,
    LLMConnectionError,
)
from ragnarok_ai.evaluators.judge import JudgeResult
from ragnarok_ai.judges import CallTrace, JudgeConfig, OrdinalBands, OrdinalScore, judge_many, sample_identifier


@dataclass
class Sample:
    question: str
    context: str
    answer: str
    sample_id: str | None = None


class ScriptedJudge:
    """A JudgeProtocol judge whose behaviour per sample is scripted by the answer text."""

    criteria = frozenset({"faithfulness", "relevance"})

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[str] = []
        self.delay = delay

    def config(self, criterion: str) -> JudgeConfig:
        if criterion not in self.criteria:
            raise ValueError(criterion)
        return JudgeConfig(
            judge_class="tests.ScriptedJudge",
            judge_kind="custom",
            model="scripted",
            criterion=criterion,
            verdict_policy=OrdinalBands(4, 2),
            ragnarok_ai_version="test",
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        self.calls.append(answer)
        if self.delay:
            await asyncio.sleep(self.delay)
        if answer == "parse":
            raise JudgeParseError("Unparseable faithfulness judgment", criterion=criterion, raw_output="garbage")
        if answer == "timeout":
            raise LLMConnectionError("Request to Ollama timed out after 300.0s")
        if answer == "backend":
            raise LLMConnectionError("Ollama API error: 500 - boom")
        if answer == "asyncio-timeout":
            raise asyncio.TimeoutError
        if answer == "eval":
            raise EvaluationError("something else went wrong")
        if answer == "config":
            raise JudgeModelNotFoundError("m", [], "http://x")
        raw = int(answer) if answer.isdigit() else 4
        return JudgeResult(
            verdict="PASS" if raw >= 4 else "FAIL",
            score=(raw - 1) / 4,
            explanation="ok",
            criteria=criterion,
            raw_score=raw,
            output=OrdinalScore(raw, 1, 5),
            trace=CallTrace(prompt_tokens=10, raw_output="raw text"),
        )


def samples(*answers: str) -> list[Sample]:
    return [Sample("q", "c", a) for a in answers]


class TestOrderAndIdentifiers:
    @pytest.mark.asyncio
    async def test_attempts_follow_sample_order_and_positions(self):
        judge = ScriptedJudge()
        attempts = await judge_many(judge, "faithfulness", samples("5", "parse", "2"))

        assert [a.sample_id for a in attempts] == ["0", "1", "2"]
        assert [a.ok for a in attempts] == [True, False, True]
        assert all(a.criterion == "faithfulness" for a in attempts)
        assert all(a.config == judge.config("faithfulness") for a in attempts)

    @pytest.mark.asyncio
    async def test_explicit_sample_ids_are_kept(self):
        items = [Sample("q", "c", "5", sample_id="p012"), Sample("q", "c", "3")]
        attempts = await judge_many(ScriptedJudge(), "relevance", items)
        assert [a.sample_id for a in attempts] == ["p012", "1"]

    def test_sample_identifier_fallbacks(self):
        assert sample_identifier(Sample("q", "c", "a", sample_id=""), 7) == "7"
        assert sample_identifier(CalibrationSample(question="q", context="c", answer="a", human_pass=True), 3) == "3"
        assert (
            sample_identifier(
                CalibrationSample(question="q", context="c", answer="a", human_pass=True, sample_id="s1"), 3
            )
            == "s1"
        )

    @pytest.mark.asyncio
    async def test_concurrency_preserves_order(self):
        judge = ScriptedJudge(delay=0.01)
        attempts = await judge_many(judge, "faithfulness", samples("1", "2", "3", "4", "5"), concurrency=5)
        assert [a.result.raw_score for a in attempts if a.result] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_invalid_concurrency_and_criterion(self):
        with pytest.raises(ValueError, match="concurrency"):
            await judge_many(ScriptedJudge(), "faithfulness", samples("5"), concurrency=0)
        with pytest.raises(ValueError, match="Unknown criterion 'toxicity'"):
            await judge_many(ScriptedJudge(), "toxicity", samples("5"))


class TestFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("answer", "kind", "fragment"),
        [
            ("parse", "parse", "Unparseable"),
            ("timeout", "timeout", "timed out"),
            ("asyncio-timeout", "timeout", "timed out"),
            ("backend", "backend", "500"),
            ("eval", "backend", "something else"),
        ],
    )
    async def test_per_sample_failures_become_judge_failures(self, answer, kind, fragment):
        attempts = await judge_many(ScriptedJudge(), "faithfulness", samples("5", answer, "5"))

        failed = attempts[1]
        assert failed.ok is False and failed.result is None and failed.error is not None
        assert failed.error.kind == kind
        assert fragment in failed.error.message
        assert attempts[0].ok and attempts[2].ok  # the run continues

    @pytest.mark.asyncio
    async def test_raw_output_is_dropped_unless_requested(self):
        kept = await judge_many(ScriptedJudge(), "faithfulness", samples("5", "parse"), keep_raw_output=True)
        dropped = await judge_many(ScriptedJudge(), "faithfulness", samples("5", "parse"))

        assert kept[0].trace is not None and kept[0].trace.raw_output == "raw text"
        assert kept[1].error is not None and kept[1].error.raw_output == "garbage"
        assert dropped[0].trace is not None and dropped[0].trace.raw_output is None
        assert dropped[0].trace.prompt_tokens == 10  # the rest of the trace is kept
        assert dropped[1].error is not None and dropped[1].error.raw_output is None

    @pytest.mark.asyncio
    async def test_configuration_errors_are_never_caught(self):
        judge = ScriptedJudge()
        with pytest.raises(JudgeModelNotFoundError):
            await judge_many(judge, "faithfulness", samples("5", "config", "5"))
        assert judge.calls == ["5", "config"]  # stopped at the failing call

    @pytest.mark.asyncio
    async def test_configuration_error_in_config_stops_before_any_call(self):
        class BrokenConfig(ScriptedJudge):
            def config(self, criterion: str) -> JudgeConfig:  # noqa: ARG002
                raise JudgeDependencyError()

        judge = BrokenConfig()
        with pytest.raises(JudgeDependencyError):
            await judge_many(judge, "faithfulness", samples("5"))
        assert judge.calls == []
