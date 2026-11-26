"""FaithfulnessEvaluator and HallucinationDetector under the judge contract."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from ragnarok_ai.benchmarks.judges import attempt_to_dict, load_records, run_benchmark
from ragnarok_ai.calibration import CalibrationSample, JudgeCalibrator
from ragnarok_ai.cli.main import app
from ragnarok_ai.core import JudgeProtocol
from ragnarok_ai.core.exceptions import ConfigurationError, NoUnitsError
from ragnarok_ai.evaluators import FaithfulnessEvaluator, HallucinationDetector
from ragnarok_ai.evaluators import faithfulness as faithfulness_module
from ragnarok_ai.judges import (
    CLAIM_AGGREGATION,
    AggregationPolicy,
    ClaimLevelJudge,
    ScoreThreshold,
    UnitJudgment,
    aggregate_units,
    judge_many,
)
from tests.unit.judges.claim_judges import ScriptedLLM

if TYPE_CHECKING:
    from pathlib import Path

CONTEXT = "Paris is the capital of France. The Eiffel Tower is in Paris."
MIXED = "Paris is the capital of France. The tower is 900 meters tall (unsupported). The tower is in Paris."
ALL_GOOD = "Paris is the capital of France. The Eiffel Tower is in Paris."


class TestFaithfulnessResult:
    @pytest.mark.asyncio
    async def test_round_trip_to_a_detailed_dictionary(self):
        evaluator = FaithfulnessEvaluator(ScriptedLLM(options={"temperature": 0, "seed": 7}))
        result = await evaluator.evaluate_detailed(MIXED, CONTEXT)
        judged = result.to_judge_result(evaluator.config())

        assert (judged.criteria, judged.raw_score, judged.output) == ("faithfulness", None, None)
        assert judged.score == result.score == pytest.approx(2 / 3)  # the share of supported claims, unchanged
        assert judged.verdict == "FAIL"  # one unsupported claim is enough
        assert judged.units is not None and len(judged.units) == 3
        assert [(u.verdict, u.score, u.label, u.span) for u in judged.units] == [
            ("PASS", 1.0, "HOLDS", None),
            ("FAIL", 0.0, "FAILS", None),
            ("PASS", 1.0, "HOLDS", None),
        ]
        assert judged.units[1].text == "The tower is 900 meters tall (unsupported)."
        assert judged.explanation.splitlines()[0] == "2 of 3 claim(s) hold."
        assert "[FAIL] The tower is 900 meters tall (unsupported).: not in the context" in judged.explanation

        detailed = judged.to_dict(detailed=True)
        text = json.dumps(detailed)  # serializable as it is, None spans included
        assert json.loads(text)["units"][0]["span"] is None
        assert detailed["config"]["unit"] == "claim" and detailed["config"]["criterion"] == "faithfulness"
        assert {k: detailed[k] for k in ("verdict", "raw_score", "criteria")} == {
            "verdict": "FAIL",
            "raw_score": None,
            "criteria": "faithfulness",
        }

    @pytest.mark.asyncio
    async def test_every_claim_supported_passes(self):
        evaluator = FaithfulnessEvaluator(ScriptedLLM())
        judged = (await evaluator.evaluate_detailed(ALL_GOOD, CONTEXT)).to_judge_result()
        assert (judged.verdict, judged.score, judged.config) == ("PASS", 1.0, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("response", ["", "   "])
    async def test_an_answer_without_a_claim_is_not_scored(self, response: str):
        evaluator = FaithfulnessEvaluator(ScriptedLLM())
        result = await evaluator.evaluate_detailed(response, CONTEXT)
        assert result.score == 1.0 and result.claims == []  # the evaluator's own convention, unchanged
        with pytest.raises(NoUnitsError, match="No claim to judge: Empty response"):
            result.to_judge_result()

    @pytest.mark.asyncio
    async def test_nothing_extracted_is_not_scored(self):
        class Silent(ScriptedLLM):
            async def generate(self, prompt: str) -> str:
                return "[]" if prompt.startswith("Extract") else await super().generate(prompt)

        result = await FaithfulnessEvaluator(Silent()).evaluate_detailed("Hello there.", CONTEXT)
        with pytest.raises(NoUnitsError, match="No verifiable claims"):
            result.to_judge_result()


class TestHallucinationResult:
    @pytest.mark.asyncio
    async def test_score_is_inverted_and_the_criterion_says_so(self):
        detector = HallucinationDetector(ScriptedLLM())
        result = await detector.evaluate_detailed(MIXED, CONTEXT)
        # the detector's own result: share of hallucinated claims, 0 is good
        assert result.score == pytest.approx(1 / 3) and result.total_claims == 3
        assert [h.claim for h in result.hallucinations] == ["The tower is 900 meters tall (unsupported)."]

        judged = result.to_judge_result(detector.config())
        assert judged.criteria == "hallucination_free" == HallucinationDetector.JUDGE_CRITERION
        assert judged.score == pytest.approx(1 - result.score) == pytest.approx(2 / 3)  # higher is better
        assert judged.verdict == "FAIL"
        assert [u.verdict for u in judged.units] == ["PASS", "FAIL", "PASS"]  # type: ignore[union-attr]
        assert dict(judged.config.judge_specific)["score"].startswith("share of claims not hallucinated")  # type: ignore[union-attr]
        json.dumps(judged.to_dict(detailed=True))

    @pytest.mark.asyncio
    async def test_the_result_keeps_every_claim_with_its_status(self):
        result = await HallucinationDetector(ScriptedLLM()).evaluate_detailed(MIXED, CONTEXT)
        assert [(c.claim, c.hallucinated, c.reason) for c in result.claims] == [
            ("Paris is the capital of France.", False, "in the context"),
            ("The tower is 900 meters tall (unsupported).", True, "not in the context"),
            ("The tower is in Paris.", False, "in the context"),
        ]
        assert len(result.claims) == result.total_claims
        dumped = result.model_dump()
        assert set(dumped) == {"score", "hallucinations", "total_claims", "reasoning", "claims"}
        # the three shortcuts of evaluate_detailed leave the list empty, as before for the other fields
        empty = await HallucinationDetector(ScriptedLLM()).evaluate_detailed("", CONTEXT)
        assert (empty.score, empty.claims, empty.total_claims) == (0.0, [], 0)

    @pytest.mark.asyncio
    async def test_no_hallucination_passes_and_no_claim_is_not_scored(self):
        detector = HallucinationDetector(ScriptedLLM())
        clean = (await detector.evaluate_detailed(ALL_GOOD, CONTEXT)).to_judge_result()
        assert (clean.verdict, clean.score) == ("PASS", 1.0)
        with pytest.raises(NoUnitsError, match="No claim to judge"):
            (await detector.evaluate_detailed("", CONTEXT)).to_judge_result()


class TestJudgePath:
    @pytest.mark.parametrize(
        ("evaluator_class", "criterion"),
        [(FaithfulnessEvaluator, "faithfulness"), (HallucinationDetector, "hallucination_free")],
    )
    def test_is_a_judge_with_one_criterion(self, evaluator_class, criterion: str):
        judge = evaluator_class(ScriptedLLM()).as_judge()
        assert isinstance(judge, ClaimLevelJudge) and isinstance(judge, JudgeProtocol)
        assert judge.criteria == frozenset({criterion})
        assert judge.config(criterion).criterion == criterion
        with pytest.raises(ValueError, match="Unknown criterion 'relevance'"):
            judge.config("relevance")

    @pytest.mark.asyncio
    async def test_judge_gives_the_converted_result_with_a_latency(self):
        llm = ScriptedLLM()
        judge = FaithfulnessEvaluator(llm).as_judge()
        result = await judge.judge("faithfulness", question="unused", context=CONTEXT, answer=MIXED)
        assert (result.verdict, result.score) == ("FAIL", pytest.approx(2 / 3))
        assert result.config == judge.config("faithfulness")
        assert result.trace is not None and result.trace.latency_ms is not None and result.trace.prompt_tokens is None
        assert (llm.extractions, llm.checks) == (1, 3)
        with pytest.raises(ValueError, match="Unknown criterion"):
            await judge.judge("hallucination", question="", context=CONTEXT, answer=MIXED)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context", ["", "  \n "])
    @pytest.mark.parametrize(
        ("evaluator_class", "criterion"),
        [(FaithfulnessEvaluator, "faithfulness"), (HallucinationDetector, "hallucination_free")],
    )
    async def test_empty_context_fails_every_claim_without_a_check_call(self, evaluator_class, criterion, context):
        llm = ScriptedLLM()
        evaluator = evaluator_class(llm)
        result = await evaluator.as_judge().judge(criterion, question="", context=context, answer=ALL_GOOD)
        # not a no-units case: the claims are there, nothing supports them
        assert (result.verdict, result.score) == ("FAIL", 0.0)
        assert [(u.text, u.verdict, u.span) for u in result.units] == [  # type: ignore[union-attr]
            ("Paris is the capital of France.", "FAIL", None),
            ("The Eiffel Tower is in Paris.", "FAIL", None),
        ]
        assert "No context provided" in result.explanation
        assert (llm.extractions, llm.checks) == (1, 0)  # one extraction, no verification call

        # evaluate_detailed keeps its shortcut: a score, no claim, no call at all
        llm2 = ScriptedLLM()
        detailed = await evaluator_class(llm2).evaluate_detailed(ALL_GOOD, context)
        assert (llm2.extractions, llm2.checks) == (0, 0)
        assert detailed.score == (0.0 if criterion == "faithfulness" else 1.0)
        assert detailed.claims == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("evaluator_class", "criterion"),
        [(FaithfulnessEvaluator, "faithfulness"), (HallucinationDetector, "hallucination_free")],
    )
    async def test_no_claim_is_a_no_units_failure_in_a_run_not_a_score(self, evaluator_class, criterion: str):
        judge = evaluator_class(ScriptedLLM()).as_judge()
        samples = [
            SimpleNamespace(question="q", context=CONTEXT, answer=ALL_GOOD, sample_id="ok"),
            SimpleNamespace(question="q", context=CONTEXT, answer="", sample_id="empty-answer"),
            SimpleNamespace(question="q", context="", answer="", sample_id="both-empty"),
            SimpleNamespace(question="q", context="", answer=ALL_GOOD, sample_id="empty-context"),
        ]
        attempts = await judge_many(judge, criterion, samples)
        assert [a.ok for a in attempts] == [True, False, False, True]
        assert [a.error.kind for a in attempts if a.error] == ["no_units", "no_units"]
        assert attempts[3].result is not None and attempts[3].result.verdict == "FAIL"
        row = attempt_to_dict(attempts[0])
        json.dumps(row)
        assert row["result"]["score"] == 1.0 and row["result"]["raw_score"] is None


class TestConfiguration:
    def test_model_name_is_required_when_the_llm_does_not_expose_one(self):
        anonymous = ScriptedLLM(model=None)
        assert not hasattr(anonymous, "model")
        for evaluator in (FaithfulnessEvaluator(anonymous), HallucinationDetector(anonymous)):
            with pytest.raises(ConfigurationError, match=r'does not expose a model name.*as_judge\(model="<name>"\)'):
                evaluator.as_judge()
            with pytest.raises(ConfigurationError, match="does not expose a model name"):
                evaluator.config()
            judge = evaluator.as_judge(model="my-model:7b")
            assert judge.config(evaluator.JUDGE_CRITERION).model == "my-model:7b"

    def test_an_explicit_name_wins_over_the_llm_s(self):
        evaluator = FaithfulnessEvaluator(ScriptedLLM(model="from-the-llm"))
        assert evaluator.config().model == "from-the-llm"
        assert evaluator.as_judge(model="explicit").config("faithfulness").model == "explicit"

    def test_temperature_is_recorded_when_available(self):
        from_options = FaithfulnessEvaluator(ScriptedLLM(options={"temperature": 0, "seed": 7})).config()
        assert from_options.options == (("seed", 7), ("temperature", 0))

        llm = ScriptedLLM()
        llm.temperature = 0.2  # type: ignore[attr-defined]
        assert FaithfulnessEvaluator(llm).config().options == (("temperature", 0.2),)
        # the options' own value is kept when both exist
        llm.options = {"temperature": 0.7}  # type: ignore[attr-defined]
        assert FaithfulnessEvaluator(llm).config().options == (("temperature", 0.7),)
        assert FaithfulnessEvaluator(ScriptedLLM()).config().options == ()

    def test_fields_of_the_configuration(self):
        config = FaithfulnessEvaluator(ScriptedLLM()).config("faithfulness")
        assert config.judge_class == "ragnarok_ai.evaluators.faithfulness.FaithfulnessEvaluator"
        assert (config.judge_kind, config.model, config.unit, config.prompt_format) == (
            "llm",
            "scripted-1",
            "claim",
            "claims-json",
        )
        assert config.aggregation == CLAIM_AGGREGATION == AggregationPolicy("fraction_pass", "worst")
        assert config.verdict_policy == ScoreThreshold(pass_threshold=1.0)
        assert config.verdict_policy.pass_threshold_on_score() == 1.0
        assert config.segmenter is not None and config.segmenter.startswith("llm-claim-extraction:")
        assert config.prompt_template_sha256 is not None and len(config.prompt_template_sha256) == 64
        assert config.input_transforms == () and config.model_digest is None
        json.dumps(config.to_dict())
        with pytest.raises(ValueError, match="Unknown criterion 'hallucination'"):
            FaithfulnessEvaluator(ScriptedLLM()).config("hallucination")
        with pytest.raises(ValueError, match="Unknown criterion 'hallucination'"):
            HallucinationDetector(ScriptedLLM()).config("hallucination")  # the name is hallucination_free

    def test_runs_are_comparable_only_with_the_same_setup(self, monkeypatch: pytest.MonkeyPatch):
        base = FaithfulnessEvaluator(ScriptedLLM()).config()
        assert base.comparable_with(FaithfulnessEvaluator(ScriptedLLM()).config())
        assert not base.comparable_with(FaithfulnessEvaluator(ScriptedLLM(model="other")).config())
        assert not base.comparable_with(FaithfulnessEvaluator(ScriptedLLM(options={"temperature": 0.5})).config())
        medical = FaithfulnessEvaluator(ScriptedLLM(), medical_mode=True).config()
        assert medical.input_transforms and medical.input_transforms[0].startswith("medical_normalizer:")
        assert not base.comparable_with(medical)
        # the two evaluators are two judges
        assert not base.comparable_with(HallucinationDetector(ScriptedLLM()).config())
        # a change in either prompt changes the configuration
        monkeypatch.setattr(faithfulness_module, "CLAIM_VERIFICATION_PROMPT", "Another prompt {claim} {context}")
        changed = FaithfulnessEvaluator(ScriptedLLM()).config()
        assert changed.prompt_template_sha256 != base.prompt_template_sha256 and changed.segmenter == base.segmenter
        monkeypatch.setattr(faithfulness_module, "CLAIM_EXTRACTION_PROMPT", "Another extraction {response}")
        assert FaithfulnessEvaluator(ScriptedLLM()).config().segmenter != base.segmenter


class TestUnitsWithoutASpan:
    def test_a_unit_may_have_no_span_and_a_span_is_still_checked(self):
        unit = UnitJudgment(text="A claim.", span=None, score=1.0, verdict="PASS")
        assert unit.span is None
        assert UnitJudgment(text="A sentence.", span=(0, 11), score=1.0, verdict="PASS").span == (0, 11)
        with pytest.raises(ValueError, match="span must satisfy"):
            UnitJudgment(text="x", span=(5, 2), score=1.0, verdict="PASS")
        with pytest.raises(ValueError, match="needs a score unless it is skipped"):
            UnitJudgment(text="x", span=None)
        skipped = UnitJudgment(text="x", span=None, skipped="not a factual claim")
        assert skipped.skipped == "not a factual claim"

    def test_aggregation_with_units_without_a_span(self):
        units = [
            UnitJudgment(text="a", span=None, score=1.0, verdict="PASS"),
            UnitJudgment(text="b", span=None, score=0.0, verdict="FAIL"),
            UnitJudgment(text="c", span=None, skipped="skipped"),
        ]
        assert aggregate_units(units, CLAIM_AGGREGATION, ScoreThreshold(1.0)) == (0.5, "FAIL")
        # mixed with units that have one: nothing reads the span to aggregate
        mixed = [units[0], UnitJudgment(text="d", span=(3, 4), score=1.0, verdict="PASS")]
        assert aggregate_units(mixed, CLAIM_AGGREGATION, ScoreThreshold(1.0)) == (1.0, "PASS")
        # the error that used to print the span names the unit by its text
        broken = [UnitJudgment(text="a claim without a verdict", span=None, score=1.0)]
        with pytest.raises(ValueError, match="Judged unit 'a claim without a verdict' has no score or no verdict"):
            aggregate_units(broken, CLAIM_AGGREGATION, ScoreThreshold(1.0))
        with pytest.raises(ValueError, match=r"Judged unit \(3, 9\) has no score or no verdict"):
            aggregate_units([UnitJudgment(text="x", span=(3, 9), score=1.0)], CLAIM_AGGREGATION, ScoreThreshold(1.0))


def answer_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(6):
        rows.append(
            {
                "id": f"p{i}",
                "group_id": f"g{i}",
                "question": "q",
                "context": CONTEXT,
                "answer": ALL_GOOD,
                "label": "PASS",
            }
        )
    for i in range(5):
        rows.append(
            {"id": f"f{i}", "group_id": f"h{i}", "question": "q", "context": CONTEXT, "answer": MIXED, "label": "FAIL"}
        )
    rows.append(
        {
            "id": "empty",
            "group_id": "e",
            "question": "q",
            "context": CONTEXT,
            "answer": "Hello there (no claims).",
            "label": "FAIL",
        }
    )
    return rows


class TestInTheHarnessAndTheCalibrator:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("evaluator_class", "criterion"),
        [(FaithfulnessEvaluator, "faithfulness"), (HallucinationDetector, "hallucination_free")],
    )
    async def test_judge_bench_on_an_answer_level_set(self, tmp_path: Path, evaluator_class, criterion: str):
        data = tmp_path / "answers.jsonl"
        rows = answer_rows()[:-1]
        data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        records = load_records(data)
        run = await run_benchmark(
            evaluator_class(ScriptedLLM()).as_judge(),
            records,
            out_dir=tmp_path / "out",
            criterion=criterion,
            bootstrap_resamples=50,
        )
        report = run.report
        assert report["level"] == "answer" and report["criterion"] == criterion
        assert report["records"] == {"total": 11, "evaluated": 11, "scored": 11, "failed": 0}
        # PASS at a score of 1.0: the six clean answers pass, the five with one bad claim do not
        assert report["threshold"]["value"] == 1.0 and report["threshold"]["source"] == "verdict_policy"
        assert report["binary"]["accuracy"] == 1.0 and report["binary"]["kappa"] == 1.0
        assert report["config"]["unit"] == "claim" and report["config"]["aggregation"]["verdict_rule"] == "worst"
        assert report["recall_by_label"] is None  # an answer-level set
        md = run.report_md.read_text()
        assert f"# Judge benchmark: {evaluator_class.__name__}" in md or "# Judge benchmark:" in md
        assert "Threshold 1.00 (source: verdict_policy)" in md
        written = [json.loads(line) for line in run.attempts_path.read_text().splitlines()[1:]]
        assert len(written) == 11 and all(row["ok"] for row in written)
        assert sorted({round(row["result"]["score"], 6) for row in written}) == [round(2 / 3, 6), 1.0]

    def test_judge_bench_from_the_command_line(self, tmp_path: Path):
        data = tmp_path / "answers.jsonl"
        data.write_text("".join(json.dumps(r) + "\n" for r in answer_rows()), encoding="utf-8")
        out = tmp_path / "out"
        result = CliRunner().invoke(
            app,
            [
                "--json",
                "judge-bench",
                "--judge",
                "python:tests.unit.judges.claim_judges:ScriptedFaithfulnessJudge",
                "--data",
                str(data),
                "--out",
                str(out),
                "--bootstrap",
                "20",
            ],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)["data"]["report"]
        # the answer without a claim is a recorded failure, not a score
        assert report["records"] == {"total": 12, "evaluated": 12, "scored": 11, "failed": 1}
        assert report["failures"] == {"no_units": 1}
        assert (out / "report.md").exists() and "failures by kind: {'no_units': 1}" in (out / "report.md").read_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("evaluator_class", "criterion"),
        [(FaithfulnessEvaluator, "faithfulness"), (HallucinationDetector, "hallucination_free")],
    )
    async def test_calibrator_runs_the_judge_on_its_own_criterion(self, evaluator_class, criterion: str):
        samples = [
            CalibrationSample(question="q", context=CONTEXT, answer=ALL_GOOD, human_pass=True, sample_id=f"p{i}")
            for i in range(6)
        ] + [
            CalibrationSample(question="q", context=CONTEXT, answer=MIXED, human_pass=False, sample_id=f"f{i}")
            for i in range(6)
        ]
        calibrator = JudgeCalibrator(evaluator_class(ScriptedLLM()).as_judge(), min_samples=1)
        assert calibrator.uses_protocol is True and calibrator.threshold_for(criterion) == 1.0
        report = await calibrator.calibrate(samples, criteria=[criterion])
        calibration = report.criteria[criterion]
        assert (calibration.threshold, calibration.accuracy, calibration.kappa) == (1.0, 1.0, 1.0)
        json.dumps(report.to_dict() if hasattr(report, "to_dict") else report.model_dump(), default=str)
        # a criterion the judge does not have is refused by name, with the judge's own list
        with pytest.raises(ValueError, match=rf"Unknown criterion 'relevance'; expected one of \('{criterion}',\)"):
            await calibrator.calibrate(samples, criteria=["relevance"])
