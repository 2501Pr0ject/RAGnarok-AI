"""Judge benchmark harness: records, sampling, resume, thresholds, metrics, reports, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragnarok_ai.benchmarks.judges import (
    binary_metrics,
    calibration_metrics,
    choose_threshold,
    grouped_bootstrap,
    load_records,
    parse_record,
    read_attempts,
    run_benchmark,
    stratified_sample,
    three_class_metrics,
)
from ragnarok_ai.cli.main import app
from tests.unit.benchmarks.fake_judge import ScriptedOrdinalJudge, ScriptedProbabilityJudge, ScriptedThreeClassJudge

runner = CliRunner()


def sentence_row(i: int, label: str, sentence: str, **extra: object) -> dict[str, object]:
    return {
        "id": f"rt-{i}",
        "source_id": f"src-{i // 2}",
        "response_id": f"resp-{i}",
        "task": "QA" if i % 2 == 0 else "Summary",
        "source_dataset": "fixture",
        "generator": "gen-a" if i % 3 else "gen-b",
        "question": f"Question {i}?",
        "context": f"Context {i}.",
        "prev_sentence": None if i % 4 == 0 else f"Previous sentence {i}.",
        "sentence": sentence,
        "label": label,
        "rule": "fixture",
        **extra,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def study_like_rows() -> list[dict[str, object]]:
    """62 sentences: 58 supported, 4 not; the ordinal judge fails 24 supported and passes 2 unsupported."""
    rows: list[dict[str, object]] = []
    i = 0
    for _ in range(34):
        rows.append(sentence_row(i, "SUPPORTED", f"Supported sentence {i}."))
        i += 1
    for _ in range(24):
        rows.append(sentence_row(i, "SUPPORTED", f"Supported but rejected {i} (raw 1)."))
        i += 1
    for _ in range(2):
        rows.append(sentence_row(i, "UNSUPPORTED", f"Unsupported but accepted {i}."))
        i += 1
    for _ in range(2):
        rows.append(sentence_row(i, "CONTRADICTED", f"Contradicted and rejected {i} (raw 1)."))
        i += 1
    return rows


class TestRecords:
    def test_sentence_record_is_normalized(self):
        rec = parse_record(sentence_row(3, "unsupported", "A sentence."), 1)
        assert (rec.level, rec.label, rec.human_pass, rec.answer) == ("sentence", "UNSUPPORTED", False, "A sentence.")
        assert rec.group_id == "src-1"  # source_id wins over response_id
        assert rec.has_prev_sentence is True
        assert (
            rec.metadata["task"] == "Summary" and "sentence" not in rec.metadata and "prev_sentence" not in rec.metadata
        )

    def test_group_fallbacks_and_answer_level(self):
        rec = parse_record({"id": "a1", "group_id": "g9", "answer": "Full answer.", "label": "pass", "context": "c"}, 1)
        assert (rec.level, rec.group_id, rec.human_pass, rec.question) == ("answer", "g9", True, "")
        assert parse_record({"id": "a2", "answer": "x", "label": "FAIL"}, 1).group_id == "a2"

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"sentence": "x", "label": "SUPPORTED"}, "missing 'id'"),
            ({"id": "1", "label": "SUPPORTED"}, "needs 'sentence'"),
            ({"id": "1", "sentence": "x", "label": "PASS"}, "not in"),
            ({"id": "1", "answer": "x", "label": "SUPPORTED"}, "not in"),
            ({"id": "1", "sentence": "", "label": "SUPPORTED"}, "empty text"),
        ],
    )
    def test_invalid_records(self, row, message):
        with pytest.raises(ValueError, match=message):
            parse_record(row, 7)

    def test_load_rejects_mixed_levels_and_duplicates(self, tmp_path: Path):
        mixed = write_jsonl(
            tmp_path / "mixed.jsonl", [sentence_row(0, "SUPPORTED", "s"), {"id": "x", "answer": "a", "label": "PASS"}]
        )
        with pytest.raises(ValueError, match="mixed levels"):
            load_records(mixed)
        dup = write_jsonl(
            tmp_path / "dup.jsonl", [sentence_row(0, "SUPPORTED", "s"), sentence_row(0, "SUPPORTED", "t")]
        )
        with pytest.raises(ValueError, match="duplicate ids"):
            load_records(dup)
        with pytest.raises(ValueError, match="no record"):
            load_records(write_jsonl(tmp_path / "empty.jsonl", []))


class TestSampling:
    def test_stratified_sample_is_deterministic_and_proportional(self):
        records = load_records_from_rows(study_like_rows())
        a = stratified_sample(records, 31, "label", seed=3)
        b = stratified_sample(records, 31, "label", seed=3)
        assert [r.id for r in a] == [r.id for r in b]
        labels = [r.label for r in a]
        assert labels.count("SUPPORTED") == 29 and len(a) == 31  # 58/62 * 31 = 29
        assert [r.id for r in stratified_sample(records, 31, "label", seed=4)] != [r.id for r in a]
        assert [r.id for r in a] == sorted([r.id for r in a], key=[r.id for r in records].index)  # input order kept

    def test_sample_larger_than_set_returns_everything(self):
        records = load_records_from_rows(study_like_rows()[:5])
        assert stratified_sample(records, 50, "label", seed=0) == records
        assert len(stratified_sample(records, 3, None, seed=0)) == 3


def load_records_from_rows(rows: list[dict[str, object]]):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return load_records(write_jsonl(Path(tmp) / "set.jsonl", rows))


class TestMetrics:
    def test_study_matrix(self):
        judge_pass = [True] * 34 + [False] * 24 + [True] * 2 + [False] * 2
        human_pass = [True] * 58 + [False] * 4
        m = binary_metrics(judge_pass, human_pass)
        assert m["accuracy"] == pytest.approx(36 / 62)
        assert m["kappa"] == pytest.approx(0.024, abs=5e-4)
        assert m["majority_baseline_accuracy"] == pytest.approx(58 / 62)
        assert m["recall_not_supported"] == pytest.approx(0.5)  # 2 of 4 unsupported caught
        assert m["precision_not_supported"] == pytest.approx(2 / 26)

    def test_choose_threshold_maximizes_f1_of_not_supported_with_tie_to_fallback(self):
        scores = [0.9, 0.8, 0.3, 0.2, 0.6, 0.4]
        human = [True, True, False, False, True, False]
        assert choose_threshold(scores, human, 0.5) == 0.5  # perfect separation at 0.5, which is the fallback
        assert choose_threshold([0.9, 0.2], [True, True], 0.7) == 0.7  # no negative: F1 is 0 everywhere, keep fallback

    def test_three_class_metrics(self):
        m = three_class_metrics(
            ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "SUPPORTED"],
            ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "SUPPORTED"],
        )
        assert m["per_class"]["SUPPORTED"]["f1"] == 1.0
        assert m["confusion"]["CONTRADICTED"]["UNSUPPORTED"] == 1
        assert m["per_class"]["UNSUPPORTED"]["support"] == 0.0

    def test_calibration_metrics(self):
        perfect = calibration_metrics([1.0, 0.0, 1.0], [True, False, True])
        assert perfect["brier"] == 0.0 and perfect["ece"] == pytest.approx(0.0)
        poor = calibration_metrics([0.9, 0.9], [True, False])
        assert poor["brier"] == pytest.approx((0.01 + 0.81) / 2)
        assert poor["ece"] == pytest.approx(0.4)
        assert calibration_metrics([], [])["brier"] is None

    def test_grouped_bootstrap_is_seeded_and_counts_exclusions(self):
        judge_pass = [True, False] * 10
        human_pass = [True, True, False, False] * 5
        groups = [f"g{i // 2}" for i in range(20)]
        a = grouped_bootstrap(judge_pass, human_pass, groups, seed=1, resamples=200)
        b = grouped_bootstrap(judge_pass, human_pass, groups, seed=1, resamples=200)
        assert a == b and a["groups"] == 10 and a["reliable"] is True
        assert all(ci is not None for ci in a["intervals"].values())
        rare = grouped_bootstrap(
            [True] * 19 + [False], [True] * 19 + [False], [str(i) for i in range(20)], seed=0, resamples=300
        )
        assert rare["excluded_single_class_draws"] > 30 and rare["reliable"] is False


class TestRun:
    @pytest.mark.asyncio
    async def test_sentence_set_calls_the_judge_with_the_sentence_only(self, tmp_path: Path):
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()))
        judge = ScriptedOrdinalJudge()
        run = await run_benchmark(
            judge, records, out_dir=tmp_path / "out", bootstrap_resamples=200, by=["task", "generator"]
        )

        assert len(judge.received) == 62
        assert judge.received[0] == {
            "question": "Question 0?",
            "context": "Context 0.",
            "answer": "Supported sentence 0.",
        }
        assert all("Previous sentence" not in str(call) for call in judge.received)
        report = run.report
        assert report["prev_sentence"] == {"records_with_prev_sentence": 46, "transmitted_to_judge": False}
        assert report["level"] == "sentence"
        assert report["threshold"] == {"value": 0.75, "source": "verdict_policy", "policy_threshold": 0.75}
        assert report["binary"]["kappa"] == pytest.approx(0.024, abs=5e-4)
        assert report["binary"]["accuracy"] == pytest.approx(36 / 62)
        assert report["binary"]["majority_baseline_accuracy"] == pytest.approx(58 / 62)
        assert report["failures"] == {} and report["truncations"] == 0
        assert set(report["by"]) == {"task", "generator"} and set(report["by"]["task"]) == {"QA", "Summary"}
        assert report["bootstrap"]["groups"] == 31
        assert "three_class" not in report and "calibration" not in report
        assert run.report_json.exists() and run.report_md.exists()
        assert "not transmitted to the judge" in run.report_md.read_text(encoding="utf-8")
        assert len(read_attempts(run.attempts_path)) == 62

    @pytest.mark.asyncio
    async def test_resume_skips_attempts_already_written(self, tmp_path: Path):
        rows = study_like_rows()[:10]
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        first = ScriptedOrdinalJudge()
        await run_benchmark(first, records[:4], out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert len(first.received) == 4

        second = ScriptedOrdinalJudge()
        run = await run_benchmark(second, records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert len(second.received) == 6  # only the 6 missing ids were judged
        assert run.report["records"]["scored"] == 10
        assert [r["sample_id"] for r in map(json.loads, run.attempts_path.read_text().splitlines())][:4] == [
            r.id for r in records[:4]
        ]

    @pytest.mark.asyncio
    async def test_failures_are_counted_and_excluded(self, tmp_path: Path):
        rows = study_like_rows()[:6]
        rows[2]["sentence"] = "Broken (fail-parse)."
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert run.report["records"] == {"total": 6, "evaluated": 6, "scored": 5, "failed": 1}
        assert run.report["failures"] == {"parse": 1}

    @pytest.mark.asyncio
    async def test_probabilistic_judge_threshold_comes_from_dev_only(self, tmp_path: Path):
        # dev: supported sentences at p=0.6, unsupported at p=0.4 -> best cut 0.5 ... but with p=0.7/0.65 the cut moves
        dev_rows = [sentence_row(i, "SUPPORTED", "ok p=0.7") for i in range(6)] + [
            sentence_row(10 + i, "UNSUPPORTED", "bad p=0.6") for i in range(6)
        ]
        test_rows = [sentence_row(100 + i, "SUPPORTED", "ok p=0.6") for i in range(4)] + [
            sentence_row(200 + i, "CONTRADICTED", "bad p=0.3") for i in range(4)
        ]
        dev = load_records(write_jsonl(tmp_path / "dev.jsonl", dev_rows))
        test = load_records(write_jsonl(tmp_path / "test.jsonl", test_rows))

        run = await run_benchmark(
            ScriptedProbabilityJudge(), test, out_dir=tmp_path / "out", dev_records=dev, bootstrap_resamples=0
        )

        assert run.report["threshold"]["value"] == 0.65  # separates dev; the policy's 0.5 would not
        assert run.report["threshold"]["source"].startswith("dev (12 scored records")
        # applied unchanged to the test set: 0.6 < 0.65 -> every supported test sentence is a false reject
        assert run.report["binary"]["recall_not_supported"] == 1.0
        assert run.report["binary"]["accuracy"] == pytest.approx(0.5)
        assert "calibration" in run.report and run.report["calibration"]["brier"] is not None
        assert (tmp_path / "out" / "dev_attempts.jsonl").exists()

        without_dev = await run_benchmark(
            ScriptedProbabilityJudge(), test, out_dir=tmp_path / "out2", bootstrap_resamples=0
        )
        assert without_dev.report["threshold"] == {"value": 0.5, "source": "verdict_policy", "policy_threshold": 0.5}

    @pytest.mark.asyncio
    async def test_three_class_judge_gets_a_confusion_matrix(self, tmp_path: Path):
        rows = [
            sentence_row(0, "SUPPORTED", "fine"),
            sentence_row(1, "CONTRADICTED", "wrong [C]"),
            sentence_row(2, "UNSUPPORTED", "made up [U]"),
            sentence_row(3, "UNSUPPORTED", "made up but called contradicted [C]"),
        ]
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(ScriptedThreeClassJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        three = run.report["three_class"]
        assert three["confusion"]["UNSUPPORTED"]["CONTRADICTED"] == 1
        assert three["per_class"]["SUPPORTED"]["f1"] == 1.0
        assert run.report["binary"]["accuracy"] == 1.0  # binary reading: every not-supported sentence was rejected
        assert "## Three classes" in run.report_md.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_sample_is_recorded_and_reproducible(self, tmp_path: Path):
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()))
        a = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "a", sample=20, seed=5, bootstrap_resamples=50
        )
        b = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "b", sample=20, seed=5, bootstrap_resamples=50
        )
        assert a.report["records"]["evaluated"] == 20
        assert a.report["sample"] == {"size": 20, "stratify": "label", "seed": 5}
        assert read_attempts(a.attempts_path).keys() == read_attempts(b.attempts_path).keys()
        for key in ("binary", "bootstrap", "by", "threshold"):
            assert a.report[key] == b.report[key]


class TestCli:
    def test_judge_bench_with_a_python_judge(self, tmp_path: Path):
        data = write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:12])
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--json",
                "judge-bench",
                "--judge",
                "python:tests.unit.benchmarks.fake_judge:ScriptedOrdinalJudge",
                "--data",
                str(data),
                "--out",
                str(out),
                "--bootstrap",
                "20",
                "--by",
                "task",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "success"
        assert payload["data"]["report"]["records"]["scored"] == 12
        assert (out / "report.md").exists()

    def test_sentence_level_wraps_the_judge_on_an_answer_level_set(self, tmp_path: Path):
        rows = [
            {
                "id": "a1",
                "group_id": "g1",
                "question": "q",
                "context": "c",
                "answer": "First is fine. Second is wrong (raw 1).",
                "label": "FAIL",
            },
            {
                "id": "a2",
                "group_id": "g2",
                "question": "q",
                "context": "c",
                "answer": "Only good sentences here. Still good.",
                "label": "PASS",
            },
            {"id": "a3", "group_id": "g3", "question": "q", "context": "c", "answer": "Sure!", "label": "PASS"},
        ]
        data = write_jsonl(tmp_path / "answers.jsonl", rows)
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--json",
                "judge-bench",
                "--judge",
                "python:tests.unit.benchmarks.fake_judge:ScriptedOrdinalJudge",
                "--data",
                str(data),
                "--out",
                str(out),
                "--bootstrap",
                "0",
                "--sentence-level",
            ],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)["data"]["report"]
        assert report["level"] == "answer"
        assert report["config"]["unit"] == "sentence"
        assert report["config"]["aggregation"]["score_rule"] == "fraction_pass"
        assert report["records"] == {"total": 3, "evaluated": 3, "scored": 2, "failed": 1}
        assert report["failures"] == {"no_units": 1}  # "Sure!" has no judgeable sentence
        assert report["binary"]["accuracy"] == 1.0  # a1 FAIL (worst), a2 PASS

    def test_sentence_level_is_refused_on_a_sentence_level_set(self, tmp_path: Path):
        data = write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:3])
        result = runner.invoke(
            app,
            [
                "judge-bench",
                "--judge",
                "python:tests.unit.benchmarks.fake_judge:ScriptedOrdinalJudge",
                "--data",
                str(data),
                "--out",
                str(tmp_path / "o"),
                "--sentence-level",
            ],
        )
        assert result.exit_code == 2 and "already segmented" in result.output

    def test_judge_bench_help_and_errors(self, tmp_path: Path):
        assert "--prev" not in runner.invoke(app, ["judge-bench", "--help"]).output
        assert runner.invoke(app, ["judge-bench", "--help"]).exit_code == 0
        missing = runner.invoke(
            app,
            ["judge-bench", "--judge", "python:x:Y", "--data", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path)],
        )
        assert missing.exit_code == 2 and "File not found" in missing.output
        data = write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:2])
        unknown = runner.invoke(
            app, ["judge-bench", "--judge", "oracle", "--data", str(data), "--out", str(tmp_path / "o")]
        )
        assert unknown.exit_code == 2 and "Unknown judge" in unknown.output
        bad_format = runner.invoke(
            app,
            [
                "judge-bench",
                "--judge",
                "llm",
                "--data",
                str(data),
                "--out",
                str(tmp_path / "o"),
                "--prompt-format",
                "x",
            ],
        )
        assert bad_format.exit_code == 2 and "Invalid --prompt-format" in bad_format.output
