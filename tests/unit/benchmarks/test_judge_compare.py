"""judge-compare: paired comparison of two judge-bench runs on the same records."""

from __future__ import annotations

import json
import math
import random
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from ragnarok_ai.benchmarks.compare import (
    ComparisonError,
    auroc,
    auroc_reweighted,
    compare_runs,
    figures,
    load_run,
    mcnemar_exact,
)
from ragnarok_ai.benchmarks.judges import file_sha256, load_records, prevalence_adjusted, run_benchmark
from ragnarok_ai.cli.main import app
from ragnarok_ai.core.exceptions import JudgeParseError
from ragnarok_ai.evaluators.judge import JudgeResult
from ragnarok_ai.judges import BinaryProbability, JudgeConfig, OrdinalBands, OrdinalScore, ScoreThreshold

if TYPE_CHECKING:
    from pathlib import Path

    from ragnarok_ai.benchmarks.judges import BenchRecord

CRITERIA = frozenset({"faithfulness"})


class MarkerJudge:
    """A probabilistic judge reading its P(supported) in the text: 'a=0.9' for the judge named 'a'.

    '(fail-a)' makes that judge fail to parse; another judge ignores it.
    """

    criteria = CRITERIA

    def __init__(self, name: str, threshold: float = 0.5) -> None:
        self.name = name
        self.threshold = threshold

    def config(self, criterion: str) -> JudgeConfig:
        return JudgeConfig(
            "tests.MarkerJudge", "classifier", f"marker-{self.name}", criterion, ScoreThreshold(self.threshold), "test"
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        if f"(fail-{self.name})" in answer:
            raise JudgeParseError("Unparseable", criterion=criterion, raw_output=answer)
        p = float(answer.split(f"{self.name}=", 1)[1].split()[0])
        return JudgeResult(
            verdict="PASS" if p >= self.threshold else "FAIL",
            score=p,
            explanation="",
            criteria=criterion,
            raw_score=None,
            output=BinaryProbability(p),
        )


class OrdinalMarkerJudge:
    """An ordinal judge (no probability): raw 5 when its marker is at or above 0.5, raw 1 otherwise."""

    criteria = CRITERIA

    def config(self, criterion: str) -> JudgeConfig:
        return JudgeConfig(
            "tests.OrdinalMarkerJudge", "custom", "ordinal-marker", criterion, OrdinalBands(4, 2), "test"
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        raw = 5 if float(answer.split("a=", 1)[1].split()[0]) >= 0.5 else 1
        return JudgeResult(
            verdict="PASS" if raw >= 4 else "FAIL",
            score=(raw - 1) / 4,
            explanation="",
            criteria=criterion,
            raw_score=raw,
            output=OrdinalScore(raw, 1, 5),
        )


def row(i: int, label: str, a: float, b: float, extra: str = "") -> dict[str, Any]:
    return {
        "id": f"r{i}",
        "source_id": f"src-{i}",
        "question": f"Question {i}?",
        "context": f"Context {i}.",
        "sentence": f"Sentence {i} a={a} b={b} {extra}".strip(),
        "label": label,
    }


def hand_table() -> list[dict[str, Any]]:
    """Twelve sentences, one group each. At threshold 0.5:

    A passes 5 of 6 supported, detects 1 of 3 contradicted and 3 of 3 unsupported;
    B passes 6 of 6 supported, detects 3 of 3 contradicted and 1 of 3 unsupported.
    """
    supported = [("SUPPORTED", 0.9, 0.8)] * 5 + [("SUPPORTED", 0.3, 0.7)]
    contradicted = [("CONTRADICTED", 0.2, 0.1), ("CONTRADICTED", 0.6, 0.2), ("CONTRADICTED", 0.7, 0.3)]
    unsupported = [("UNSUPPORTED", 0.1, 0.4), ("UNSUPPORTED", 0.2, 0.6), ("UNSUPPORTED", 0.4, 0.9)]
    return [row(i, label, a, b) for i, (label, a, b) in enumerate(supported + contradicted + unsupported)]


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


async def bench(judge: Any, data: Path, out: Path, **kwargs: Any) -> Path:
    await run_benchmark(
        judge, load_records(data), out_dir=out, data_sha256=file_sha256(data), bootstrap_resamples=0, **kwargs
    )
    return out


async def two_runs(
    tmp_path: Path, rows: list[dict[str, Any]], **kwargs: Any
) -> tuple[Path, Path, Path, list[BenchRecord]]:
    data = write(tmp_path / "data.jsonl", rows)
    a = await bench(MarkerJudge("a"), data, tmp_path / "run-a", **kwargs)
    b = await bench(MarkerJudge("b"), data, tmp_path / "run-b", **kwargs)
    return a, b, data, load_records(data)


class TestStatistics:
    @pytest.mark.parametrize(
        ("only_a", "only_b", "expected"),
        [(0, 0, 1.0), (2, 3, 1.0), (0, 5, 2 * 1 / 32), (1, 9, 2 * (1 + 10) / 1024), (9, 1, 2 * (1 + 10) / 1024)],
    )
    def test_exact_mcnemar(self, only_a: int, only_b: int, expected: float):
        assert mcnemar_exact(only_a, only_b) == pytest.approx(expected)

    def test_auroc_with_a_tie(self):
        # supported at 0.9, 0.8, 0.4; not supported at 0.4, 0.1: 2 + 2 + (0.5 + 1) wins out of 6 pairs
        assert auroc([0.9, 0.8, 0.4, 0.4, 0.1], [True, True, True, False, False]) == pytest.approx(5.5 / 6)
        assert auroc([0.1, 0.9], [True, False]) == 0.0
        assert auroc([0.5, 0.5], [True, False]) == 0.5
        assert auroc([0.9, 0.8], [True, True]) is None  # a single class

    def test_auroc_reweighted_weighs_each_not_supported_label_by_its_share_in_the_full_set(self):
        # supported always above the contradicted (AUROC 1), below the unsupported half of the time (0.5)
        scores = [0.8, 0.6, 0.1, 0.2, 0.7, 0.5]
        labels = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "CONTRADICTED", "UNSUPPORTED", "UNSUPPORTED"]
        full = {"SUPPORTED": 1000, "CONTRADICTED": 40, "UNSUPPORTED": 160}
        assert auroc([0.8, 0.6, 0.7, 0.5], [True, True, False, False]) == 0.75
        assert auroc_reweighted(scores, labels, full) == pytest.approx((40 * 1.0 + 160 * 0.75) / 200)
        # pooled, the sample's own mix (one contradicted for one unsupported) gives another figure
        assert auroc(scores, [label == "SUPPORTED" for label in labels]) == pytest.approx((1.0 + 0.75) / 2)
        assert auroc_reweighted(scores[:4], labels[:4], full) is None  # no unsupported record in the sample


class TestHandComputedTable:
    @pytest.mark.asyncio
    async def test_figures_differences_and_mcnemar(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, hand_table())
        run = compare_runs(a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), resamples=0)
        report = run.report
        assert report["data"] == {
            "sha256": file_sha256(data),
            "level": "sentence",
            "criterion": "faithfulness",
            "records": 12,
            "evaluated": 12,
            "compared": 12,
            "per_class": None,
            "sample": None,
            "labels_compared": {"CONTRADICTED": 3, "SUPPORTED": 6, "UNSUPPORTED": 3},
            "labels_full_set": {"CONTRADICTED": 3, "SUPPORTED": 6, "UNSUPPORTED": 3},
        }
        d = report["differences"]
        # A: 4 detected, 2 missed, 1 supported rejected. B: 4 detected, 2 missed, none rejected
        f1_a, f1_b = 2 * 0.8 * (4 / 6) / (0.8 + 4 / 6), 2 * 1.0 * (4 / 6) / (1.0 + 4 / 6)
        assert (d["f1_not_supported"]["a"], d["f1_not_supported"]["b"]) == (pytest.approx(f1_a), pytest.approx(f1_b))
        assert d["f1_not_supported"]["difference"] == pytest.approx(f1_b - f1_a) == pytest.approx(0.8 - 8 / 11)
        assert d["recall_CONTRADICTED"]["difference"] == pytest.approx(1 - 1 / 3)
        assert d["recall_UNSUPPORTED"]["difference"] == pytest.approx(1 / 3 - 1)
        assert d["specificity"]["difference"] == pytest.approx(1 - 5 / 6)
        assert d["recall_not_supported"]["difference"] == pytest.approx(0.0)  # 4 of 6 for both, not the same four
        assert d["accuracy"]["difference"] == pytest.approx(10 / 12 - 9 / 12)
        # the declared headline is the raw F1: no quotas here
        assert report["headline"]["figure"] == "f1_not_supported" and d["f1_not_supported"]["role"] == "headline"
        assert report["headline"]["difference"] == pytest.approx(f1_b - f1_a)
        # A alone correct on two unsupported, B alone on one supported and two contradicted
        assert report["mcnemar"] == {
            "both_correct": 7,
            "only_a_correct": 2,
            "only_b_correct": 3,
            "both_wrong": 0,
            "discordant": 5,
            "p_value": 1.0,
            "alpha": 0.05,
            "significant": False,
            "assumes_independent_records": True,
            "role": "paired_test",  # a full run: next to the headline figure
            "follows_sample_class_mix": False,
        }
        md = run.report_md.read_text()
        assert "## Paired test on the binary verdict (McNemar, exact)" in md
        assert md.index("## Paired test on the binary verdict") < md.index("## Exploratory figures")
        assert (
            "The exact McNemar test assumes independent records and takes no account of the groups (12 here): when "
            "records are grouped, for instance the sentences of one answer, its p-value is too optimistic. The paired "
            "bootstrap interval above, which resamples groups, is the one to rely on."
        ) in md
        assert "Under class quotas this test" not in md
        assert [(x["id"], x["label"], x["correct"]) for x in report["disagreements"]] == [
            ("r5", "SUPPORTED", "b"),
            ("r7", "CONTRADICTED", "b"),
            ("r8", "CONTRADICTED", "b"),
            ("r10", "UNSUPPORTED", "a"),
            ("r11", "UNSUPPORTED", "a"),
        ]
        assert set(report["disagreements"][0]) == {"id", "label", "a_score", "b_score", "a_pass", "b_pass", "correct"}
        # no text of the data file in what is written
        text = run.report_json.read_text() + run.report_md.read_text()
        assert "Sentence 5" not in text and "Context 5" not in text and "Question 5" not in text

    @pytest.mark.asyncio
    async def test_figures_are_those_of_each_run_s_own_report(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, hand_table())
        report = compare_runs(
            a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), resamples=0
        ).report
        for key, folder in (("a", a), ("b", b)):
            own = json.loads((folder / "report.json").read_text())
            for name in ("f1_not_supported", "accuracy", "kappa", "balanced_accuracy", "recall_not_supported"):
                assert report["differences"][name][key] == pytest.approx(own["binary"][name]), (key, name)
            by_label = own["recall_by_label"]
            for label in ("CONTRADICTED", "UNSUPPORTED"):
                assert report["differences"][f"recall_{label}"][key] == by_label["labels"][label]["recall"]
            assert report["differences"]["specificity"][key] == by_label["specificity"]["value"]
            assert report["differences"]["brier"][key] == pytest.approx(own["calibration"]["brier"])

    @pytest.mark.asyncio
    async def test_without_draws_there_is_no_interval_and_no_verdict(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, hand_table())
        report = compare_runs(
            a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), resamples=0
        ).report
        head = report["headline"]
        assert head["interval"] is None and head["contains_zero"] is None and head["ahead"] is None
        assert head["verdict"] == "No interval was computed (0 draws): the difference is given without a verdict."


def clear_rows(groups: int = 40) -> list[dict[str, Any]]:
    """Per group: 3 supported, 2 contradicted, 2 unsupported. A misses every contradicted sentence, B none."""
    rows: list[dict[str, Any]] = []
    for g in range(groups):
        cells = [("SUPPORTED", 0.9, 0.9)] * 3 + [("CONTRADICTED", 0.8, 0.1)] * 2 + [("UNSUPPORTED", 0.2, 0.2)] * 2
        for k, (label, a, b) in enumerate(cells):
            r = row(g * 10 + k, label, a, b)
            r["source_id"] = f"g{g}"
            rows.append(r)
    return rows


class TestKnownJudges:
    @pytest.mark.asyncio
    async def test_a_judge_that_detects_what_the_other_misses_is_named_ahead(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, clear_rows())
        run = compare_runs(
            a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), seed=3, resamples=200
        )
        report, d = run.report, run.report["differences"]
        # every group has the same composition: each draw gives the same figures, the intervals are points
        f1_a = 2 * 1.0 * 0.5 / 1.5  # A: precision 1, recall 80 of 160
        assert d["f1_not_supported"]["a"] == pytest.approx(f1_a) and d["f1_not_supported"]["b"] == 1.0
        assert d["f1_not_supported"]["interval"] == pytest.approx([1 - f1_a, 1 - f1_a])
        assert d["f1_not_supported"]["contains_zero"] is False
        assert d["recall_CONTRADICTED"]["interval"] == [1.0, 1.0]
        assert d["recall_UNSUPPORTED"]["interval"] == [0.0, 0.0] and d["recall_UNSUPPORTED"]["contains_zero"] is True
        assert d["specificity"]["difference"] == 0.0
        head = report["headline"]
        assert head["ahead"] == "b" and head["contains_zero"] is False
        assert head["verdict"] == (
            "B (tests.MarkerJudge) is ahead on f1_not_supported: the 95% interval of the difference excludes 0."
        )
        # B alone is right on the 80 contradicted sentences
        assert (report["mcnemar"]["only_a_correct"], report["mcnemar"]["only_b_correct"]) == (0, 80)
        assert report["mcnemar"]["p_value"] == pytest.approx(2 / 2**80) and report["mcnemar"]["significant"] is True
        # both judges give probabilities: the AUROC stands next to the headline
        free = report["threshold_free"]
        assert free["figure"] == "auroc" and free["role"] == "threshold_free" and d["auroc"]["role"] == "threshold_free"
        # A ranks every supported sentence above every other one, as B does (0.9 against 0.8 at most): the
        # ranking is perfect for both, and the difference lies entirely in where A's threshold sits
        assert (free["a"], free["b"], free["difference"]) == (1.0, 1.0, 0.0) and free["contains_zero"] is True
        md = run.report_md.read_text()
        assert "**Headline figure: `f1_not_supported`.** B (tests.MarkerJudge) is ahead" in md
        assert "## Independent of the thresholds" in md and "It stands next to the headline figure" in md
        assert "| recall_UNSUPPORTED | 1.000 | 1.000 | +0.000 | +0.000 to +0.000 | contains 0: not separated |" in md
        assert "| recall_CONTRADICTED | 0.000 | 1.000 | +1.000 | +1.000 to +1.000 | excludes 0: B better |" in md
        assert "## Exploratory figures" in md and "Not corrected for multiple comparisons" in md

    @pytest.mark.asyncio
    async def test_the_comparison_is_antisymmetric(self, tmp_path: Path):
        rng = random.Random(5)
        rows = []
        for i in range(90):
            label = rng.choice(["SUPPORTED", "SUPPORTED", "CONTRADICTED", "UNSUPPORTED"])
            r = row(i, label, round(rng.random(), 2), round(rng.random(), 2))
            r["source_id"] = f"g{i // 3}"
            rows.append(r)
        a, b, data, records = await two_runs(tmp_path, rows)
        sha = file_sha256(data)
        ab = compare_runs(a, b, records, out_dir=tmp_path / "ab", data_sha256=sha, seed=1, resamples=150).report
        ba = compare_runs(b, a, records, out_dir=tmp_path / "ba", data_sha256=sha, seed=1, resamples=150).report
        for name, entry in ab["differences"].items():
            other = ba["differences"][name]
            assert (entry["a"], entry["b"]) == (other["b"], other["a"])
            assert entry["difference"] == pytest.approx(-other["difference"])
            assert entry["interval"] == pytest.approx([-other["interval"][1], -other["interval"][0]])
            assert entry["contains_zero"] == other["contains_zero"]
        assert ab["mcnemar"]["only_a_correct"] == ba["mcnemar"]["only_b_correct"]
        assert ab["mcnemar"]["p_value"] == pytest.approx(ba["mcnemar"]["p_value"])
        # with noisy, unrelated scores the headline interval contains 0 and no judge is named
        head = ab["headline"]
        assert head["contains_zero"] is True and head["ahead"] is None
        assert head["verdict"] == (
            "No difference is established: the 95% interval of the difference of f1_not_supported contains 0. "
            "These data do not separate the two judges."
        )

    @pytest.mark.asyncio
    async def test_intervals_are_those_of_paired_draws_on_the_common_sequence(self, tmp_path: Path):
        rng = random.Random(11)
        rows = []
        for i in range(60):
            label = rng.choice(["SUPPORTED", "SUPPORTED", "CONTRADICTED", "UNSUPPORTED"])
            r = row(i, label, round(rng.random(), 2), round(rng.random(), 2))
            r["source_id"] = f"g{i // 4}"
            rows.append(r)
        a, b, data, records = await two_runs(tmp_path, rows)
        seed, resamples = 9, 120
        report = compare_runs(
            a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), seed=seed, resamples=resamples
        ).report

        # the draws regenerated as grouped_bootstrap generates them; both judges on the same indices
        rows_a, rows_b = load_run(a).rows, load_run(b).rows
        human = [r.human_pass for r in records]
        labels = [r.label for r in records]
        pass_a = [rows_a[r.id]["result"]["score"] >= 0.5 for r in records]
        pass_b = [rows_b[r.id]["result"]["score"] >= 0.5 for r in records]
        order = list(dict.fromkeys(r.group_id for r in records))
        members = {g: [i for i, r in enumerate(records) if r.group_id == g] for g in order}
        draw_rng = random.Random(seed)
        counts = dict(sorted({label: labels.count(label) for label in set(labels)}.items()))
        expected: dict[str, list[float]] = {"f1_not_supported": [], "recall_CONTRADICTED": [], "specificity": []}
        lost = dict.fromkeys(expected, 0)
        for _ in range(resamples):
            drawn = [i for _ in order for i in members[order[draw_rng.randrange(len(order))]]]
            fa, fb = (
                figures(
                    [p[i] for i in drawn],
                    [human[i] for i in drawn],
                    [labels[i] for i in drawn],
                    None,
                    level="sentence",
                    quotas=False,
                    full_label_counts=counts,
                    full_not_supported_prevalence=0.0,
                )
                for p in (pass_a, pass_b)
            )
            for name in expected:
                if fa[name] is None or fb[name] is None:
                    lost[name] += 1
                else:
                    expected[name].append(fb[name] - fa[name])

        def interval(values: list[float]) -> list[float]:
            values = sorted(values)
            return [
                values[max(0, math.floor(0.025 * (len(values) - 1)))],
                values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))],
            ]

        for name, values in expected.items():
            assert report["differences"][name]["interval"] == pytest.approx(interval(values)), name
            assert report["differences"][name]["excluded_draws"] == lost[name]
        assert report["bootstrap"] == {"resamples": 120, "seed": 9, "confidence": 0.95, "groups": 15, "paired": True}


class TestSelfComparison:
    @pytest.mark.asyncio
    async def test_a_judge_against_itself(self, tmp_path: Path):
        rows = clear_rows(groups=12)
        data = write(tmp_path / "data.jsonl", rows)
        a = await bench(MarkerJudge("a"), data, tmp_path / "run-a")
        run = compare_runs(
            a, a, load_records(data), out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), resamples=100
        )
        report = run.report
        for name, entry in report["differences"].items():
            assert entry["difference"] == 0.0, name
            assert entry["interval"] == [0.0, 0.0], name  # reduced to a point
            assert entry["contains_zero"] is True
        assert report["mcnemar"]["discordant"] == 0 and report["mcnemar"]["p_value"] == 1.0
        assert report["disagreements"] == []
        assert report["same_configuration"] is True
        assert any("repeatability of one judge" in note for note in report["notes"])
        head = report["headline"]
        assert head["ahead"] is None and head["verdict"].startswith("No difference is established")
        assert "0 record(s) on which the two verdicts differ" in run.report_md.read_text()

    @pytest.mark.asyncio
    async def test_two_runs_of_the_same_configuration_are_flagged(self, tmp_path: Path):
        data = write(tmp_path / "data.jsonl", hand_table())
        a = await bench(MarkerJudge("a"), data, tmp_path / "run-1")
        b = await bench(MarkerJudge("a"), data, tmp_path / "run-2")
        report = compare_runs(a, b, load_records(data), out_dir=tmp_path / "cmp", data_sha256=file_sha256(data)).report
        assert report["same_configuration"] is True and report["a"]["config_hash"] == report["b"]["config_hash"]


class TestQuotas:
    @staticmethod
    def rows() -> list[dict[str, Any]]:
        rng = random.Random(21)
        rows = []
        for i in range(400):
            label = "SUPPORTED" if i % 10 < 8 else ("CONTRADICTED" if i % 10 == 8 else "UNSUPPORTED")
            if i % 50 == 49:
                label = "CONTRADICTED"
            base = {"SUPPORTED": 0.75, "CONTRADICTED": 0.55, "UNSUPPORTED": 0.3}[label]
            a = min(1.0, max(0.0, round(base + rng.uniform(-0.35, 0.35), 2)))
            b = min(1.0, max(0.0, round(base + rng.uniform(-0.45, 0.25), 2)))
            r = row(i, label, a, b)
            r["source_id"] = f"g{i // 2}"
            rows.append(r)
        return rows

    @pytest.mark.asyncio
    async def test_headline_is_the_reweighted_f1_and_every_figure_matches_the_runs_reports(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, self.rows(), per_class=30, seed=4)
        run = compare_runs(
            a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), seed=2, resamples=150
        )
        report, d = run.report, run.report["differences"]
        assert report["data"]["per_class"] == 30 and report["data"]["compared"] == 90
        assert report["data"]["labels_compared"] == {"CONTRADICTED": 30, "SUPPORTED": 30, "UNSUPPORTED": 30}
        assert report["data"]["labels_full_set"]["SUPPORTED"] > 300  # the weights are the full set's, not the sample's
        assert report["headline"]["figure"] == "f1_not_supported_reweighted"
        assert d["f1_not_supported_reweighted"]["role"] == "headline" and d["f1_not_supported"]["role"] == "exploratory"
        assert report["threshold_free"]["figure"] == "auroc_reweighted" and d["auroc"]["role"] == "exploratory"
        for key, folder in (("a", a), ("b", b)):
            adjusted = json.loads((folder / "report.json").read_text())["prevalence_adjusted"]
            pairs = {
                "f1_not_supported_reweighted": "f1_not_supported_reweighted",
                "recall_not_supported_reweighted": "recall_not_supported",
                "precision_not_supported_reweighted": "precision_not_supported_reweighted",
                "accuracy_reweighted": "accuracy_reweighted",
                "brier_reweighted": "brier_reweighted",
            }
            for name, source in pairs.items():
                assert d[name][key] == pytest.approx(adjusted[source]), (key, name)
        for name in ("f1_not_supported_reweighted", "auroc_reweighted", "brier_reweighted"):
            low, high = d[name]["interval"]
            assert low <= d[name]["difference"] <= high
        assert d["brier_reweighted"]["higher_is_better"] is False
        md = run.report_md.read_text()
        assert "class quotas of 30 per label" in md and "| brier_reweighted (lower is better) |" in md
        # under class quotas McNemar is an exploratory figure: it tests the accuracy on the sample's class mix
        mcnemar = report["mcnemar"]
        assert (mcnemar["role"], mcnemar["follows_sample_class_mix"], mcnemar["assumes_independent_records"]) == (
            "exploratory",
            True,
            True,
        )
        assert "\n## Paired test on the binary verdict" not in md  # not a section of its own
        assert md.index("## Exploratory figures") < md.index("### Paired test on the binary verdict (McNemar, exact)")
        assert (
            "Under class quotas this test is about the accuracy on the sample's class mix, not the set's: it can "
            "point the other way than the reweighted figures."
        ) in md
        assert "its p-value is too optimistic" in md and "is the one to rely on" in md

    @pytest.mark.asyncio
    async def test_the_reweighting_is_redone_in_each_draw(self, tmp_path: Path):
        # per group: 2 supported (passed by both), 1 contradicted, 2 unsupported. A detects the unsupported
        # only, B the contradicted only: the per-label recalls are constant in every draw, so the reweighted
        # recalls are the full set's in every draw and the interval of their difference is a point
        rows = []
        for g in range(30):
            cells = [("SUPPORTED", 0.9, 0.9)] * 2 + [("CONTRADICTED", 0.8, 0.1)] + [("UNSUPPORTED", 0.1, 0.8)] * 2
            for k, (label, a, b) in enumerate(cells):
                r = row(g * 10 + k, label, a, b)
                r["source_id"] = f"g{g}-{k}"  # one group per record: a draw changes the mix of labels
                rows.append(r)
        a, b, data, records = await two_runs(tmp_path, rows, per_class=20, seed=1)
        d = compare_runs(
            a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), seed=5, resamples=200
        ).report["differences"]
        # full set: 30 contradicted, 60 unsupported. A: (30 * 0 + 60 * 1) / 90, B: (30 * 1 + 60 * 0) / 90
        assert (d["recall_not_supported_reweighted"]["a"], d["recall_not_supported_reweighted"]["b"]) == (
            pytest.approx(2 / 3),
            pytest.approx(1 / 3),
        )
        low, high = d["recall_not_supported_reweighted"]["interval"]
        assert low == pytest.approx(-1 / 3) and high == pytest.approx(-1 / 3)
        # the raw recall of the same draws follows their mix of the two labels: its interval has a width
        raw_low, raw_high = d["recall_not_supported"]["interval"]
        assert raw_high - raw_low > 0.1
        assert d["recall_not_supported"]["difference"] == pytest.approx(0.0)  # 20 of 40 for both in the sample


class TestRefusals:
    @pytest.mark.asyncio
    async def test_runs_on_other_records_are_refused_and_the_field_is_named(self, tmp_path: Path):
        rows = TestQuotas.rows()
        data = write(tmp_path / "data.jsonl", rows)
        records, sha = load_records(data), file_sha256(data)
        quota_30 = await bench(MarkerJudge("a"), data, tmp_path / "q30", per_class=30, seed=4)
        quota_20 = await bench(MarkerJudge("b"), data, tmp_path / "q20", per_class=20, seed=4)
        other_seed = await bench(MarkerJudge("b"), data, tmp_path / "s9", per_class=30, seed=9)
        full = await bench(MarkerJudge("b"), data, tmp_path / "full")
        with pytest.raises(
            ComparisonError, match=r"did not evaluate the same records.*differ on per_class \(30 for A, 20 for B\)"
        ):
            compare_runs(quota_30, quota_20, records, out_dir=tmp_path / "c1", data_sha256=sha)
        with pytest.raises(
            ComparisonError, match=r"did not evaluate the same records.*differ on seed \(4 for A, 9 for B\)"
        ):
            compare_runs(quota_30, other_seed, records, out_dir=tmp_path / "c2", data_sha256=sha)
        # a full run against a quota run: not the same records either
        with pytest.raises(ComparisonError, match=r"310 only in B.*differ on per_class \(30 for A, None for B\)"):
            compare_runs(quota_30, full, records, out_dir=tmp_path / "c3", data_sha256=sha)
        assert not (tmp_path / "c1").exists()  # nothing is written when the comparison is refused

    @pytest.mark.asyncio
    async def test_different_seeds_on_full_runs_are_accepted_and_stated(self, tmp_path: Path):
        data = write(tmp_path / "data.jsonl", hand_table())
        a = await bench(MarkerJudge("a"), data, tmp_path / "a", seed=1)
        b = await bench(MarkerJudge("b"), data, tmp_path / "b", seed=2)
        report = compare_runs(a, b, load_records(data), out_dir=tmp_path / "c", data_sha256=file_sha256(data)).report
        assert any("different seeds (1 for A, 2 for B)" in note for note in report["notes"])

    @pytest.mark.asyncio
    async def test_another_data_file_level_or_criterion(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, hand_table())
        other = write(tmp_path / "other.jsonl", [*hand_table()[:-1], row(99, "SUPPORTED", 0.9, 0.9)])
        with pytest.raises(ComparisonError, match="The data file is not the one the runs were made on: 'data_sha256'"):
            compare_runs(a, b, load_records(other), out_dir=tmp_path / "c", data_sha256=file_sha256(other))

        answers = write(
            tmp_path / "answers.jsonl",
            [{"id": f"r{i}", "answer": f"Answer a=0.9 b=0.9 {i}", "context": "c", "label": "PASS"} for i in range(12)],
        )
        answer_run = await bench(MarkerJudge("b"), answers, tmp_path / "answer-run")
        with pytest.raises(
            ComparisonError, match=r"The two runs differ on 'level': 'sentence' for A .* 'answer' for B"
        ):
            compare_runs(a, answer_run, records, out_dir=tmp_path / "c", data_sha256=file_sha256(data))

        # another criterion, for instance hallucination against hallucination_free
        attempts = b / "attempts.jsonl"
        lines = attempts.read_text().splitlines()
        header = json.loads(lines[0])
        header["criterion"] = "hallucination_free"
        attempts.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
        with pytest.raises(
            ComparisonError, match=r"differ on 'criterion': 'faithfulness' for A .* 'hallucination_free'"
        ):
            compare_runs(a, b, records, out_dir=tmp_path / "c", data_sha256=file_sha256(data))

    @pytest.mark.asyncio
    async def test_incomplete_or_unidentified_runs_and_an_output_in_a_run_folder(self, tmp_path: Path):
        a, b, data, records = await two_runs(tmp_path, hand_table())
        sha = file_sha256(data)
        with pytest.raises(ComparisonError, match=r"--out is the folder of a run .*report\.json would be overwritten"):
            compare_runs(a, b, records, out_dir=b, data_sha256=sha)
        assert json.loads((b / "report.json").read_text())["judge"] == "tests.MarkerJudge"  # untouched

        attempts = b / "attempts.jsonl"
        lines = attempts.read_text().splitlines()
        attempts.write_text("\n".join(lines[:-1]) + "\n")
        with pytest.raises(ComparisonError, match=r"Run B .* is incomplete: its report counts 12 .* holds 11"):
            compare_runs(a, b, records, out_dir=tmp_path / "c", data_sha256=sha)
        attempts.write_text("\n".join(lines[1:]) + "\n")
        with pytest.raises(ComparisonError, match="no header line"):
            compare_runs(a, b, records, out_dir=tmp_path / "c", data_sha256=sha)
        attempts.write_text("\n".join(lines) + "\n")
        (b / "report.json").unlink()
        with pytest.raises(
            ComparisonError, match=r"no report\.json\. The pass threshold of a run is recorded in its report"
        ):
            compare_runs(a, b, records, out_dir=tmp_path / "c", data_sha256=sha)
        with pytest.raises(ComparisonError, match=r"no attempts\.jsonl"):
            compare_runs(a, tmp_path / "nowhere", records, out_dir=tmp_path / "c", data_sha256=sha)


class TestPairing:
    @pytest.mark.asyncio
    async def test_failures_of_either_judge_are_counted_and_left_out(self, tmp_path: Path):
        rows = hand_table()
        rows[0]["sentence"] += " (fail-a)"
        rows[6]["sentence"] += " (fail-b)"
        rows[9]["sentence"] += " (fail-a) (fail-b)"
        a, b, data, records = await two_runs(tmp_path, rows)
        run = compare_runs(a, b, records, out_dir=tmp_path / "cmp", data_sha256=file_sha256(data), resamples=0)
        report = run.report
        assert report["failures"] == {
            "a_only": 1,
            "b_only": 1,
            "both": 1,
            "by_kind": {"a": {"parse": 2}, "b": {"parse": 2}},
        }
        assert (report["data"]["evaluated"], report["data"]["compared"]) == (12, 9)
        assert report["data"]["labels_compared"] == {"CONTRADICTED": 2, "SUPPORTED": 5, "UNSUPPORTED": 2}
        assert (
            sum(report["mcnemar"][k] for k in ("both_correct", "only_a_correct", "only_b_correct", "both_wrong")) == 9
        )
        md = run.report_md.read_text()
        assert "Records: 9 compared of 12 evaluated" in md
        assert "Failures: 3 record(s) left out of the comparison (1 failed in A only, 1 in B only, 1 in both" in md

    @pytest.mark.asyncio
    async def test_thresholds_differ_and_one_is_tuned(self, tmp_path: Path):
        data = write(tmp_path / "data.jsonl", hand_table())
        dev = write(
            tmp_path / "dev.jsonl",
            [row(100 + i, "SUPPORTED", 0.8, 0.8) for i in range(6)]
            + [row(200 + i, "UNSUPPORTED", 0.65, 0.65) for i in range(6)],
        )
        a = tmp_path / "a"
        await run_benchmark(
            MarkerJudge("a"),
            load_records(data),
            out_dir=a,
            dev_records=load_records(dev),
            data_sha256=file_sha256(data),
            dev_sha256=file_sha256(dev),
            bootstrap_resamples=0,
        )
        b = await bench(MarkerJudge("b"), data, tmp_path / "b")
        report = compare_runs(a, b, load_records(data), out_dir=tmp_path / "c", data_sha256=file_sha256(data)).report
        assert report["a"]["threshold"] == 0.7 and report["a"]["threshold_source"].startswith("dev")
        assert report["b"]["threshold"] == 0.5 and report["b"]["threshold_source"] == "verdict_policy"
        notes = " ".join(report["notes"])
        assert "Each judge is compared at its own pass threshold: 0.70 for A" in notes
        assert "The threshold of A was chosen on a development set and the one of B is its verdict policy's" in notes
        # A's verdicts are taken at 0.70: the supported sentence scored 0.9 passes, the unsupported at 0.4 does not
        assert report["differences"]["specificity"]["a"] == pytest.approx(5 / 6)

    @pytest.mark.asyncio
    async def test_no_threshold_free_comparison_unless_both_judges_give_probabilities(self, tmp_path: Path):
        data = write(tmp_path / "data.jsonl", hand_table())
        a = await bench(OrdinalMarkerJudge(), data, tmp_path / "a")
        b = await bench(MarkerJudge("b"), data, tmp_path / "b")
        run = compare_runs(a, b, load_records(data), out_dir=tmp_path / "c", data_sha256=file_sha256(data))
        assert run.report["threshold_free"] is None
        assert "auroc" not in run.report["differences"] and "brier" not in run.report["differences"]
        assert "Independent of the thresholds" not in run.report_md.read_text()
        assert run.report["a"]["threshold"] == 0.75  # the ordinal judge's own policy

    @pytest.mark.asyncio
    async def test_claim_level_judges_are_paired_on_answers(self, tmp_path: Path):
        from ragnarok_ai.evaluators import FaithfulnessEvaluator, HallucinationDetector
        from tests.unit.judges.claim_judges import ScriptedLLM

        context = "Paris is the capital of France."
        good, mixed = (
            "Paris is the capital of France. It is in Europe.",
            "Paris is big. It has 90 districts (unsupported).",
        )
        rows = [
            {
                "id": f"x{i}",
                "group_id": f"g{i}",
                "context": context,
                "answer": good if i % 2 else mixed,
                "label": "PASS" if i % 2 else "FAIL",
            }
            for i in range(10)
        ] + [{"id": "none", "group_id": "gn", "context": context, "answer": "Hello (no claims).", "label": "FAIL"}]
        data = write(tmp_path / "answers.jsonl", rows)
        records, sha = load_records(data), file_sha256(data)
        a = tmp_path / "faithfulness"
        await run_benchmark(
            FaithfulnessEvaluator(ScriptedLLM(model="m1")).as_judge(),
            records,
            out_dir=a,
            data_sha256=sha,
            bootstrap_resamples=0,
        )
        b = tmp_path / "faithfulness-2"
        await run_benchmark(
            FaithfulnessEvaluator(ScriptedLLM(model="m2")).as_judge(),
            records,
            out_dir=b,
            data_sha256=sha,
            bootstrap_resamples=0,
        )
        run = compare_runs(a, b, records, out_dir=tmp_path / "c", data_sha256=sha, resamples=50)
        report = run.report
        # the pair is the answer: 10 answers scored by both, the one without a claim failed in both
        assert report["data"]["level"] == "answer" and report["data"]["compared"] == 10
        assert report["failures"]["both"] == 1 and report["failures"]["by_kind"] == {
            "a": {"no_units": 1},
            "b": {"no_units": 1},
        }
        assert report["mcnemar"]["discordant"] == 0 and report["headline"]["difference"] == 0.0
        assert not any(name.startswith("recall_CONTRADICTED") for name in report["differences"])
        assert '"units"' not in json.dumps(report)  # no unit of either run is read or written
        # the detector's criterion is hallucination_free: not comparable with faithfulness
        c = tmp_path / "detector"
        await run_benchmark(
            HallucinationDetector(ScriptedLLM(model="m1")).as_judge(),
            records,
            out_dir=c,
            criterion="hallucination_free",
            data_sha256=sha,
            bootstrap_resamples=0,
        )
        with pytest.raises(
            ComparisonError, match=r"differ on 'criterion': 'faithfulness' for A .* 'hallucination_free' for B"
        ):
            compare_runs(a, c, records, out_dir=tmp_path / "c2", data_sha256=sha)


class TestFiguresAgainstTheHarness:
    def test_reweighted_figures_come_from_prevalence_adjusted(self):
        judge = [True] * 18 + [False] * 2 + [False] * 5 + [True] * 15 + [False] * 15 + [True] * 5
        labels = ["SUPPORTED"] * 20 + ["CONTRADICTED"] * 20 + ["UNSUPPORTED"] * 20
        human = [label == "SUPPORTED" for label in labels]
        full = {"SUPPORTED": 1000, "CONTRADICTED": 40, "UNSUPPORTED": 160}
        out = figures(
            judge,
            human,
            labels,
            None,
            level="sentence",
            quotas=True,
            full_label_counts=full,
            full_not_supported_prevalence=200 / 1200,
        )
        adjusted = prevalence_adjusted(judge, human, 200 / 1200, labels=labels, full_label_counts=full)
        assert out["f1_not_supported_reweighted"] == adjusted["f1_not_supported_reweighted"]
        assert out["recall_not_supported_reweighted"] == adjusted["recall_not_supported"] == pytest.approx(0.65)
        assert (out["recall_CONTRADICTED"], out["recall_UNSUPPORTED"], out["specificity"]) == (0.25, 0.75, 0.9)
        assert "auroc" not in out
        # a single human class: the figures the report's bootstrap excludes are undefined
        single = figures(
            [True, False],
            [True, True],
            ["SUPPORTED"] * 2,
            None,
            level="sentence",
            quotas=False,
            full_label_counts=full,
            full_not_supported_prevalence=0.1,
        )
        assert single["f1_not_supported"] is None and single["kappa"] is None and single["accuracy"] == 0.5


class TestCli:
    @pytest.mark.asyncio
    async def test_judge_compare_writes_the_report_and_refuses_with_the_field_named(self, tmp_path: Path):
        a, b, data, _ = await two_runs(tmp_path, clear_rows(groups=10))
        out = tmp_path / "cmp"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--json",
                "judge-compare",
                "--a",
                str(a),
                "--b",
                str(b),
                "--data",
                str(data),
                "--out",
                str(out),
                "--resamples",
                "40",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "success"
        report = payload["data"]["report"]
        assert report["bootstrap"]["resamples"] == 40 and report["headline"]["ahead"] == "b"
        assert (out / "report.json").exists() and (out / "report.md").exists()

        text = runner.invoke(
            app, ["judge-compare", "--a", str(a), "--b", str(b), "--data", str(data), "--out", str(out)]
        )
        assert text.exit_code == 0 and "Headline figure: f1_not_supported" in text.output
        assert "B (tests.MarkerJudge) is ahead" in text.output

        default = json.loads(
            runner.invoke(
                app, ["--json", "judge-compare", "--a", str(a), "--b", str(b), "--data", str(data), "--out", str(out)]
            ).output
        )
        assert default["data"]["report"]["bootstrap"]["resamples"] == 1000  # the default

        refused = runner.invoke(
            app, ["--json", "judge-compare", "--a", str(a), "--b", str(b), "--data", str(data), "--out", str(a)]
        )
        assert refused.exit_code != 0
        assert "--out is the folder of a run" in json.loads(refused.output)["errors"][0]
        missing = runner.invoke(
            app,
            ["judge-compare", "--a", str(a), "--b", str(tmp_path / "nowhere"), "--data", str(data), "--out", str(out)],
        )
        assert missing.exit_code != 0 and "Not a folder" in missing.output
