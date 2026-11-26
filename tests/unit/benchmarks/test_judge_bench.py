"""Judge benchmark harness: records, sampling, resume, thresholds, metrics, reports, CLI."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragnarok_ai.benchmarks.judges import (
    ATTEMPTS_HEADER_FIELDS,
    AttemptsMismatchError,
    attempts_header,
    binary_metrics,
    calibration_metrics,
    choose_threshold,
    file_sha256,
    grouped_bootstrap,
    load_records,
    ollama_model_memory,
    parse_record,
    prevalence_adjusted,
    quota_sample,
    read_attempts,
    recall_by_label,
    records_sha256,
    reweighted_brier,
    run_benchmark,
    stratified_sample,
    three_class_metrics,
)
from ragnarok_ai.cli.main import app
from ragnarok_ai.core.exceptions import JudgeParseError
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

    def test_quota_sample_takes_at_most_n_per_label_deterministically(self):
        records = load_records_from_rows(study_like_rows())  # 58 SUPPORTED, 2 UNSUPPORTED, 2 CONTRADICTED
        picked = quota_sample(records, 10, seed=3)
        labels = [r.label for r in picked]
        assert labels.count("SUPPORTED") == 10  # capped
        assert labels.count("UNSUPPORTED") == 2 and labels.count("CONTRADICTED") == 2  # small classes taken whole
        assert [r.id for r in picked] == [r.id for r in quota_sample(records, 10, seed=3)]
        assert [r.id for r in picked] != [r.id for r in quota_sample(records, 10, seed=4)]
        assert [r.id for r in picked] == [r.id for r in records if r.id in {p.id for p in picked}]  # input order
        with pytest.raises(ValueError, match="per_class must be at least 1"):
            quota_sample(records, 0, seed=0)

    def test_prevalence_adjusted_figures(self):
        # quota sample: 10 supported (8 passed), 10 not supported (7 caught); full set 94% supported
        judge = [True] * 8 + [False] * 2 + [False] * 7 + [True] * 3
        human = [True] * 10 + [False] * 10
        adjusted = prevalence_adjusted(judge, human, 0.06)
        assert adjusted["recall_not_supported"] == pytest.approx(0.7)
        assert adjusted["specificity"] == adjusted["recall_supported"] == pytest.approx(0.8)
        pi = 0.06
        assert adjusted["precision_not_supported_reweighted"] == pytest.approx(pi * 0.7 / (pi * 0.7 + 0.94 * 0.2))
        assert adjusted["accuracy_reweighted"] == pytest.approx(pi * 0.7 + 0.94 * 0.8)
        raw_precision = 7 / 9  # on the balanced sample: far higher than at the real prevalence
        assert adjusted["precision_not_supported_reweighted"] < raw_precision
        single = prevalence_adjusted([True] * 5, [True] * 5, 0.06)
        assert single["recall_not_supported"] is None and single["precision_not_supported_reweighted"] is None
        # without labels the two classes are the strata
        assert (adjusted["strata"], adjusted["labels"], adjusted["missing_labels"]) == ("class", None, [])

    @staticmethod
    def _sample(counts: dict[str, tuple[int, int]]) -> tuple[list[bool], list[bool], list[str]]:
        """Verdicts, human classes and labels of a sample given, per label, (records read as the humans do, n)."""
        judge: list[bool] = []
        labels: list[str] = []
        for label, (agreeing, n) in counts.items():
            supported = label == "SUPPORTED"
            judge += [supported] * agreeing + [not supported] * (n - agreeing)
            labels += [label] * n
        return judge, [label == "SUPPORTED" for label in labels], labels

    def test_prevalence_adjusted_reweights_per_label_when_both_negative_labels_are_subsampled(self):
        # full set: 1000 supported, 40 contradicted, 160 unsupported; the quota takes 20 of each,
        # that is half of the contradicted and an eighth of the unsupported
        full = {"SUPPORTED": 1000, "CONTRADICTED": 40, "UNSUPPORTED": 160}
        judge, human, labels = self._sample({"SUPPORTED": (18, 20), "CONTRADICTED": (5, 20), "UNSUPPORTED": (15, 20)})
        pi = 200 / 1200
        adjusted = prevalence_adjusted(judge, human, pi, labels=labels, full_label_counts=full)
        assert adjusted["strata"] == "label" and adjusted["missing_labels"] == []
        # per-label recalls 0.25 and 0.75, weighted 40 to 160 as in the full set
        assert pytest.approx(0.65) == (40 * 0.25 + 160 * 0.75) / 200
        assert adjusted["recall_not_supported"] == pytest.approx(0.65)
        assert adjusted["specificity"] == adjusted["recall_supported"] == pytest.approx(0.9)
        assert adjusted["precision_not_supported_reweighted"] == pytest.approx(pi * 0.65 / (pi * 0.65 + (1 - pi) * 0.1))
        assert adjusted["accuracy_reweighted"] == pytest.approx(pi * 0.65 + (1 - pi) * 0.9)
        assert adjusted["labels"] == {
            "CONTRADICTED": {
                "human_class": "not supported",
                "full_set_count": 40,
                "full_set_share": pytest.approx(40 / 1200),
                "n": 20,
                "rate": pytest.approx(0.25),
            },
            "SUPPORTED": {
                "human_class": "supported",
                "full_set_count": 1000,
                "full_set_share": pytest.approx(1000 / 1200),
                "n": 20,
                "rate": pytest.approx(0.9),
            },
            "UNSUPPORTED": {
                "human_class": "not supported",
                "full_set_count": 160,
                "full_set_share": pytest.approx(160 / 1200),
                "n": 20,
                "rate": pytest.approx(0.75),
            },
        }
        # pooling the two labels in the sample, as up to 1.13.1, follows the quota's 20-to-20 mix
        pooled = prevalence_adjusted(judge, human, pi)
        assert pooled["recall_not_supported"] == pytest.approx(0.5)

    def test_prevalence_adjusted_on_the_counts_of_a_real_quota_run(self):
        # HHEM-2.1-Open on the RAGTruth test set at threshold 0.25, counts read in the attempts of the
        # two runs: --per-class 300 (seed 7) and the full set. Same scores, the 139 contradicted are common.
        full = {"SUPPORTED": 9777, "CONTRADICTED": 139, "UNSUPPORTED": 469}
        full_detected = {"CONTRADICTED": 22, "UNSUPPORTED": 275}
        judge, human, labels = self._sample(
            {"SUPPORTED": (288, 300), "CONTRADICTED": (22, 139), "UNSUPPORTED": (166, 300)}
        )
        pi = 608 / 10385
        adjusted = prevalence_adjusted(judge, human, pi, labels=labels, full_label_counts=full)
        assert adjusted["recall_not_supported"] == pytest.approx((22 + 469 * 166 / 300) / 608)
        assert adjusted["recall_not_supported"] == pytest.approx(0.4630, abs=5e-5)
        assert adjusted["specificity"] == pytest.approx(0.96)
        assert adjusted["precision_not_supported_reweighted"] == pytest.approx(0.4185, abs=5e-5)
        assert adjusted["accuracy_reweighted"] == pytest.approx(0.9309, abs=5e-5)
        # what 1.13.0 and 1.13.1 reported for this run, and the value on the full set
        pooled = prevalence_adjusted(judge, human, pi)["recall_not_supported"]
        on_full_set = sum(full_detected.values()) / 608
        assert pooled == pytest.approx(188 / 439)
        assert pooled == pytest.approx(0.428246, abs=5e-7)
        assert on_full_set == pytest.approx(0.4885, abs=5e-5)
        # the mixing bias is removed; what remains is the draw of 300 unsupported out of 469 (166/300 against 275/469)
        assert pooled < adjusted["recall_not_supported"] < on_full_set
        with_full_rates = (22 + 469 * (275 / 469)) / 608
        assert with_full_rates == pytest.approx(on_full_set)

    def test_prevalence_adjusted_with_a_label_absent_from_the_sample(self):
        full = {"SUPPORTED": 100, "CONTRADICTED": 10, "UNSUPPORTED": 40}
        judge, human, labels = self._sample({"SUPPORTED": (9, 10), "UNSUPPORTED": (6, 10)})
        adjusted = prevalence_adjusted(judge, human, 50 / 150, labels=labels, full_label_counts=full)
        assert adjusted["missing_labels"] == ["CONTRADICTED"]
        assert adjusted["labels"]["CONTRADICTED"]["n"] == 0 and adjusted["labels"]["CONTRADICTED"]["rate"] is None
        # not renormalized over the unsupported alone
        assert adjusted["recall_not_supported"] is None
        assert adjusted["precision_not_supported_reweighted"] is None and adjusted["accuracy_reweighted"] is None
        assert adjusted["specificity"] == pytest.approx(0.9)  # its own label is there
        # a label the full set does not hold is not a stratum
        only_one = prevalence_adjusted(
            judge, human, 40 / 140, labels=labels, full_label_counts={**full, "CONTRADICTED": 0}
        )
        assert only_one["missing_labels"] == [] and only_one["recall_not_supported"] == pytest.approx(0.6)
        assert "CONTRADICTED" not in only_one["labels"]

    def test_reweighted_brier_on_a_hand_computed_table(self):
        # supported: errors 0.01, 0.09 (mean 0.05); not supported: errors 0.04, 0.36, 0.64 (mean 1.04 / 3)
        probabilities = [0.9, 0.7, 0.2, 0.6, 0.8]
        human = [True, True, False, False, False]
        raw = calibration_metrics(probabilities, human)["brier"]
        assert raw == pytest.approx((0.01 + 0.09 + 0.04 + 0.36 + 0.64) / 5)
        assert reweighted_brier(probabilities, human, 0.06) == pytest.approx(0.06 * 1.04 / 3 + 0.94 * 0.05)
        # at the sample's own prevalence the reweighting gives the raw Brier back
        assert reweighted_brier(probabilities, human, 3 / 5) == pytest.approx(raw)
        assert reweighted_brier([0.9, 0.8], [True, True], 0.06) is None  # a class is absent
        assert reweighted_brier([], [], 0.06) is None

    def test_reweighted_brier_per_label_when_both_negative_labels_are_subsampled(self):
        # full set: 1000 supported, 40 contradicted, 160 unsupported; two records of each in the sample.
        # squared errors: supported 0.01 and 0.09 (mean 0.05), contradicted 0.64 and 0.36 (mean 0.50),
        # unsupported 0.04 and 0.00 (mean 0.02)
        full = {"SUPPORTED": 1000, "CONTRADICTED": 40, "UNSUPPORTED": 160}
        probabilities = [0.9, 0.7, 0.8, 0.6, 0.2, 0.0]
        labels = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "CONTRADICTED", "UNSUPPORTED", "UNSUPPORTED"]
        human = [label == "SUPPORTED" for label in labels]
        pi = 200 / 1200
        per_label = reweighted_brier(probabilities, human, pi, labels=labels, full_label_counts=full)
        assert per_label == pytest.approx((1000 * 0.05 + 40 * 0.50 + 160 * 0.02) / 1200)
        # pooling the two labels weighs the contradicted half of the not-supported class instead of a fifth
        pooled = reweighted_brier(probabilities, human, pi)
        assert pooled == pytest.approx(pi * 0.26 + (1 - pi) * 0.05)
        assert pooled > per_label
        # a label of the full set with no record: nothing to weigh it with
        without = [i for i, label in enumerate(labels) if label != "CONTRADICTED"]
        assert (
            reweighted_brier(
                [probabilities[i] for i in without],
                [human[i] for i in without],
                pi,
                labels=[labels[i] for i in without],
                full_label_counts=full,
            )
            is None
        )
        # a label the full set does not hold is not a stratum
        assert reweighted_brier(
            [probabilities[i] for i in without],
            [human[i] for i in without],
            160 / 1160,
            labels=[labels[i] for i in without],
            full_label_counts={**full, "CONTRADICTED": 0},
        ) == pytest.approx((1000 * 0.05 + 160 * 0.02) / 1160)

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
        assert m["minority_count"] == 4 and m["kappa_interpretable"] == 0.0  # four unsupported: below ten

    def test_minority_count_and_interpretability_follow_the_calibrator_threshold(self):
        ten = binary_metrics([True] * 10 + [False] * 10, [True] * 10 + [False] * 10)
        nine = binary_metrics([True] * 11 + [False] * 9, [True] * 11 + [False] * 9)
        assert (ten["minority_count"], ten["kappa_interpretable"]) == (10, 1.0)
        assert (nine["minority_count"], nine["kappa_interpretable"]) == (9, 0.0)
        assert binary_metrics([], [])["minority_count"] == 0

    @pytest.mark.asyncio
    async def test_breakdowns_carry_minority_count_and_interpretability(self, tmp_path: Path):
        rows = study_like_rows()
        for i, r in enumerate(rows):
            r["generator"] = "gen-small" if i < 20 else "gen-large"  # gen-small: 20 supported only
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=0, by=["generator"]
        )
        by = run.report["by"]["generator"]
        assert (by["gen-small"]["minority_count"], by["gen-small"]["kappa_interpretable"]) == (0, 0.0)
        assert (by["gen-large"]["minority_count"], by["gen-large"]["kappa_interpretable"]) == (4, 0.0)
        assert run.report["binary"]["minority_count"] == 4 and run.report["binary"]["kappa_interpretable"] == 0.0
        md = run.report_md.read_text()
        assert "| Minority human class (n) | 4 |" in md and "| Kappa interpretable (minority >= 10) | no |" in md
        assert "| Value | n | Minority class n | Kappa interpretable (>= 10) |" in md
        assert "| gen-large | 42 | 4 | no |" in md and "| gen-small | 20 | 0 | no |" in md

    def test_kappa_is_null_when_judge_and_humans_use_a_single_class(self):
        """pe = 1: no chance correction is possible; None, never 1.0 (the gpt-4 cell of the acceptance breakdown)."""
        m = binary_metrics([True] * 20, [True] * 20)
        assert m["kappa"] is None and m["accuracy"] == 1.0 and m["balanced_accuracy"] == 0.5
        assert binary_metrics([False] * 5, [True] * 5)["kappa"] == 0.0  # unanimous against unanimous: defined
        assert binary_metrics([], [])["kappa"] is None

    @pytest.mark.asyncio
    async def test_report_and_breakdown_print_an_undefined_kappa(self, tmp_path: Path):
        rows = study_like_rows()[:34]  # all SUPPORTED, all passed by the ordinal judge
        for r in rows:
            r["generator"] = "gen-a"
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=5, by=["generator"]
        )
        assert run.report["binary"]["kappa"] is None
        assert run.report["by"]["generator"]["gen-a"]["kappa"] is None
        assert run.report["bootstrap"]["intervals"]["kappa"] is None  # every draw is single-class
        md = run.report_md.read_text()
        assert "| Kappa | undefined |" in md and "| gen-a | 34 | 0 | no | 0.000 | 0.500 | undefined |" in md
        assert json.loads(run.report_json.read_text())["binary"]["kappa"] is None

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


class TestRecallByLabel:
    @staticmethod
    def _table() -> tuple[list[bool], list[str], list[str]]:
        """Six groups of one label each: 8 contradicted (2 detected), 8 unsupported (6 detected), 10 supported (9 passed)."""
        spec = [
            ("g-c1", "CONTRADICTED", [False, False, True, True]),  # False: rejected, that is detected
            ("g-c2", "CONTRADICTED", [True, True, True, True]),
            ("g-u1", "UNSUPPORTED", [False, False, False, False]),
            ("g-u2", "UNSUPPORTED", [False, False, True, True]),
            ("g-s1", "SUPPORTED", [True, True, True, True, False]),
            ("g-s2", "SUPPORTED", [True, True, True, True, True]),
        ]
        judge: list[bool] = []
        labels: list[str] = []
        groups: list[str] = []
        for group, label, verdicts in spec:
            judge += verdicts
            labels += [label] * len(verdicts)
            groups += [group] * len(verdicts)
        return judge, labels, groups

    def test_hand_computed_table(self):
        judge, labels, groups = self._table()
        block = recall_by_label(judge, labels, groups, 0.25, seed=0, resamples=0)
        assert block["threshold"] == 0.25 and block["sample_dependent"] is False
        assert block["labels"] == {
            "CONTRADICTED": {"n": 8, "detected": 2, "recall": 0.25, "ci95": None, "excluded_draws": 0},
            "UNSUPPORTED": {"n": 8, "detected": 6, "recall": 0.75, "ci95": None, "excluded_draws": 0},
        }
        assert block["specificity"] == {
            "label": "SUPPORTED",
            "n": 10,
            "recognized": 9,
            "value": 0.9,
            "ci95": None,
            "excluded_draws": 0,
        }
        assert block["bootstrap"] == {"resamples": 0, "seed": 0, "groups": 6, "reliable": True}
        # the pooled recall of `binary` is the mean of the two, weighted by their counts
        pooled = binary_metrics(judge, [label == "SUPPORTED" for label in labels])["recall_not_supported"]
        assert pooled == pytest.approx((8 * 0.25 + 8 * 0.75) / 16)

    def test_a_label_absent_from_the_sample_keeps_its_key(self):
        judge, labels, groups = self._table()
        keep = [i for i, label in enumerate(labels) if label != "CONTRADICTED"]
        block = recall_by_label(
            [judge[i] for i in keep], [labels[i] for i in keep], [groups[i] for i in keep], 0.5, seed=3, resamples=50
        )
        assert block["labels"]["CONTRADICTED"] == {
            "n": 0,
            "detected": 0,
            "recall": None,
            "ci95": None,
            "excluded_draws": 0,  # no interval to compute: nothing excluded from it
        }
        assert block["labels"]["UNSUPPORTED"]["recall"] == 0.75 and block["labels"]["UNSUPPORTED"]["ci95"] is not None
        assert list(block["labels"]) == ["CONTRADICTED", "UNSUPPORTED"]

    def test_intervals_come_from_the_draws_of_the_main_bootstrap_and_count_those_without_the_label(self):
        judge, labels, groups = self._table()
        seed, resamples = 11, 200
        block = recall_by_label(judge, labels, groups, 0.25, seed=seed, resamples=resamples)

        # the same draws, generated here as grouped_bootstrap does: six groups drawn with replacement
        order = list(dict.fromkeys(groups))
        members = {g: [i for i, name in enumerate(groups) if name == g] for g in order}
        rng = random.Random(seed)
        expected: dict[str, list[float]] = {"CONTRADICTED": [], "UNSUPPORTED": [], "SUPPORTED": []}
        missing = dict.fromkeys(expected, 0)
        single_class = 0
        for _ in range(resamples):
            drawn = [i for _ in order for i in members[order[rng.randrange(len(order))]]]
            human = [labels[i] == "SUPPORTED" for i in drawn]
            single_class += all(human) or not any(human)
            for label in expected:
                verdicts = [judge[i] for i in drawn if labels[i] == label]
                if not verdicts:
                    missing[label] += 1
                    continue
                agreeing = sum(1 for v in verdicts if v == (label == "SUPPORTED"))
                expected[label].append(agreeing / len(verdicts))

        def interval(values: list[float]) -> list[float]:
            values = sorted(values)
            return [
                values[max(0, math.floor(0.025 * (len(values) - 1)))],
                values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))],
            ]

        for label in ("CONTRADICTED", "UNSUPPORTED"):
            assert block["labels"][label]["excluded_draws"] == missing[label] > 0
            assert block["labels"][label]["ci95"] == pytest.approx(interval(expected[label]))
        assert block["specificity"]["excluded_draws"] == missing["SUPPORTED"] > 0
        assert block["specificity"]["ci95"] == pytest.approx(interval(expected["SUPPORTED"]))
        # with two groups per label out of six, a label is absent from about (4/6)**6 = 9% of the draws
        assert block["bootstrap"]["reliable"] is all(count <= 0.10 * resamples for count in missing.values())
        # the main bootstrap ran on the same sequence: it excluded exactly the single-class draws counted above
        main = grouped_bootstrap(judge, [label == "SUPPORTED" for label in labels], groups, seed, resamples)
        assert main["excluded_single_class_draws"] == single_class
        assert recall_by_label(judge, labels, groups, 0.25, seed=seed, resamples=resamples) == block  # seeded

    def test_interval_collapses_when_every_group_has_the_same_recall_and_flags_many_lost_draws(self):
        # every contradicted group is detected at 50%, whatever the draw; one unsupported group out of ten
        judge: list[bool] = []
        labels: list[str] = []
        groups: list[str] = []
        for g in range(5):
            judge += [False, True]
            labels += ["CONTRADICTED"] * 2
            groups += [f"c{g}"] * 2
        for g in range(4):
            judge += [True, True]
            labels += ["SUPPORTED"] * 2
            groups += [f"s{g}"] * 2
        judge += [False]
        labels += ["UNSUPPORTED"]
        groups += ["u0"]
        block = recall_by_label(judge, labels, groups, 0.5, seed=1, resamples=300)
        assert block["labels"]["CONTRADICTED"]["ci95"] == [0.5, 0.5]
        # the single unsupported group is absent from about (9/10)**10 = 35% of the draws
        assert block["labels"]["UNSUPPORTED"]["excluded_draws"] > 30
        assert block["labels"]["UNSUPPORTED"]["ci95"] == [1.0, 1.0]
        assert block["bootstrap"]["reliable"] is False


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
        assert report["threshold"] == {
            "value": 0.75,
            "source": "verdict_policy",
            "policy_threshold": 0.75,
            "rule": "PASS if score >= 0.75 (source: verdict_policy); PARTIAL and FAIL counted as not supported",
        }
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
        """An interrupted run (attempts file cut after 4 rows) resumes on the 6 missing ids only."""
        rows = study_like_rows()[:10]
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        first = ScriptedOrdinalJudge()
        complete = await run_benchmark(first, records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert len(first.received) == 10
        kept = complete.attempts_path.read_text().splitlines()[:5]  # header + 4 attempts
        complete.attempts_path.write_text("\n".join(kept) + "\n")

        second = ScriptedOrdinalJudge()
        run = await run_benchmark(second, records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert len(second.received) == 6  # only the 6 missing ids were judged
        assert run.report["records"]["scored"] == 10
        lines = [json.loads(line) for line in run.attempts_path.read_text().splitlines()]
        assert lines[0] == {
            "header": 1,
            "config_hash": run.report["config_hash"],
            "criterion": "faithfulness",
            "level": "sentence",
            "sample": None,
            "per_class": None,
            "seed": 0,
            "stratify": "label",
            "data_sha256": records_sha256(records),  # no file digest given: the records' content
            "source_commit": None,  # the scripted judge's configuration carries no commit
        }
        assert sum(1 for row in lines if "header" in row) == 1  # the header is written once
        assert [row["sample_id"] for row in lines[1:5]] == [r.id for r in records[:4]]
        assert all(row["config_hash"] == run.report["config_hash"] for row in lines[1:])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "second", "field_name"),
        [
            ({"judge": "ordinal"}, {"judge": "probability"}, "config_hash"),
            ({"seed": 0}, {"seed": 1}, "seed"),
            ({"sample": 4}, {"sample": 3}, "sample"),
            ({"stratify": "label"}, {"stratify": None}, "stratify"),
            ({"criterion": "faithfulness"}, {"criterion": "relevance"}, "criterion"),
            ({"data_sha256": "a" * 64}, {"data_sha256": "b" * 64}, "data_sha256"),
        ],
    )
    async def test_resume_refuses_another_run_and_names_the_field(self, tmp_path: Path, first, second, field_name):
        """Same --out, another judge, seed, sample, stratify, criterion or data file: refused, nothing mixed."""
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:6]))
        judges = {"ordinal": ScriptedOrdinalJudge, "probability": ScriptedProbabilityJudge}

        async def run(params: dict[str, object]):
            judge = judges[str(params.pop("judge", "ordinal"))]()
            return await run_benchmark(judge, records, out_dir=tmp_path / "out", bootstrap_resamples=0, **params), judge

        await run(dict(first))
        before = (tmp_path / "out" / "attempts.jsonl").read_text()
        with pytest.raises(AttemptsMismatchError, match=f"another {field_name} ") as excinfo:
            await run(dict(second))
        assert "attempts.jsonl" in str(excinfo.value)
        assert (tmp_path / "out" / "attempts.jsonl").read_text() == before  # untouched

    @pytest.mark.asyncio
    async def test_resume_refuses_a_row_with_another_config_hash_or_no_header(self, tmp_path: Path):
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:4]))
        out = tmp_path / "out"
        await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=out, bootstrap_resamples=0)
        path = out / "attempts.jsonl"
        lines = path.read_text().splitlines()
        tampered = json.loads(lines[2])
        tampered["config_hash"] = "0" * 16
        path.write_text("\n".join([*lines[:2], json.dumps(tampered), *lines[3:]]) + "\n")
        with pytest.raises(AttemptsMismatchError, match="line 3: config_hash"):
            await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=out, bootstrap_resamples=0)

        path.write_text("\n".join(lines[1:]) + "\n")  # rows without the header line
        with pytest.raises(AttemptsMismatchError, match="no header line"):
            await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=out, bootstrap_resamples=0)
        assert len(read_attempts(path)) == 4  # a plain read still works, the header is skipped

    @pytest.mark.asyncio
    async def test_dev_attempts_have_their_own_header_checked_on_the_dev_file(self, tmp_path: Path):
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:4]))
        dev = load_records(write_jsonl(tmp_path / "dev.jsonl", study_like_rows()[10:14]))
        out = tmp_path / "out"
        kwargs = {"dev_records": dev, "bootstrap_resamples": 0, "sample": 2}
        await run_benchmark(ScriptedProbabilityJudge(), records, out_dir=out, dev_sha256="d" * 64, **kwargs)
        header = json.loads((out / "dev_attempts.jsonl").read_text().splitlines()[0])
        assert header["data_sha256"] == "d" * 64
        assert (header["sample"], header["seed"], header["stratify"]) == (None, None, None)  # dev is never sampled
        await run_benchmark(ScriptedProbabilityJudge(), records, out_dir=out, dev_sha256="d" * 64, **kwargs)
        with pytest.raises(AttemptsMismatchError, match=r"dev_attempts\.jsonl was produced with another data_sha256"):
            await run_benchmark(ScriptedProbabilityJudge(), records, out_dir=out, dev_sha256="e" * 64, **kwargs)

    @pytest.mark.asyncio
    async def test_python_api_is_protected_by_the_records_digest(self, tmp_path: Path):
        """Without a file digest, a changed record (same ids) is refused on resume, and the dev set too."""
        rows = study_like_rows()[:6]
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        dev = load_records(write_jsonl(tmp_path / "dev.jsonl", study_like_rows()[10:14]))
        out = tmp_path / "out"
        await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=out, dev_records=dev, bootstrap_resamples=0)
        dev_header = json.loads((out / "dev_attempts.jsonl").read_text().splitlines()[0])
        assert dev_header["data_sha256"] == records_sha256(dev)
        assert dev_header["data_sha256"] != records_sha256(records)

        rows[3] = dict(rows[3], sentence="Another text under the same id.")
        changed = load_records(write_jsonl(tmp_path / "test2.jsonl", rows))
        assert [r.id for r in changed] == [r.id for r in records] and records_sha256(changed) != records_sha256(records)
        with pytest.raises(AttemptsMismatchError, match=r"attempts\.jsonl was produced with another data_sha256"):
            await run_benchmark(ScriptedOrdinalJudge(), changed, out_dir=out, dev_records=dev, bootstrap_resamples=0)
        with pytest.raises(AttemptsMismatchError, match=r"dev_attempts\.jsonl was produced with another data_sha256"):
            await run_benchmark(
                ScriptedOrdinalJudge(), records, out_dir=out, dev_records=dev[:3], bootstrap_resamples=0
            )

    def test_records_sha256_is_canonical(self):
        records = load_records_from_rows(study_like_rows()[:3])
        again = load_records_from_rows(study_like_rows()[:3])
        assert records_sha256(records) == records_sha256(again) and len(records_sha256(records)) == 64
        assert records_sha256(records[::-1]) != records_sha256(records)  # order is part of the content
        assert records_sha256([]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    @pytest.mark.asyncio
    async def test_report_gives_the_share_of_inputs_beyond_the_declared_training_length(self, tmp_path: Path):
        rows = study_like_rows()[:5]
        rows[1]["context"] = " ".join(["word"] * 30)  # 30 + sentence words > 20
        rows[3]["context"] = " ".join(["word"] * 40)
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(
            ScriptedProbabilityJudge(training_length=20), records, out_dir=tmp_path / "out", bootstrap_resamples=0
        )
        length = run.report["input_length"]
        assert length["training_length"] == 20 and length["with_token_count"] == 5
        assert length["over_training_length"] == 2 and length["share_over_training_length"] == pytest.approx(0.4)
        assert length["max_prompt_tokens"] >= 40 and length["kind"] == "training length"
        assert "inputs beyond the judge's declared training length (20 tokens): 2 of 5 with a token count (40.0%)" in (
            run.report_md.read_text()
        )

        capped = await run_benchmark(
            ScriptedProbabilityJudge(training_length=20, training_length_kind="official truncation length"),
            records,
            out_dir=tmp_path / "out3",
            bootstrap_resamples=0,
        )
        assert capped.report["input_length"]["kind"] == "official truncation length"
        assert "declared official truncation length (20 tokens)" in capped.report_md.read_text()

        silent = await run_benchmark(
            ScriptedProbabilityJudge(), records, out_dir=tmp_path / "out2", bootstrap_resamples=0
        )
        assert silent.report["input_length"] is None  # the judge declares no training length
        assert "training length" not in silent.report_md.read_text()

    @pytest.mark.asyncio
    async def test_resume_refuses_another_known_source_commit_but_tolerates_an_unknown_one(self, tmp_path: Path):
        from dataclasses import replace

        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:4]))

        class Committed(ScriptedOrdinalJudge):
            commit: str | None = "a" * 40

            def config(self, criterion):
                return replace(super().config(criterion), source_commit=self.commit)

        first = Committed()
        run = await run_benchmark(first, records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        header = json.loads(run.attempts_path.read_text().splitlines()[0])
        assert header["source_commit"] == "a" * 40
        assert first.config("faithfulness").config_hash == ScriptedOrdinalJudge().config("faithfulness").config_hash

        same = Committed()
        await run_benchmark(same, records, out_dir=tmp_path / "out", bootstrap_resamples=0)  # resumed
        assert same.received == []

        unknown = Committed()
        unknown.commit = None
        await run_benchmark(unknown, records, out_dir=tmp_path / "out", bootstrap_resamples=0)  # a wheel may resume
        assert unknown.received == []

        other = Committed()
        other.commit = "b" * 40
        with pytest.raises(
            AttemptsMismatchError, match="another source commit \\(aaaaaaaaaaaa, this run: bbbbbbbbbbbb\\)"
        ):
            await run_benchmark(other, records, out_dir=tmp_path / "out", bootstrap_resamples=0)

    async def test_per_class_run_flags_raw_metrics_and_adds_adjusted_ones(self, tmp_path: Path):
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()))
        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", per_class=6, seed=1, bootstrap_resamples=0
        )
        report = run.report
        assert report["records"]["evaluated"] == 10 and report["sample"] == {
            "per_class": 6,
            "taken": {"CONTRADICTED": 2, "SUPPORTED": 6, "UNSUPPORTED": 2},
            "seed": 1,
        }
        assert report["sample_dependent_metrics"] is True
        adjusted = report["prevalence_adjusted"]
        assert adjusted["full_set_not_supported_prevalence"] == pytest.approx(4 / 62)  # the full set, not the sample
        assert adjusted["recall_not_supported"] == pytest.approx(0.5)  # 2 of the 4 unsupported are caught
        assert 0.0 <= adjusted["specificity"] <= 1.0 and adjusted["accuracy_reweighted"] is not None
        header = json.loads(run.attempts_path.read_text().splitlines()[0])
        assert header["per_class"] == 6 and header["sample"] is None
        md = run.report_md.read_text()
        assert (
            "Class quotas: at most 6 records per label, taken CONTRADICTED 2, SUPPORTED 6, UNSUPPORTED 2 (seed 1)" in md
        )
        assert (
            "## Prevalence-adjusted (independent of the quota)" in md
            and "Not-supported prevalence in the full set: 6.5%." in md
        )
        # resuming with another quota is refused like any other sampling parameter
        with pytest.raises(AttemptsMismatchError, match="another per_class"):
            await run_benchmark(
                ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", per_class=5, seed=1, bootstrap_resamples=0
            )
        with pytest.raises(ValueError, match="exclusive"):
            await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out2", per_class=5, sample=10)

        plain = await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out3", bootstrap_resamples=0)
        assert plain.report["sample_dependent_metrics"] is False and plain.report["prevalence_adjusted"] is None
        assert "Prevalence-adjusted" not in plain.report_md.read_text()

    def test_attempts_header_fields_and_file_sha256(self, tmp_path: Path):
        header = attempts_header("h", "faithfulness", "sentence", sample=3, seed=1, stratify=None, data_sha256="x")
        assert set(ATTEMPTS_HEADER_FIELDS) <= set(header) and header["header"] == 1
        path = tmp_path / "f"
        path.write_bytes(b"abc")
        assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    @pytest.mark.asyncio
    async def test_report_states_the_prediction_rule_and_the_harness_memory(self, tmp_path: Path):
        records = load_records(write_jsonl(tmp_path / "test.jsonl", study_like_rows()[:6]))

        async def memory() -> dict[str, object]:
            return {
                "source": "ollama /api/ps",
                "model": "m:7b",
                "available": True,
                "size_mb": 4096.0,
                "size_vram_mb": 4000.0,
            }

        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=0, model_memory=memory
        )
        assert run.report["threshold"]["rule"] == (
            "PASS if score >= 0.75 (source: verdict_policy); PARTIAL and FAIL counted as not supported"
        )
        assert "peak_rss_mb" not in run.report and run.report["peak_rss_mb_harness"] > 0
        assert run.report["model_memory"]["size_mb"] == 4096.0
        md = run.report_md.read_text()
        assert (
            "Prediction rule: PASS if score >= 0.75 (source: verdict_policy); PARTIAL and FAIL counted as not supported."
            in md
        )
        assert "harness process only; an Ollama model runs in another process" in md
        assert "model memory (ollama /api/ps): m:7b, 4096.0 MiB resident, 4000.0 MiB in VRAM" in md

        plain = await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out2", bootstrap_resamples=0)
        assert plain.report["model_memory"] is None and "model memory" not in plain.report_md.read_text()

    @pytest.mark.asyncio
    async def test_out_of_memory_in_a_local_judge_is_a_recorded_failure(self, tmp_path: Path):
        from ragnarok_ai.judges import HHEMJudge
        from tests.unit.judges.test_memory import HHEMBackend

        rows = [
            sentence_row(0, "SUPPORTED", "Fits."),
            sentence_row(1, "SUPPORTED", "Does not fit (oom)."),
            sentence_row(2, "UNSUPPORTED", "Fits too."),
        ]
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        judge = HHEMJudge(backend=HHEMBackend(), device="mps", unit="answer")
        run = await run_benchmark(judge, records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert run.report["records"] == {"total": 3, "evaluated": 3, "scored": 2, "failed": 1}
        assert run.report["failures"] == {"backend": 1}
        written = [json.loads(line) for line in run.attempts_path.read_text().splitlines()[1:]]
        assert [row["error"]["kind"] if row.get("error") else None for row in written] == [None, "backend", None]
        assert "out of memory on device 'mps'" in written[1]["error"]["message"]
        assert "failures by kind: {'backend': 1}" in run.report_md.read_text()

    @pytest.mark.asyncio
    async def test_per_class_run_flags_calibration_and_reweights_brier(self, tmp_path: Path):
        rows = (
            [sentence_row(i, "SUPPORTED", "ok p=0.9") for i in range(20)]
            + [sentence_row(100 + i, "UNSUPPORTED", "bad p=0.2") for i in range(2)]
            + [sentence_row(200 + i, "CONTRADICTED", "bad p=0.6") for i in range(2)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        quota = await run_benchmark(
            ScriptedProbabilityJudge(), records, out_dir=tmp_path / "quota", per_class=2, seed=1, bootstrap_resamples=0
        )
        calibration = quota.report["calibration"]
        assert calibration["sample_dependent"] is True
        assert calibration["brier"] == pytest.approx((2 * 0.01 + 2 * 0.04 + 2 * 0.36) / 6)  # the sample: 2 per label
        pi = 4 / 24
        assert quota.report["prevalence_adjusted"]["brier_reweighted"] == pytest.approx(pi * 0.20 + (1 - pi) * 0.01)
        assert json.loads(quota.report_json.read_text())["calibration"]["sample_dependent"] is True
        md = quota.report_md.read_text()
        assert "Calibration (sample-dependent, follows the class quotas): Brier 0.1367, ECE (10 bins)" in md
        assert "| Brier, reweighted to the full set | 0.0417 |" in md
        assert "ECE is not reweighted" in md

        plain = await run_benchmark(
            ScriptedProbabilityJudge(), records, out_dir=tmp_path / "plain", bootstrap_resamples=0
        )
        assert plain.report["calibration"]["sample_dependent"] is False
        # on the full set the raw Brier is the reweighted one of the quota run: same scores, same class mix
        assert plain.report["calibration"]["brier"] == pytest.approx(
            quota.report["prevalence_adjusted"]["brier_reweighted"]
        )
        plain_md = plain.report_md.read_text()
        assert "Calibration: Brier" in plain_md and "sample-dependent" not in plain_md

        # a judge without probabilities has no calibration block, and nothing to reweight
        ordinal = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "ordinal", per_class=2, seed=1, bootstrap_resamples=0
        )
        assert "calibration" not in ordinal.report and "brier_reweighted" not in ordinal.report["prevalence_adjusted"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    async def test_per_class_run_is_free_of_the_mix_between_the_two_negative_labels(self, tmp_path: Path, seed: int):
        # the judge accepts every contradicted sentence and rejects every unsupported one: the per-label
        # recalls are 0 and 1 whatever the draw, so the adjusted figures must equal the full run's exactly
        rows = (
            [sentence_row(i, "SUPPORTED", f"Supported {i}.") for i in range(60)]
            + [sentence_row(100 + i, "CONTRADICTED", f"Contradicted but accepted {i}.") for i in range(10)]
            + [sentence_row(200 + i, "UNSUPPORTED", f"Unsupported and rejected {i} (raw 1).") for i in range(40)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        full = await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "full", bootstrap_resamples=0)
        assert full.report["binary"]["recall_not_supported"] == pytest.approx(0.8)  # 40 of the 50 not supported

        quota = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "quota", per_class=5, seed=seed, bootstrap_resamples=0
        )
        # both negative labels are subsampled, at different rates: 5 of 10 and 5 of 40
        assert quota.report["sample"]["taken"] == {"CONTRADICTED": 5, "SUPPORTED": 5, "UNSUPPORTED": 5}
        assert quota.report["binary"]["recall_not_supported"] == pytest.approx(0.5)  # the raw figure follows the quota
        adjusted = quota.report["prevalence_adjusted"]
        assert adjusted["strata"] == "label" and adjusted["missing_labels"] == []
        assert adjusted["recall_not_supported"] == pytest.approx(0.8)
        assert adjusted["precision_not_supported_reweighted"] == pytest.approx(
            full.report["binary"]["precision_not_supported"]
        )
        assert adjusted["accuracy_reweighted"] == pytest.approx(full.report["binary"]["accuracy"])
        md = quota.report_md.read_text()
        assert "Rates are estimated per human label and combined with the labels' shares in the full set" in md
        assert "| CONTRADICTED | not supported | 10 (9.1%) | 5 | 0.000 |" in md
        assert "| UNSUPPORTED | not supported | 40 (36.4%) | 5 | 1.000 |" in md
        assert "| SUPPORTED | supported | 60 (54.5%) | 5 | 1.000 |" in md
        assert "| Recall (not supported) | 0.800 |" in md

    @pytest.mark.asyncio
    async def test_per_class_run_with_a_label_left_without_a_scored_record(self, tmp_path: Path):
        rows = (
            [sentence_row(i, "SUPPORTED", f"Supported {i}.") for i in range(12)]
            + [sentence_row(100 + i, "CONTRADICTED", f"Unreadable {i} (fail-parse).") for i in range(3)]
            + [sentence_row(200 + i, "UNSUPPORTED", f"Rejected {i} (raw 1).") for i in range(6)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", per_class=4, seed=1, bootstrap_resamples=0
        )
        assert run.report["failures"] == {"parse": 3}
        adjusted = run.report["prevalence_adjusted"]
        assert adjusted["missing_labels"] == ["CONTRADICTED"] and adjusted["labels"]["CONTRADICTED"]["n"] == 0
        assert adjusted["recall_not_supported"] is None and adjusted["accuracy_reweighted"] is None
        assert adjusted["specificity"] == pytest.approx(1.0)
        md = run.report_md.read_text()
        assert "No scored record for CONTRADICTED: the figures that need its rate are not estimated (n/a)." in md
        assert "| CONTRADICTED | not supported | 3 (14.3%) | 0 | n/a |" in md
        assert "| Recall (not supported) | n/a |" in md

    @pytest.mark.asyncio
    async def test_per_class_run_on_an_answer_level_set_keeps_the_two_classes_as_strata(self, tmp_path: Path):
        rows = [
            {
                "id": f"p{i}",
                "answer": f"Good answer {i}." + (" (raw 1)" if i < 2 else ""),
                "context": "c",
                "label": "PASS",
            }
            for i in range(12)
        ] + [
            {
                "id": f"f{i}",
                "answer": f"Bad answer {i}." + (" (raw 1)" if i < 4 else ""),
                "context": "c",
                "label": "FAIL",
            }
            for i in range(6)
        ]
        records = load_records(write_jsonl(tmp_path / "answers.jsonl", rows))
        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", per_class=6, seed=3, bootstrap_resamples=0
        )
        assert run.report["level"] == "answer" and run.report["sample"]["taken"] == {"FAIL": 6, "PASS": 6}
        adjusted = run.report["prevalence_adjusted"]
        # the quota strata are the two classes: the figures are those of the two-class formulas, as before
        assert (adjusted["strata"], adjusted["labels"], adjusted["missing_labels"]) == ("class", None, [])
        passed = [json.loads(line) for line in run.attempts_path.read_text().splitlines()[1:]]
        kept = sum(1 for row in passed if row["sample_id"].startswith("p") and row["result"]["raw_score"] == 5)
        recall, specificity, pi = 4 / 6, kept / 6, 6 / 18
        assert adjusted["recall_not_supported"] == pytest.approx(recall)
        assert adjusted["specificity"] == pytest.approx(specificity)
        assert adjusted["precision_not_supported_reweighted"] == pytest.approx(
            pi * recall / (pi * recall + (1 - pi) * (1 - specificity))
        )
        assert adjusted["accuracy_reweighted"] == pytest.approx(pi * recall + (1 - pi) * specificity)
        assert "Rates are estimated per human label" not in run.report_md.read_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    async def test_reweighted_brier_of_a_quota_run_equals_the_brier_of_the_full_run(self, tmp_path: Path, seed: int):
        # one probability per label, so each label's squared error is the same whatever the draw:
        # supported 0.01, contradicted 0.36, unsupported 0.04
        rows = (
            [sentence_row(i, "SUPPORTED", "ok p=0.9") for i in range(60)]
            + [sentence_row(100 + i, "CONTRADICTED", "bad p=0.6") for i in range(10)]
            + [sentence_row(200 + i, "UNSUPPORTED", "bad p=0.2") for i in range(40)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        full = await run_benchmark(
            ScriptedProbabilityJudge(), records, out_dir=tmp_path / "full", bootstrap_resamples=0
        )
        on_full_set = (60 * 0.01 + 10 * 0.36 + 40 * 0.04) / 110
        assert full.report["calibration"]["brier"] == pytest.approx(on_full_set)

        quota = await run_benchmark(
            ScriptedProbabilityJudge(),
            records,
            out_dir=tmp_path / "quota",
            per_class=5,
            seed=seed,
            bootstrap_resamples=0,
        )
        # both negative labels are subsampled, at different rates: 5 of 10 and 5 of 40
        assert quota.report["sample"]["taken"] == {"CONTRADICTED": 5, "SUPPORTED": 5, "UNSUPPORTED": 5}
        assert quota.report["calibration"]["brier"] == pytest.approx((0.01 + 0.36 + 0.04) / 3)  # the sample's
        assert quota.report["prevalence_adjusted"]["brier_reweighted"] == pytest.approx(on_full_set)
        # on two classes the contradicted would weigh half of the not-supported class instead of a fifth
        pi = 50 / 110
        assert pi * 0.20 + (1 - pi) * 0.01 != pytest.approx(on_full_set)

    @pytest.mark.asyncio
    async def test_reweighted_brier_is_null_when_a_label_has_no_scored_record(self, tmp_path: Path):
        class FailingOnContradicted(ScriptedProbabilityJudge):
            async def judge(self, criterion, *, question, context, answer):
                if "(fail-parse)" in answer:
                    raise JudgeParseError("Unparseable", criterion=criterion, raw_output=answer)
                return await super().judge(criterion, question=question, context=context, answer=answer)

        rows = (
            [sentence_row(i, "SUPPORTED", "ok p=0.9") for i in range(12)]
            + [sentence_row(100 + i, "CONTRADICTED", "bad (fail-parse)") for i in range(3)]
            + [sentence_row(200 + i, "UNSUPPORTED", "bad p=0.2") for i in range(6)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(
            FailingOnContradicted(), records, out_dir=tmp_path / "out", per_class=4, seed=1, bootstrap_resamples=0
        )
        adjusted = run.report["prevalence_adjusted"]
        assert adjusted["missing_labels"] == ["CONTRADICTED"] and adjusted["brier_reweighted"] is None
        assert run.report["calibration"]["brier"] is not None  # the sample's Brier is still there, flagged
        assert "Brier, reweighted" not in run.report_md.read_text()

    @pytest.mark.asyncio
    async def test_report_gives_the_recall_of_each_human_label_at_the_report_threshold(self, tmp_path: Path):
        # dev puts the threshold at 0.65 (as in the dev-threshold test); the policy's own is 0.5
        dev_rows = [sentence_row(i, "SUPPORTED", "ok p=0.7") for i in range(6)] + [
            sentence_row(10 + i, "UNSUPPORTED", "bad p=0.6") for i in range(6)
        ]
        # contradicted at 0.6 (4) and 0.8 (4); unsupported at 0.3 (6) and 0.7 (2); supported at 0.9 (9) and 0.6 (1)
        test_rows = (
            [sentence_row(100 + i, "CONTRADICTED", "bad p=0.6" if i < 4 else "bad p=0.8") for i in range(8)]
            + [sentence_row(200 + i, "UNSUPPORTED", "bad p=0.3" if i < 6 else "bad p=0.7") for i in range(8)]
            + [sentence_row(300 + i, "SUPPORTED", "ok p=0.9" if i < 9 else "ok p=0.6") for i in range(10)]
        )
        dev = load_records(write_jsonl(tmp_path / "dev.jsonl", dev_rows))
        test = load_records(write_jsonl(tmp_path / "test.jsonl", test_rows))
        run = await run_benchmark(
            ScriptedProbabilityJudge(), test, out_dir=tmp_path / "out", dev_records=dev, seed=5, bootstrap_resamples=100
        )
        block = run.report["recall_by_label"]
        assert block["threshold"] == run.report["threshold"]["value"] == 0.65
        # at 0.65 the contradicted at 0.6 are detected; at the policy's 0.5 none would be
        assert (block["labels"]["CONTRADICTED"]["detected"], block["labels"]["CONTRADICTED"]["recall"]) == (4, 0.5)
        assert (block["labels"]["UNSUPPORTED"]["detected"], block["labels"]["UNSUPPORTED"]["recall"]) == (6, 0.75)
        assert (block["specificity"]["recognized"], block["specificity"]["value"]) == (9, 0.9)
        assert block["sample_dependent"] is False
        low, high = block["labels"]["CONTRADICTED"]["ci95"]
        assert 0.0 <= low <= 0.5 <= high <= 1.0
        # same bootstrap as the main intervals
        assert {k: block["bootstrap"][k] for k in ("resamples", "seed", "groups")} == {
            k: run.report["bootstrap"][k] for k in ("resamples", "seed", "groups")
        }
        assert run.report["binary"]["recall_not_supported"] == pytest.approx((4 + 6) / 16)
        assert json.loads(run.report_json.read_text())["recall_by_label"] == block
        md = run.report_md.read_text()
        assert "## Recall by human label" in md
        assert "counts as not supported (score < 0.65). Per-label rates: they do not depend on the class mix." in md
        assert "| Human label | n | Detected | Rate | 95% CI |" in md
        assert f"| CONTRADICTED | 8 | 4 | 0.500 | {low:.3f} to {high:.3f} |" in md
        assert "| SUPPORTED (specificity, passed) | 10 | 9 | 0.900 |" in md
        assert md.index("## Recall by human label") < md.index("## Failures and cost")

        without = await run_benchmark(
            ScriptedProbabilityJudge(), test, out_dir=tmp_path / "out2", bootstrap_resamples=0
        )
        block = without.report["recall_by_label"]
        assert block["threshold"] == 0.5 and block["labels"]["CONTRADICTED"]["recall"] == 0.0
        assert block["labels"]["CONTRADICTED"]["ci95"] is None and block["specificity"]["ci95"] is None
        assert "| CONTRADICTED | 8 | 0 | 0.000 | n/a |" in without.report_md.read_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [0, 1, 2])
    async def test_recall_by_label_and_prevalence_adjusted_give_the_same_rates(self, tmp_path: Path, seed: int):
        # a quota run where both negative labels are subsampled and each label has a mixed outcome
        rows = (
            [sentence_row(i, "SUPPORTED", f"Supported {i}." + (" (raw 1)" if i % 5 == 0 else "")) for i in range(60)]
            + [sentence_row(100 + i, "CONTRADICTED", f"C {i}." + (" (raw 1)" if i % 3 == 0 else "")) for i in range(12)]
            + [sentence_row(200 + i, "UNSUPPORTED", f"U {i}." + (" (raw 1)" if i % 4 else "")) for i in range(40)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(
            ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", per_class=8, seed=seed, bootstrap_resamples=50
        )
        block, adjusted = run.report["recall_by_label"], run.report["prevalence_adjusted"]
        assert adjusted["strata"] == "label"
        for label, entry in block["labels"].items():
            assert (entry["n"], entry["recall"]) == (adjusted["labels"][label]["n"], adjusted["labels"][label]["rate"])
            assert entry["detected"] == round(entry["recall"] * entry["n"])
        supported = adjusted["labels"]["SUPPORTED"]
        assert (block["specificity"]["n"], block["specificity"]["value"]) == (supported["n"], supported["rate"])
        assert block["specificity"]["value"] == adjusted["specificity"]
        # under quotas: same block, not flagged, where the raw metrics are
        assert block["sample_dependent"] is False and run.report["sample_dependent_metrics"] is True
        assert {label: entry["n"] for label, entry in block["labels"].items()} == {"CONTRADICTED": 8, "UNSUPPORTED": 8}

    @pytest.mark.asyncio
    async def test_recall_by_label_with_a_label_left_without_a_scored_record(self, tmp_path: Path):
        rows = (
            [sentence_row(i, "SUPPORTED", f"Supported {i}.") for i in range(6)]
            + [sentence_row(100 + i, "CONTRADICTED", f"Unreadable {i} (fail-parse).") for i in range(3)]
            + [sentence_row(200 + i, "UNSUPPORTED", f"Rejected {i} (raw 1).") for i in range(4)]
        )
        records = load_records(write_jsonl(tmp_path / "test.jsonl", rows))
        run = await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=20)
        block = run.report["recall_by_label"]
        assert block["labels"]["CONTRADICTED"] == {
            "n": 0,
            "detected": 0,
            "recall": None,
            "ci95": None,
            "excluded_draws": 0,
        }
        assert block["labels"]["UNSUPPORTED"]["recall"] == 1.0
        assert "| CONTRADICTED | 0 | 0 | n/a | n/a |" in run.report_md.read_text()

    @pytest.mark.asyncio
    async def test_answer_level_set_has_no_recall_by_label(self, tmp_path: Path):
        rows = [{"id": f"p{i}", "answer": f"Good {i}.", "context": "c", "label": "PASS"} for i in range(4)] + [
            {"id": f"f{i}", "answer": f"Bad {i} (raw 1).", "context": "c", "label": "FAIL"} for i in range(4)
        ]
        records = load_records(write_jsonl(tmp_path / "answers.jsonl", rows))
        run = await run_benchmark(ScriptedOrdinalJudge(), records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        assert run.report["level"] == "answer" and run.report["recall_by_label"] is None
        assert "Recall by human label" not in run.report_md.read_text()

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
        assert without_dev.report["threshold"] == {
            "value": 0.5,
            "source": "verdict_policy",
            "policy_threshold": 0.5,
            "rule": "PASS if score >= 0.50 (source: verdict_policy); PARTIAL and FAIL counted as not supported",
        }

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


class TestOllamaModelMemory:
    @pytest.mark.asyncio
    async def test_reads_the_loaded_model_size_on_api_ps(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/ps"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "other:latest", "model": "other:latest", "size": 1, "size_vram": 1},
                        {
                            "name": "prometheus:7b",
                            "model": "prometheus:7b",
                            "size": 4 * 1024**3,
                            "size_vram": 3 * 1024**3,
                        },
                    ]
                },
            )

        result = await ollama_model_memory(
            "http://ollama.test", "prometheus:7b", transport=httpx.MockTransport(handler)
        )
        assert result == {
            "source": "ollama /api/ps",
            "model": "prometheus:7b",
            "available": True,
            "size_mb": 4096.0,
            "size_vram_mb": 3072.0,
            "reason": None,
        }
        bare = await ollama_model_memory("http://ollama.test", "other", transport=httpx.MockTransport(handler))
        assert bare["available"] and bare["model"] == "other:latest"  # the :latest tag is implied

    @pytest.mark.asyncio
    async def test_says_why_when_unavailable(self):
        import httpx

        empty = httpx.MockTransport(lambda _request: httpx.Response(200, json={"models": []}))
        result = await ollama_model_memory("http://ollama.test", "prometheus:7b", transport=empty)
        assert not result["available"] and result["reason"] == "model not loaded (listed: none)"

        def down(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        result = await ollama_model_memory("http://ollama.test", "prometheus:7b", transport=httpx.MockTransport(down))
        assert not result["available"] and result["reason"] == "ConnectError: refused"
        assert result["size_mb"] is None


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

    def test_cli_refuses_to_resume_after_the_data_file_changed(self, tmp_path: Path):
        rows = study_like_rows()[:4]
        data = write_jsonl(tmp_path / "test.jsonl", rows)
        args = [
            "judge-bench",
            "--judge",
            "python:tests.unit.benchmarks.fake_judge:ScriptedOrdinalJudge",
            "--data",
            str(data),
            "--out",
            str(tmp_path / "out"),
            "--bootstrap",
            "0",
        ]
        assert runner.invoke(app, args).exit_code == 0
        header = json.loads((tmp_path / "out" / "attempts.jsonl").read_text().splitlines()[0])
        assert header["data_sha256"] == file_sha256(data)
        assert runner.invoke(app, args).exit_code == 0  # identical run: resumed
        write_jsonl(data, [*rows, sentence_row(99, "SUPPORTED", "One more sentence.")])
        changed = runner.invoke(app, args)
        assert changed.exit_code == 2 and "attempts.jsonl was produced with another data_sha256" in changed.output
        other_seed = runner.invoke(app, [*args, "--seed", "5"])
        assert other_seed.exit_code == 2 and "another seed (0, this run: 5)" in other_seed.output

    def test_cli_per_class_option_and_exclusivity(self, tmp_path: Path):
        data = write_jsonl(tmp_path / "test.jsonl", study_like_rows())
        base = [
            "--json",
            "judge-bench",
            "--judge",
            "python:tests.unit.benchmarks.fake_judge:ScriptedOrdinalJudge",
            "--data",
            str(data),
            "--bootstrap",
            "0",
        ]
        result = runner.invoke(app, [*base, "--out", str(tmp_path / "o"), "--per-class", "6"])
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)["data"]["report"]
        assert report["sample"]["per_class"] == 6 and report["prevalence_adjusted"] is not None
        both = runner.invoke(app, [*base, "--out", str(tmp_path / "o2"), "--per-class", "6", "--sample", "10"])
        assert both.exit_code == 2 and "exclusive" in both.output
        zero = runner.invoke(app, [*base, "--out", str(tmp_path / "o3"), "--per-class", "0"])
        assert zero.exit_code == 2

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
