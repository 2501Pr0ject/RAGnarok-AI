"""Paired comparison of two ``judge-bench`` runs on the same records.

A ``judge-bench`` report describes one judge. Two reports cannot say
whether one judge is better than the other: their intervals are marginal,
and both judges ran on the same records, so their errors are correlated and
the interval of a difference is not the difference of two intervals. This
module compares two runs record by record.

Inputs: the two run folders (``attempts.jsonl`` with its header line, and
``report.json``, from which only the pass threshold of each judge is read)
and the data file, which holds what the attempts do not: the human labels
and the groups. Every figure is recomputed from the attempts and the data
file; nothing else is read from the reports, whose prevalence-adjusted
figures were biased before 1.14 (#182).

What is refused, with the field named: a data file that is not the one of
the runs (SHA-256 of the headers), two runs of different levels or
criteria, two runs that did not evaluate the same records (different
quotas, samples or seeds of a sample give different records), an
incomplete run. What is accepted and stated: different thresholds (each
judge keeps its own), different seeds on full runs.

Pairing: on the records scored by both judges. A record that failed in
either run is left out and counted, by run and by kind. For claim-level
judges the pair is the answer, as for any judge: the attempts hold the
answer's score, not its claims.

Differences are B minus A. Each interval is a percentile interval of the
differences over the draws of ``_group_draws`` (the draws of every
interval of a report): in a draw both judges are evaluated on the same
resampled records and the difference is taken inside the draw. Under class
quotas the per-label reweighting is redone in each draw for each judge.

One figure is declared the headline: the F1 of the not-supported class,
reweighted to the full set when the runs used class quotas. The verdict is
about that figure only. When the interval of its difference contains 0,
the verdict says that no difference is established and names no judge.
When both judges give probabilities, the difference of AUROC is reported
next to it, as a comparison that does not depend on the thresholds. Every
other figure is exploratory: its interval is not corrected for multiple
comparisons.

An exact McNemar test compares how often the two binary verdicts are
right. It assumes independent records and takes no account of the groups:
when the records are grouped (the sentences of one answer, the answers to
one question), its p-value is too optimistic. The paired bootstrap
interval, which resamples groups, is the one to rely on. On a full run the
test stands next to the headline figure; under class quotas it tests the
accuracy on the sample's class mix, not the set's, and is listed with the
exploratory figures.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.benchmarks.judges import (
    BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    NEGATIVE_CLASS,
    PASS_LABELS,
    _balanced_accuracy,
    _f1_not_supported,
    _group_draws,
    _is_header,
    _json_lines,
    _label_agreement,
    _percentile_interval,
    _probability_of,
    _score_of,
    prevalence_adjusted,
    records_sha256,
    reweighted_brier,
)
from ragnarok_ai.calibration.calibrator import _cohens_kappa

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ragnarok_ai.benchmarks.judges import BenchRecord

HEADLINE_RAW = "f1_not_supported"
HEADLINE_REWEIGHTED = "f1_not_supported_reweighted"
# Figures where a lower value is better: the sign of a difference reads the other way
LOWER_IS_BETTER = frozenset({"brier", "brier_reweighted"})
# Header fields that must be equal in the two runs for the records to be the same
SAME_RECORDS_FIELDS = ("level", "criterion", "data_sha256")
# Header fields that decide which records a run evaluates: named when the two sets of ids differ
SAMPLING_FIELDS = ("per_class", "sample", "stratify", "seed")
ALPHA = 0.05


class ComparisonError(ValueError):
    """The two runs cannot be compared; the message names what differs."""


@dataclass(frozen=True)
class RunFiles:
    """What a ``judge-bench`` run left on disk, as far as a comparison needs it."""

    directory: Path
    header: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    threshold: float
    threshold_source: str
    judge: str
    evaluated: int


@dataclass(frozen=True)
class ComparisonRun:
    """The comparison and where it was written."""

    report: dict[str, Any]
    report_json: Path
    report_md: Path


def load_run(directory: str | Path) -> RunFiles:
    """Read the header, the attempts and the threshold of one run.

    Raises:
        ComparisonError: If the folder holds no attempts file with a
            header, or no report (the run was interrupted before its
            report: its threshold is not known and is not guessed).
    """
    folder = Path(directory)
    attempts = folder / "attempts.jsonl"
    if not attempts.is_file():
        msg = f"{folder}: no attempts.jsonl, not the output folder of a judge-bench run"
        raise ComparisonError(msg)
    lines = _json_lines(attempts)
    if not lines or not _is_header(lines[0][1]):
        msg = f"{attempts}: no header line; the run cannot be identified"
        raise ComparisonError(msg)
    header = lines[0][1]
    rows: dict[str, dict[str, Any]] = {}
    for _, row in lines[1:]:
        if "sample_id" in row:
            rows[str(row["sample_id"])] = row  # a resumed run may repeat an id: the last row is the attempt
    report_path = folder / "report.json"
    if not report_path.is_file():
        msg = (
            f"{folder}: no report.json. The pass threshold of a run is recorded in its report; "
            "finish the run (judge-bench resumes from attempts.jsonl) before comparing it"
        )
        raise ComparisonError(msg)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return RunFiles(
        directory=folder,
        header=header,
        rows=rows,
        threshold=float(report["threshold"]["value"]),
        threshold_source=str(report["threshold"]["source"]),
        judge=str(report.get("judge", "")),
        evaluated=int(report["records"]["evaluated"]),
    )


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two counts of discordant pairs.

    Under the hypothesis that the judges are equally often right, each
    discordant pair goes either way with probability one half: the smaller
    count follows a binomial law. 1.0 without any discordant pair.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    """Probability that a positive record scores above a negative one, ties counting one half.

    ``None`` when one of the two classes is absent.
    """
    n_pos = sum(1 for p in positive if p)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1  # ranks are 1-based; tied scores share their mean rank
        rank_sum += average_rank * sum(1 for k in range(i, j + 1) if positive[order[k]])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_reweighted(
    scores: Sequence[float], labels: Sequence[str], full_label_counts: Mapping[str, int]
) -> float | None:
    """AUROC at the full set's mix of not-supported labels, from a sample drawn by class quotas.

    The AUROC is the probability that a supported record scores above a
    not-supported one. With several not-supported labels it is the mean of
    the per-label probabilities, weighted by the labels' shares among the
    not-supported records; a quota changes those shares, so each label is
    taken on its own and weighted by its count in the full set, as in
    ``prevalence_adjusted``. ``None`` when a label of the full set has no
    record in the sample.
    """
    present = {label for label, count in full_label_counts.items() if count > 0}
    positives = [s for s, label in zip(scores, labels, strict=True) if label in PASS_LABELS]
    negatives = sorted(label for label in present if label not in PASS_LABELS)
    if not positives or not negatives:
        return None
    total = sum(full_label_counts[label] for label in negatives)
    value = 0.0
    for label in negatives:
        own = [s for s, lab in zip(scores, labels, strict=True) if lab == label]
        if not own:
            return None
        part = auroc(positives + own, [True] * len(positives) + [False] * len(own))
        assert part is not None
        value += full_label_counts[label] / total * part
    return value


def figures(
    judge_pass: Sequence[bool],
    human_pass: Sequence[bool],
    labels: Sequence[str],
    probabilities: Sequence[float] | None,
    *,
    level: str,
    quotas: bool,
    full_label_counts: Mapping[str, int],
    full_not_supported_prevalence: float,
) -> dict[str, float | None]:
    """Every compared figure of one judge on one set of records; ``None`` where a figure is undefined.

    The raw figures are those of a ``judge-bench`` report (same functions).
    F1, balanced accuracy and kappa are ``None`` when the humans used a
    single class, as the report's bootstrap excludes such draws.
    """
    out: dict[str, float | None] = {}
    n = len(human_pass)
    both_classes = 0 < sum(human_pass) < n
    out["accuracy"] = (sum(1 for j, h in zip(judge_pass, human_pass, strict=True) if j == h) / n) if n else None
    precision, recall, f1 = _f1_not_supported(judge_pass, human_pass) if both_classes else (None, None, None)
    out[HEADLINE_RAW] = f1
    out["precision_not_supported"] = precision
    out["recall_not_supported"] = recall
    out["balanced_accuracy"] = _balanced_accuracy(judge_pass, human_pass) if both_classes else None
    out["kappa"] = _cohens_kappa(judge_pass, human_pass) if both_classes else None

    agreement = _label_agreement(judge_pass, labels)
    passed = [agreement[label] for label in agreement if label in PASS_LABELS]
    out["specificity"] = (sum(a for a, _ in passed) / sum(m for _, m in passed)) if passed else None
    if level == "sentence":
        for label in sorted(full_label_counts):
            if label not in PASS_LABELS and full_label_counts[label] > 0:
                counted = agreement.get(label)
                out[f"recall_{label}"] = (counted[0] / counted[1]) if counted else None

    by_label = level == "sentence"
    if quotas:
        adjusted = prevalence_adjusted(
            judge_pass,
            human_pass,
            full_not_supported_prevalence,
            labels=labels if by_label else None,
            full_label_counts=full_label_counts if by_label else None,
        )
        out[HEADLINE_REWEIGHTED] = adjusted["f1_not_supported_reweighted"]
        out["recall_not_supported_reweighted"] = adjusted["recall_not_supported"]
        out["precision_not_supported_reweighted"] = adjusted["precision_not_supported_reweighted"]
        out["accuracy_reweighted"] = adjusted["accuracy_reweighted"]

    if probabilities is not None:
        out["auroc"] = auroc(probabilities, human_pass)
        errors = [(p - (1.0 if h else 0.0)) ** 2 for p, h in zip(probabilities, human_pass, strict=True)]
        out["brier"] = (sum(errors) / n) if n else None
        if quotas:
            out["auroc_reweighted"] = (
                auroc_reweighted(probabilities, labels, full_label_counts) if by_label else out["auroc"]
            )
            out["brier_reweighted"] = reweighted_brier(
                probabilities,
                human_pass,
                full_not_supported_prevalence,
                labels=labels if by_label else None,
                full_label_counts=full_label_counts if by_label else None,
            )
    return out


def _check_same_records(a: RunFiles, b: RunFiles, records: Sequence[BenchRecord], data_sha256: str | None) -> None:
    for name in SAME_RECORDS_FIELDS:
        if a.header.get(name) != b.header.get(name):
            msg = (
                f"The two runs differ on {name!r}: {a.header.get(name)!r} for A ({a.directory}) and "
                f"{b.header.get(name)!r} for B ({b.directory}). Only two runs on the same records, at the same "
                "level and on the same criterion can be compared"
            )
            raise ComparisonError(msg)
    accepted = {records_sha256(records)} | ({data_sha256} if data_sha256 else set())
    if a.header.get("data_sha256") not in accepted:
        msg = (
            "The data file is not the one the runs were made on: 'data_sha256' is "
            f"{a.header.get('data_sha256')!r} in the runs' headers and {data_sha256 or records_sha256(records)!r} "
            "for this file. Human labels and groups are read from the data file, so it must be the same"
        )
        raise ComparisonError(msg)
    levels = {r.level for r in records}
    if levels != {a.header.get("level")}:
        msg = f"The data file is at level {sorted(levels)} and the runs at level {a.header.get('level')!r}"
        raise ComparisonError(msg)
    for name, run in (("A", a), ("B", b)):
        if len(run.rows) != run.evaluated:
            msg = (
                f"Run {name} ({run.directory}) is incomplete: its report counts {run.evaluated} evaluated records "
                f"and its attempts file holds {len(run.rows)}"
            )
            raise ComparisonError(msg)
    if set(a.rows) != set(b.rows):
        differing = [
            f"{name} ({a.header.get(name)!r} for A, {b.header.get(name)!r} for B)"
            for name in SAMPLING_FIELDS
            if a.header.get(name) != b.header.get(name)
        ]
        only_a, only_b = len(set(a.rows) - set(b.rows)), len(set(b.rows) - set(a.rows))
        msg = (
            f"The two runs did not evaluate the same records: {only_a} only in A, {only_b} only in B, "
            f"{len(set(a.rows) & set(b.rows))} in both"
            + (f". They differ on {', '.join(differing)}" if differing else "")
            + ". Run both judges with the same --per-class, --sample, --stratify and --seed"
        )
        raise ComparisonError(msg)
    unknown = set(a.rows) - {r.id for r in records}
    if unknown:
        msg = f"{len(unknown)} record id(s) of the runs are not in the data file (for instance {sorted(unknown)[0]!r})"
        raise ComparisonError(msg)


def _notes(a: RunFiles, b: RunFiles) -> list[str]:
    notes: list[str] = []
    if a.threshold != b.threshold:
        notes.append(
            f"Each judge is compared at its own pass threshold: {a.threshold:.2f} for A ({a.threshold_source}) and "
            f"{b.threshold:.2f} for B ({b.threshold_source})."
        )
    a_dev, b_dev = a.threshold_source.startswith("dev"), b.threshold_source.startswith("dev")
    if a_dev != b_dev:
        tuned, default = ("A", "B") if a_dev else ("B", "A")
        notes.append(
            f"The threshold of {tuned} was chosen on a development set and the one of {default} is its verdict "
            "policy's: the two judges are not at comparable operating points, and the threshold-dependent figures "
            "favor the tuned one. The AUROC, when available, does not depend on the thresholds."
        )
    if a.header.get("seed") != b.header.get("seed"):
        notes.append(
            f"The runs were made with different seeds ({a.header.get('seed')} for A, {b.header.get('seed')} for B). "
            "They evaluated the same records, so the seed only drove the bootstrap of each report, which this "
            "comparison does not use."
        )
    if a.header.get("config_hash") == b.header.get("config_hash"):
        notes.append(
            "The two runs have the same judge configuration: this comparison measures the repeatability of one "
            "judge, not a difference between two."
        )
    return notes


def compare_runs(
    run_a: str | Path,
    run_b: str | Path,
    records: Sequence[BenchRecord],
    *,
    out_dir: str | Path,
    data_sha256: str | None = None,
    seed: int = 0,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> ComparisonRun:
    """Compare run B with run A on *records* and write ``report.json`` and ``report.md`` in *out_dir*.

    Args:
        run_a: Output folder of the first ``judge-bench`` run.
        run_b: Output folder of the second one. Differences are B minus A.
        records: The records of the data file both runs were made on.
        out_dir: Where the comparison is written; not one of the two run folders.
        data_sha256: SHA-256 of the data file, as ``judge-bench`` records it;
            without it the canonical digest of *records* is used.
        seed: Seed of the bootstrap draws.
        resamples: Number of draws; 0 gives no interval and no verdict.

    Raises:
        ComparisonError: If the runs cannot be compared (see the module's text).
    """
    out = Path(out_dir)
    a, b = load_run(run_a), load_run(run_b)
    for run in (a, b):
        if out.resolve() == run.directory.resolve():
            msg = f"--out is the folder of a run ({run.directory}): its report.json would be overwritten"
            raise ComparisonError(msg)
    if resamples < 0:
        msg = f"resamples must be at least 0, got {resamples}"
        raise ComparisonError(msg)
    _check_same_records(a, b, records, data_sha256)

    level = str(a.header["level"])
    quotas = a.header.get("per_class") is not None
    full_label_counts = dict(sorted(Counter(r.label for r in records).items()))
    prevalence = sum(1 for r in records if not r.human_pass) / len(records) if records else 0.0

    evaluated = [r for r in records if r.id in a.rows]
    failed = {"a_only": 0, "b_only": 0, "both": 0}
    kinds: dict[str, dict[str, int]] = {"a": {}, "b": {}}
    compared: list[BenchRecord] = []
    for r in evaluated:
        a_ok, b_ok = _score_of(a.rows[r.id]) is not None, _score_of(b.rows[r.id]) is not None
        for key, ok, run in (("a", a_ok, a), ("b", b_ok, b)):
            if not ok:
                kind = (run.rows[r.id].get("error") or {}).get("kind", "unknown")
                kinds[key][kind] = kinds[key].get(kind, 0) + 1
        if a_ok and b_ok:
            compared.append(r)
        else:
            failed["both" if not a_ok and not b_ok else ("a_only" if not a_ok else "b_only")] += 1

    human = [r.human_pass for r in compared]
    labels = [r.label for r in compared]
    groups = [r.group_id for r in compared]
    scores = {key: [float(_score_of(run.rows[r.id])) for r in compared] for key, run in (("a", a), ("b", b))}  # type: ignore[arg-type]
    passes = {key: [s >= run.threshold for s in scores[key]] for key, run in (("a", a), ("b", b))}
    probabilities: dict[str, list[float] | None] = {}
    for key, run in (("a", a), ("b", b)):
        values = [_probability_of(run.rows[r.id]) for r in compared]
        probabilities[key] = [v for v in values if v is not None] if all(v is not None for v in values) else None
    if probabilities["a"] is None or probabilities["b"] is None:
        probabilities = {"a": None, "b": None}  # a threshold-free comparison needs both judges to be probabilistic

    def compute(key: str, indices: Sequence[int] | None) -> dict[str, float | None]:
        def take(values: Sequence[Any] | None) -> Any:
            return values if values is None or indices is None else [values[i] for i in indices]

        return figures(
            take(passes[key]),
            take(human),
            take(labels),
            take(probabilities[key]),
            level=level,
            quotas=quotas,
            full_label_counts=full_label_counts,
            full_not_supported_prevalence=prevalence,
        )

    point = {key: compute(key, None) for key in ("a", "b")} if compared else {"a": {}, "b": {}}
    names = list(point["a"])
    draws: dict[str, list[float]] = {name: [] for name in names}
    excluded = dict.fromkeys(names, 0)
    for drawn in _group_draws(groups, seed, resamples) if compared else ():
        draw_a, draw_b = compute("a", drawn), compute("b", drawn)
        for name in names:
            if point["a"][name] is None or point["b"][name] is None:
                continue  # no point estimate: no interval, and nothing to count
            value_a, value_b = draw_a[name], draw_b[name]
            if value_a is None or value_b is None:
                excluded[name] += 1
            else:
                draws[name].append(value_b - value_a)

    headline_name = HEADLINE_REWEIGHTED if quotas else HEADLINE_RAW
    threshold_free = None
    if probabilities["a"] is not None:
        threshold_free = "auroc_reweighted" if quotas else "auroc"

    def role(name: str) -> str:
        if name == headline_name:
            return "headline"
        return "threshold_free" if name == threshold_free else "exploratory"

    differences: dict[str, dict[str, Any]] = {}
    for name in names:
        value_a, value_b = point["a"][name], point["b"][name]
        difference = None if value_a is None or value_b is None else value_b - value_a
        interval = _percentile_interval(draws[name]) if difference is not None else None
        differences[name] = {
            "role": role(name),
            "a": value_a,
            "b": value_b,
            "difference": difference,
            "interval": interval,
            # True: these data do not separate the two judges on this figure
            "contains_zero": None if interval is None else bool(interval[0] <= 0.0 <= interval[1]),
            "excluded_draws": excluded[name],
            "higher_is_better": name not in LOWER_IS_BETTER,
        }

    correct = {key: [j == h for j, h in zip(passes[key], human, strict=True)] for key in ("a", "b")}
    only_a = sum(1 for x, y in zip(correct["a"], correct["b"], strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(correct["a"], correct["b"], strict=True) if y and not x)
    p_value = mcnemar_exact(only_a, only_b)
    mcnemar = {
        "both_correct": sum(1 for x, y in zip(correct["a"], correct["b"], strict=True) if x and y),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": sum(1 for x, y in zip(correct["a"], correct["b"], strict=True) if not x and not y),
        "discordant": only_a + only_b,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": p_value < ALPHA,
        # Records are taken as independent: with grouped records the p-value is too optimistic
        "assumes_independent_records": True,
        # Under class quotas the test is about the accuracy on the sample's class mix, not the set's
        "role": "exploratory" if quotas else "paired_test",
        "follows_sample_class_mix": quotas,
    }

    report: dict[str, Any] = {
        "a": _side(a),
        "b": _side(b),
        "data": {
            "sha256": a.header.get("data_sha256"),
            "level": level,
            "criterion": a.header.get("criterion"),
            "records": len(records),
            "evaluated": len(evaluated),
            "compared": len(compared),
            "per_class": a.header.get("per_class"),
            "sample": a.header.get("sample"),
            "labels_compared": dict(sorted(Counter(labels).items())),
            "labels_full_set": full_label_counts,
        },
        "failures": {**failed, "by_kind": kinds},
        "same_configuration": a.header.get("config_hash") == b.header.get("config_hash"),
        "notes": _notes(a, b),
        "headline": _headline(headline_name, differences.get(headline_name), a, b, resamples),
        "threshold_free": None if threshold_free is None else {"figure": threshold_free, **differences[threshold_free]},
        "mcnemar": mcnemar,
        "differences": differences,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "groups": len(set(groups)),
            "paired": True,
        },
        # Records on which the two verdicts differ: ids, labels and scores only, no text of the data file
        "disagreements": [
            {
                "id": r.id,
                "label": r.label,
                "a_score": scores["a"][i],
                "b_score": scores["b"][i],
                "a_pass": passes["a"][i],
                "b_pass": passes["b"][i],
                "correct": "a" if correct["a"][i] else "b",
            }
            for i, r in enumerate(compared)
            if passes["a"][i] != passes["b"][i]
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    report_json, report_md = out / "report.json", out / "report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md.write_text(render_comparison(report), encoding="utf-8")
    return ComparisonRun(report=report, report_json=report_json, report_md=report_md)


def _side(run: RunFiles) -> dict[str, Any]:
    return {
        "directory": str(run.directory),
        "judge": run.judge,
        "config_hash": run.header.get("config_hash"),
        "source_commit": run.header.get("source_commit"),
        "threshold": run.threshold,
        "threshold_source": run.threshold_source,
        "seed": run.header.get("seed"),
    }


def _headline(name: str, entry: dict[str, Any] | None, a: RunFiles, b: RunFiles, resamples: int) -> dict[str, Any]:
    """The declared headline figure and the verdict, which is about that figure only."""
    base: dict[str, Any] = {"figure": name, "ahead": None}
    if entry is None or entry["difference"] is None:
        return {
            **base,
            **(entry or {}),
            "verdict": f"The headline figure ({name}) cannot be estimated for both judges on these records: no verdict.",
        }
    if entry["interval"] is None:
        return {
            **base,
            **entry,
            "verdict": f"No interval was computed ({resamples} draws): the difference is given without a verdict.",
        }
    if entry["contains_zero"]:
        verdict = (
            f"No difference is established: the {BOOTSTRAP_CONFIDENCE:.0%} interval of the difference of {name} "
            "contains 0. These data do not separate the two judges."
        )
        return {**base, **entry, "verdict": verdict}
    ahead = "b" if entry["difference"] > 0 else "a"
    run = b if ahead == "b" else a
    verdict = (
        f"{ahead.upper()} ({run.judge or run.directory.name}) is ahead on {name}: the {BOOTSTRAP_CONFIDENCE:.0%} "
        "interval of the difference excludes 0."
    )
    return {**base, **entry, "ahead": ahead, "verdict": verdict}


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _reading(entry: dict[str, Any]) -> str:
    if entry["difference"] is None:
        return "not estimable"
    if entry["interval"] is None:
        return "no interval"
    if entry["contains_zero"]:
        return "contains 0: not separated"
    better_b = (entry["difference"] > 0) == entry["higher_is_better"]
    return f"excludes 0: {'B' if better_b else 'A'} better"


def _row(name: str, entry: dict[str, Any]) -> str:
    interval = entry["interval"]
    shown = "n/a" if interval is None else f"{_fmt(interval[0], signed=True)} to {_fmt(interval[1], signed=True)}"
    return (
        f"| {name}{'' if entry['higher_is_better'] else ' (lower is better)'} | {_fmt(entry['a'])} | "
        f"{_fmt(entry['b'])} | {_fmt(entry['difference'], signed=True)} | {shown} | {_reading(entry)} |"
    )


def _mcnemar_lines(m: dict[str, Any], heading: str, boot: dict[str, Any]) -> list[str]:
    """The McNemar block: next to the headline on a full run, among the exploratory figures under class quotas."""
    lines = [
        "",
        f"{heading} Paired test on the binary verdict (McNemar, exact)",
        "",
        "| | B correct | B wrong |",
        "|---|---|---|",
        f"| A correct | {m['both_correct']} | {m['only_a_correct']} |",
        f"| A wrong | {m['only_b_correct']} | {m['both_wrong']} |",
        "",
        f"{m['discordant']} discordant pair(s): A alone correct on {m['only_a_correct']}, B alone on "
        f"{m['only_b_correct']}. Two-sided exact p = {m['p_value']:.4g}: "
        + (
            f"the two judges are not right equally often (p < {m['alpha']})."
            if m["significant"]
            else f"no difference in how often they are right is established (p >= {m['alpha']})."
        ),
        "",
        "The exact McNemar test assumes independent records and takes no account of the groups"
        + (f" ({boot['groups']} here)" if boot.get("groups") else "")
        + ": when records are grouped, for instance the sentences of one answer, its p-value is too optimistic. "
        "The paired bootstrap interval above, which resamples groups, is the one to rely on.",
    ]
    if m["follows_sample_class_mix"]:
        lines += [
            "",
            "Under class quotas this test is about the accuracy on the sample's class mix, not the set's: it can "
            "point the other way than the reweighted figures.",
        ]
    return lines


def render_comparison(report: dict[str, Any]) -> str:
    """Markdown view of a comparison."""
    a, b, data, boot = report["a"], report["b"], report["data"], report["bootstrap"]
    head = report["headline"]
    lines = [
        f"# Judge comparison: B ({b['judge'] or b['directory']}) against A ({a['judge'] or a['directory']})",
        "",
        f"**Headline figure: `{head['figure']}`.** {head['verdict']}",
        "",
        f"Criterion {data['criterion']}, {data['level']} level. Records: {data['compared']} compared of "
        f"{data['evaluated']} evaluated ({data['records']} in the data file)"
        + (f", class quotas of {data['per_class']} per label" if data["per_class"] is not None else "")
        + ".",
        f"- A: `{a['directory']}`, config `{str(a['config_hash'])[:12]}`, PASS if score >= {a['threshold']:.2f} "
        f"({a['threshold_source']})",
        f"- B: `{b['directory']}`, config `{str(b['config_hash'])[:12]}`, PASS if score >= {b['threshold']:.2f} "
        f"({b['threshold_source']})",
        "",
        "Differences are B minus A. Intervals: "
        + (
            f"{boot['confidence']:.0%}, {boot['resamples']} paired draws on {boot['groups']} groups (seed {boot['seed']}); "
            "in each draw both judges are evaluated on the same records and the difference is taken in the draw."
            if boot["resamples"]
            else "none (0 draws)."
        ),
    ]
    failures = report["failures"]
    lost = failures["a_only"] + failures["b_only"] + failures["both"]
    lines += [
        "",
        f"Failures: {lost} record(s) left out of the comparison"
        + (
            f" ({failures['a_only']} failed in A only, {failures['b_only']} in B only, {failures['both']} in both; "
            f"by kind, A: {failures['by_kind']['a'] or 'none'}, B: {failures['by_kind']['b'] or 'none'})."
            if lost
            else ": none, every evaluated record was scored by both judges."
        ),
    ]
    for note in report["notes"]:
        lines += ["", f"Note: {note}"]

    header = ["| Figure | A | B | B - A | Interval of the difference | Reading |", "|---|---|---|---|---|---|"]
    differences = report["differences"]
    if head["figure"] in differences:
        lines += ["", "## Headline", "", *header, _row(head["figure"], differences[head["figure"]])]
    free = report["threshold_free"]
    if free is not None:
        lines += [
            "",
            "## Independent of the thresholds",
            "",
            "Both judges give probabilities: the AUROC compares how they rank the records, whatever the thresholds. "
            "It stands next to the headline figure and does not replace it.",
            "",
            *header,
            _row(free["figure"], free),
        ]
    m = report["mcnemar"]
    mcnemar_under_quotas = m["role"] == "exploratory"
    if not mcnemar_under_quotas:
        lines += _mcnemar_lines(m, "##", boot)
    exploratory = [(name, entry) for name, entry in differences.items() if entry["role"] == "exploratory"]
    if exploratory or mcnemar_under_quotas:
        lines += [
            "",
            "## Exploratory figures",
            "",
            "Not corrected for multiple comparisons: with this many intervals, some exclude 0 by chance. "
            "They describe where the judges differ; they do not carry the verdict.",
        ]
        if exploratory:
            lines += ["", *header, *(_row(name, entry) for name, entry in exploratory)]
        if mcnemar_under_quotas:
            lines += _mcnemar_lines(m, "###", boot)
    dropped = {name: entry["excluded_draws"] for name, entry in differences.items() if entry["excluded_draws"]}
    if dropped:
        lines += [
            "",
            "Draws left out of an interval (the figure could not be computed for one of the judges in the draw): "
            + ", ".join(f"{name} {count}" for name, count in dropped.items())
            + f" of {boot['resamples']}.",
        ]
    disagreements = report["disagreements"]
    lines += [
        "",
        "## Disagreements",
        "",
        f"{len(disagreements)} record(s) on which the two verdicts differ ({NEGATIVE_CLASS}: score below the judge's "
        "threshold). The full list, with ids, labels and scores, is in `report.json`.",
    ]
    by_label: dict[str, Counter[str]] = {}
    for d in disagreements:
        by_label.setdefault(d["label"], Counter())[d["correct"]] += 1
    if by_label:
        lines += ["", "| Human label | A alone correct | B alone correct |", "|---|---|---|"]
        lines += [f"| {label} | {c['a']} | {c['b']} |" for label, c in sorted(by_label.items())]
    return "\n".join(lines) + "\n"


__all__ = [
    "ComparisonError",
    "ComparisonRun",
    "RunFiles",
    "auroc",
    "auroc_reweighted",
    "compare_runs",
    "figures",
    "load_run",
    "mcnemar_exact",
    "render_comparison",
]
