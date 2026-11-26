"""Benchmark harness: compare judges on a labeled JSONL set, at sentence or answer level.

No data ships with the package. The input is the JSONL written by the
judge-training conversion (sentence level: ``id``, ``question``, ``context``,
``prev_sentence``, ``sentence``, ``label`` in SUPPORTED / CONTRADICTED /
UNSUPPORTED, plus free metadata such as ``task``, ``generator``,
``source_id``) or an answer-level file (``answer``, ``label`` in PASS / FAIL).

On a sentence-level set the judge is called directly with the sentence as
the answer: the segmentation was done by the conversion, ``SentenceLevel``
is not involved. ``prev_sentence`` is never transmitted to the judge; the
report says how many records carry one.

Every call goes through ``judge_many``. Attempts are appended to
``attempts.jsonl`` as they complete, so an interrupted run resumes from that
file by ``id``. The first line of the file is a header (config hash,
criterion, level, sample, seed, stratify, SHA-256 of the data file); a run
whose parameters differ from the header is refused, never mixed. For a
probabilistic judge the pass threshold is chosen on
the ``--dev`` set and applied unchanged to the evaluated set; without a dev
set the judge's own verdict policy applies. Nothing is tuned on the
evaluated set.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.calibration.calibrator import MIN_MINORITY_FOR_KAPPA, _cohens_kappa
from ragnarok_ai.judges.contract import (
    BinaryProbability,
    ClassProbabilities,
    JudgeAttempt,
    _jsonable,
)
from ragnarok_ai.judges.runner import judge_many

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence

    from ragnarok_ai.core.protocols import JudgeProtocol

SENTENCE_LABELS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")
ANSWER_LABELS = ("PASS", "FAIL")
# The labels read as a human PASS; every other label is the not-supported class
PASS_LABELS = ("SUPPORTED", "PASS")
# The class of interest: a sentence the context does not support
NEGATIVE_CLASS = "not supported"
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
# The level of every bootstrap interval of a report, written as `confidence` in its bootstrap blocks so a
# reader of the JSON does not have to assume it. A constant, not a parameter: no function and no CLI option
# takes it. The percentile bounds (2.5% and 97.5%) are literals in `_percentile_interval`; deriving them
# from this number would change them by a rounding error, and with them intervals already published
BOOTSTRAP_CONFIDENCE = 0.95
# The figures of `prevalence_adjusted` that get an interval, in the order of the report
ADJUSTED_FIGURES = (
    "recall_not_supported",
    "specificity",
    "precision_not_supported_reweighted",
    "accuracy_reweighted",
    "f1_not_supported_reweighted",
)
ECE_BINS = 10
ATTEMPTS_FORMAT = 1
# Fields of the attempts header; a run must match all of them to resume. Checked in this
# order so that the message names the plain field (criterion differs) before the hash it changes.
ATTEMPTS_HEADER_FIELDS = (
    "criterion",
    "level",
    "sample",
    "per_class",
    "seed",
    "stratify",
    "data_sha256",
    "config_hash",
)
# Checked only when both sides know it: a wheel (None) resumes a checkout's run and vice versa
ATTEMPTS_PROVENANCE_FIELD = "source_commit"


class AttemptsMismatchError(ValueError):
    """An existing attempts file was produced by a run with other parameters."""


# =============================================================================
# Records
# =============================================================================


@dataclass(frozen=True)
class BenchRecord:
    """One labeled unit of the benchmark set.

    Attributes:
        id: Unique identifier, used to resume a run.
        group_id: Grouping key for the bootstrap (``group_id``, else
            ``source_id``, else ``response_id``, else ``id``).
        question: The question, possibly empty.
        context: The retrieved context.
        answer: What the judge receives as the answer: the sentence on a
            sentence-level set, the answer otherwise.
        label: The human label as written in the file.
        human_pass: Binary reading of the label: SUPPORTED or PASS.
        level: ``sentence`` or ``answer``.
        has_prev_sentence: Whether the record carried a ``prev_sentence``
            (kept in the file, never transmitted to the judge).
        metadata: Every other field of the record, for breakdowns.
    """

    id: str
    group_id: str
    question: str
    context: str
    answer: str
    label: str
    human_pass: bool
    level: str
    has_prev_sentence: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        """``judge_many`` identifier."""
        return self.id


_CORE_KEYS = {
    "id",
    "group_id",
    "source_id",
    "response_id",
    "question",
    "context",
    "prev_sentence",
    "sentence",
    "answer",
    "label",
}


def parse_record(raw: dict[str, Any], line_number: int) -> BenchRecord:
    """Validate and normalize one JSONL object."""
    if "id" not in raw or raw["id"] in (None, ""):
        msg = f"line {line_number}: missing 'id'"
        raise ValueError(msg)
    allowed: tuple[str, ...]
    if "sentence" in raw:
        level, answer, allowed = "sentence", raw["sentence"], SENTENCE_LABELS
    elif "answer" in raw:
        level, answer, allowed = "answer", raw["answer"], ANSWER_LABELS
    else:
        msg = f"line {line_number}: record needs 'sentence' (sentence level) or 'answer' (answer level)"
        raise ValueError(msg)
    label = str(raw.get("label", "")).upper()
    if label not in allowed:
        msg = f"line {line_number}: label {raw.get('label')!r} not in {allowed}"
        raise ValueError(msg)
    if not isinstance(answer, str) or not answer:
        msg = f"line {line_number}: empty text to judge"
        raise ValueError(msg)
    group = next(
        (str(raw[k]) for k in ("group_id", "source_id", "response_id") if raw.get(k) not in (None, "")), str(raw["id"])
    )
    return BenchRecord(
        id=str(raw["id"]),
        group_id=group,
        question=str(raw.get("question") or ""),
        context=str(raw.get("context") or ""),
        answer=answer,
        label=label,
        human_pass=label in PASS_LABELS,
        level=level,
        has_prev_sentence=raw.get("prev_sentence") not in (None, ""),
        metadata={k: v for k, v in raw.items() if k not in _CORE_KEYS},
    )


def load_records(path: str | Path) -> list[BenchRecord]:
    """Read a JSONL set; every record must be at the same level, ids must be unique."""
    records: list[BenchRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records.append(parse_record(json.loads(line), number))
    if not records:
        msg = f"{path}: no record"
        raise ValueError(msg)
    levels = {r.level for r in records}
    if len(levels) > 1:
        msg = f"{path}: mixed levels {sorted(levels)}; a set is sentence-level or answer-level, not both"
        raise ValueError(msg)
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})[:5]
        msg = f"{path}: duplicate ids {duplicates}"
        raise ValueError(msg)
    return records


def stratified_sample(records: Sequence[BenchRecord], size: int, stratify: str | None, seed: int) -> list[BenchRecord]:
    """A deterministic sample of *size* records, proportional per stratum when *stratify* is given.

    *stratify* is ``label`` or a metadata field. Rounding leftovers go to the
    largest strata. The result keeps the input order.
    """
    if size >= len(records):
        return list(records)
    rng = random.Random(seed)
    if stratify is None:
        chosen = set(rng.sample(range(len(records)), size))
        return [r for i, r in enumerate(records) if i in chosen]
    strata: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        key = r.label if stratify == "label" else str(r.metadata.get(stratify))
        strata.setdefault(key, []).append(i)
    quotas = {k: size * len(v) / len(records) for k, v in strata.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    leftover = size - sum(counts.values())
    for k in sorted(strata, key=lambda k: (-(quotas[k] - counts[k]), -len(strata[k]), k)):
        if leftover <= 0:
            break
        if counts[k] < len(strata[k]):
            counts[k] += 1
            leftover -= 1
    picked: set[int] = set()
    for k, indices in sorted(strata.items()):
        picked.update(rng.sample(indices, min(counts[k], len(indices))))
    return [r for i, r in enumerate(records) if i in picked]


def quota_sample(records: Sequence[BenchRecord], per_class: int, seed: int) -> list[BenchRecord]:
    """At most *per_class* records of every label, drawn with *seed*; a smaller class is taken whole.

    The result keeps the input order. Metrics on such a sample depend on the
    quota, not on the set's class mix, calibration (Brier, ECE) included:
    the report flags them and adds prevalence-adjusted figures
    (``prevalence_adjusted``).
    """
    if per_class < 1:
        msg = f"per_class must be at least 1, got {per_class}"
        raise ValueError(msg)
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_label.setdefault(r.label, []).append(i)
    picked: set[int] = set()
    for _label, indices in sorted(by_label.items()):
        picked.update(rng.sample(indices, min(per_class, len(indices))))
    return [r for i, r in enumerate(records) if i in picked]


def _label_agreement(judge_pass: Sequence[bool], labels: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Per human label: (records the judge reads as the humans do, records).

    A supported label (``PASS_LABELS``) is read as the humans do when the
    judge passes it, a not-supported one when the judge rejects it. The
    rate of a label is the first number over the second: the specificity
    for a supported label, the recall for a not-supported one. This is the
    single computation behind ``prevalence_adjusted`` and
    ``recall_by_label``.
    """
    counts: dict[str, tuple[int, int]] = {}
    for label, passed in zip(labels, judge_pass, strict=True):
        agreeing, n = counts.get(label, (0, 0))
        counts[label] = (agreeing + (1 if passed == (label in PASS_LABELS) else 0), n + 1)
    return counts


def prevalence_adjusted(
    judge_pass: Sequence[bool],
    human_pass: Sequence[bool],
    full_not_supported_prevalence: float,
    *,
    labels: Sequence[str] | None = None,
    full_label_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Figures that do not depend on the sample's class mix, given the full set's composition.

    Precision and accuracy are reweighted by the prevalence of the
    not-supported class in the full set (``pi``), from the recall ``r`` of
    that class and the specificity ``s`` (supported records recognized):
    precision = ``pi r / (pi r + (1 - pi)(1 - s))``, accuracy =
    ``pi r + (1 - pi) s``.

    ``r`` and ``s`` must themselves be free of the quota. A class quota is
    drawn per human label, so a rate is quota-free only within one label.
    With *labels* (the human label of each scored record) and
    *full_label_counts* (records per label in the full set), each label's
    rate is computed on its own and the rates are combined with the full
    set's label shares (``strata`` is ``"label"``): ``r`` is the sum over
    the not-supported labels of ``N_label / N_not_supported * r_label``.
    On a sentence-level set the not-supported class holds two labels,
    CONTRADICTED and UNSUPPORTED, whose shares the quota changes; pooling
    them in the sample, as versions up to 1.13.1 did, gives a recall that
    follows the quota's mix.

    Without them the two classes are the strata (``strata`` is
    ``"class"``): exact when the quota is drawn on those two classes, as on
    an answer-level set (PASS / FAIL).

    A label present in the full set with no scored record in the sample is
    listed in ``missing_labels``; the rate of its class, and the figures
    derived from it, are then ``None`` rather than renormalized over the
    labels that are there.

    ``f1_not_supported_reweighted`` is the harmonic mean of the reweighted
    precision and the recall: the F1 of the not-supported class at the full
    set's composition, the figure to compare judges on a quota run.
    ``None`` when either is, or when both are 0. The intervals of these
    figures are computed by ``prevalence_adjusted_intervals``.
    """
    pi = full_not_supported_prevalence
    by_label: dict[str, dict[str, Any]] | None = None
    missing: list[str] = []
    recall: float | None
    specificity: float | None
    if labels is None or full_label_counts is None:
        negatives = [j for j, h in zip(judge_pass, human_pass, strict=True) if not h]
        positives = [j for j, h in zip(judge_pass, human_pass, strict=True) if h]
        recall = (sum(1 for j in negatives if not j) / len(negatives)) if negatives else None
        specificity = (sum(1 for j in positives if j) / len(positives)) if positives else None
    else:
        agreement = _label_agreement(judge_pass, labels)
        total = sum(full_label_counts.values())
        by_label = {}
        # Per class (True: supported): the full-set count and the weighted sum of the rates of its labels
        full_class = {True: 0, False: 0}
        weighted = {True: 0.0, False: 0.0}
        incomplete = {True: False, False: False}
        for label in sorted(full_label_counts):
            count = full_label_counts[label]
            if count <= 0:
                continue
            supported = label in PASS_LABELS
            # The share of the label the judge reads as the humans do: passed if supported, rejected if not
            agreeing, n = agreement.get(label, (0, 0))
            rate = (agreeing / n) if n else None
            by_label[label] = {
                "human_class": "supported" if supported else NEGATIVE_CLASS,
                "full_set_count": count,
                "full_set_share": count / total,
                "n": n,
                "rate": rate,
            }
            full_class[supported] += count
            if rate is None:
                missing.append(label)
                incomplete[supported] = True
            else:
                weighted[supported] += count * rate
        recall = (weighted[False] / full_class[False]) if full_class[False] and not incomplete[False] else None
        specificity = (weighted[True] / full_class[True]) if full_class[True] and not incomplete[True] else None
    precision: float | None = None
    accuracy: float | None = None
    f1: float | None = None
    if recall is not None and specificity is not None:
        denominator = pi * recall + (1 - pi) * (1 - specificity)
        precision = (pi * recall / denominator) if denominator > 0 else None
        accuracy = pi * recall + (1 - pi) * specificity
        if precision is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
    return {
        "full_set_not_supported_prevalence": pi,
        "strata": "class" if by_label is None else "label",
        "labels": by_label,
        "missing_labels": missing,
        "recall_not_supported": recall,
        "recall_supported": specificity,
        "specificity": specificity,
        "precision_not_supported_reweighted": precision,
        "accuracy_reweighted": accuracy,
        "f1_not_supported_reweighted": f1,
    }


def reweighted_brier(
    probabilities: Sequence[float],
    human_pass: Sequence[bool],
    full_not_supported_prevalence: float,
    *,
    labels: Sequence[str] | None = None,
    full_label_counts: Mapping[str, int] | None = None,
) -> float | None:
    """Brier score at the full set's composition, from a sample whose mix was set by quotas.

    Brier is a mean of per-record squared errors, so it splits exactly by
    stratum: the mean squared error of each stratum, weighted by the
    stratum's share in the full set. The strata are those of
    ``prevalence_adjusted``. With *labels* and *full_label_counts* they are
    the human labels, the unit the quota is drawn on: on a sentence-level
    set the not-supported class holds CONTRADICTED and UNSUPPORTED, and
    pooling them in the sample would weigh their errors by the quota's mix.
    Without them they are the two classes, weighted ``pi`` and ``1 - pi``
    (*full_not_supported_prevalence*): exact when the quota is drawn on
    those two classes, as on an answer-level set.

    ``None`` when a stratum of the full set has no record in the sample.
    ECE has no such split (bin membership would have to be reweighted
    record by record) and is not reweighted.
    """
    errors = [(p - (1.0 if h else 0.0)) ** 2 for p, h in zip(probabilities, human_pass, strict=True)]
    strata: dict[Any, list[float]] = {}
    weights: dict[Any, float]
    if labels is None or full_label_counts is None:
        for error, h in zip(errors, human_pass, strict=True):
            strata.setdefault(h, []).append(error)
        weights = {False: full_not_supported_prevalence, True: 1 - full_not_supported_prevalence}
    else:
        for error, label in zip(errors, labels, strict=True):
            strata.setdefault(label, []).append(error)
        total = sum(full_label_counts.values())
        weights = {label: count / total for label, count in full_label_counts.items() if count > 0}
    if not weights or any(not strata.get(stratum) for stratum in weights):
        return None
    return sum(weight * statistics.fmean(strata[stratum]) for stratum, weight in weights.items())


# =============================================================================
# Attempts on disk
# =============================================================================


def attempt_to_dict(attempt: JudgeAttempt) -> dict[str, Any]:
    """JSON view of an attempt: identifiers, the five result fields, native output, trace, failure."""
    result = attempt.result
    return {
        "sample_id": attempt.sample_id,
        "criterion": attempt.criterion,
        "ok": attempt.ok,
        "result": None
        if result is None
        else {
            "verdict": result.verdict,
            "score": result.score,
            "raw_score": result.raw_score,
            "explanation": result.explanation,
            "output": _jsonable(result.output) if result.output is not None else None,
        },
        "error": None if attempt.error is None else _jsonable(attempt.error),
        "trace": None if attempt.trace is None else _jsonable(attempt.trace),
        "config_hash": attempt.config.config_hash,
    }


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records: Sequence[BenchRecord]) -> str:
    """Hex SHA-256 of the records' content, as canonical JSON (sorted keys, one record per line).

    Used for the attempts header when the caller gives no file digest, so a
    run through the Python API is protected like the CLI: the same records
    give the same digest whatever the file they came from.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def attempts_header(
    config_hash: str,
    criterion: str,
    level: str,
    *,
    sample: int | None,
    seed: int | None,
    stratify: str | None,
    data_sha256: str | None,
    per_class: int | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """The first line of an attempts file: what the run must match to resume from it.

    ``source_commit`` (the checkout's commit, ``None`` for a wheel) is
    compared only when both runs know it.
    """
    return {
        "header": ATTEMPTS_FORMAT,
        "config_hash": config_hash,
        "criterion": criterion,
        "level": level,
        "sample": sample,
        "per_class": per_class,
        "seed": seed,
        "stratify": stratify,
        "data_sha256": data_sha256,
        ATTEMPTS_PROVENANCE_FIELD: source_commit,
    }


def _json_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Non-empty lines of a JSONL file as (line number, object); a truncated last line is ignored."""
    rows: list[tuple[int, dict[str, Any]]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((number, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows


def _is_header(row: dict[str, Any]) -> bool:
    return "header" in row and "sample_id" not in row


def read_attempts(path: Path, expected: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Attempts already written, by ``sample_id``.

    With *expected* (an ``attempts_header``), the file's header must carry the
    same value for every field of ``ATTEMPTS_HEADER_FIELDS`` and every row
    must carry the expected ``config_hash``; otherwise ``AttemptsMismatchError``
    names the field, and nothing is resumed.
    """
    done: dict[str, dict[str, Any]] = {}
    rows = _json_lines(path)
    if not rows:
        return done
    first = rows[0][1]
    if expected is not None:
        if not _is_header(first):
            msg = f"{path.name}: no header line, the file comes from another run format. Use another --out directory"
            raise AttemptsMismatchError(msg)
        for field_name in ATTEMPTS_HEADER_FIELDS:
            if first.get(field_name) != expected[field_name]:
                msg = (
                    f"{path.name} was produced with another {field_name} "
                    f"({first.get(field_name)!r}, this run: {expected[field_name]!r}). Use another --out directory"
                )
                raise AttemptsMismatchError(msg)
        recorded, current = first.get(ATTEMPTS_PROVENANCE_FIELD), expected.get(ATTEMPTS_PROVENANCE_FIELD)
        if recorded and current and recorded != current:
            msg = (
                f"{path.name} was produced from another source commit ({recorded[:12]}, this run: {current[:12]}). "
                "Use another --out directory"
            )
            raise AttemptsMismatchError(msg)
    for number, row in rows:
        if _is_header(row):
            continue
        if expected is not None and row.get("config_hash") != expected["config_hash"]:
            msg = (
                f"{path.name} line {number}: config_hash {row.get('config_hash')!r} differs from this run's "
                f"{expected['config_hash']!r}. Use another --out directory"
            )
            raise AttemptsMismatchError(msg)
        done[str(row["sample_id"])] = row
    return done


# =============================================================================
# Thresholds and metrics
# =============================================================================


def _f1_not_supported(judge_pass: Sequence[bool], human_pass: Sequence[bool]) -> tuple[float, float, float]:
    """Precision, recall and F1 of the negative class (judge FAIL vs human not supported)."""
    tp = sum(1 for j, h in zip(judge_pass, human_pass, strict=True) if not j and not h)
    fp = sum(1 for j, h in zip(judge_pass, human_pass, strict=True) if not j and h)
    fn = sum(1 for j, h in zip(judge_pass, human_pass, strict=True) if j and not h)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _balanced_accuracy(judge_pass: Sequence[bool], human_pass: Sequence[bool]) -> float:
    pos = [j for j, h in zip(judge_pass, human_pass, strict=True) if h]
    neg = [j for j, h in zip(judge_pass, human_pass, strict=True) if not h]
    tpr = sum(pos) / len(pos) if pos else 0.0
    tnr = sum(1 for j in neg if not j) / len(neg) if neg else 0.0
    return (tpr + tnr) / 2


def choose_threshold(scores: Sequence[float], human_pass: Sequence[bool], fallback: float) -> float:
    """Threshold maximizing the F1 of the negative class on *dev*; ties go to the one closest to *fallback*."""
    candidates = sorted({round(t / 20, 2) for t in range(1, 20)} | {fallback})
    best = (fallback, _f1_not_supported([s >= fallback for s in scores], human_pass)[2])
    for threshold in candidates:
        f1 = _f1_not_supported([s >= threshold for s in scores], human_pass)[2]
        if f1 > best[1] + 1e-9 or (abs(f1 - best[1]) <= 1e-9 and abs(threshold - fallback) < abs(best[0] - fallback)):
            best = (threshold, f1)
    return best[0]


def binary_metrics(judge_pass: Sequence[bool], human_pass: Sequence[bool]) -> dict[str, float | None]:
    """Agreement metrics of one set or sub-group.

    ``kappa`` is ``None`` when undefined (judge and humans used a single class).
    ``minority_count`` is the size of the smaller human class and
    ``kappa_interpretable`` (0.0 or 1.0) says whether it reaches the
    calibrator's threshold of 10 samples; below it, kappa is driven by a
    handful of labels and should not be read as a measurement.
    """
    precision, recall, f1 = _f1_not_supported(judge_pass, human_pass)
    n = len(human_pass)
    majority = max(sum(human_pass), n - sum(human_pass)) / n if n else 0.0
    minority = min(sum(human_pass), n - sum(human_pass))
    return {
        "n": float(n),
        "minority_count": float(minority),
        "kappa_interpretable": float(minority >= MIN_MINORITY_FOR_KAPPA),
        "accuracy": sum(1 for j, h in zip(judge_pass, human_pass, strict=True) if j == h) / n if n else 0.0,
        "balanced_accuracy": _balanced_accuracy(judge_pass, human_pass) if n else 0.0,
        "kappa": _cohens_kappa(judge_pass, human_pass) if n else None,
        "precision_not_supported": precision,
        "recall_not_supported": recall,
        "f1_not_supported": f1,
        "majority_baseline_accuracy": majority,
        "human_pass_rate": sum(human_pass) / n if n else 0.0,
    }


def three_class_metrics(predicted: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Per-class F1 and confusion matrix when the judge predicts the set's own labels."""
    classes = sorted(set(labels) | set(predicted))
    confusion = {truth: dict.fromkeys(classes, 0) for truth in classes}
    for p, t in zip(predicted, labels, strict=True):
        confusion[t][p] += 1
    per_class: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in classes if t != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(confusion[c].values())),
        }
    macro = statistics.fmean(v["f1"] for v in per_class.values()) if per_class else 0.0
    return {"per_class": per_class, "macro_f1": macro, "confusion": confusion}


def calibration_metrics(probabilities: Sequence[float], human_pass: Sequence[bool]) -> dict[str, Any]:
    """Brier score and 10-bin expected calibration error of P(pass)."""
    n = len(probabilities)
    if n == 0:
        return {"brier": None, "ece": None, "bins": []}
    brier = sum((p - (1.0 if h else 0.0)) ** 2 for p, h in zip(probabilities, human_pass, strict=True)) / n
    bins: list[dict[str, float]] = []
    ece = 0.0
    for b in range(ECE_BINS):
        lo, hi = b / ECE_BINS, (b + 1) / ECE_BINS
        members = [
            (p, h)
            for p, h in zip(probabilities, human_pass, strict=True)
            if (lo <= p < hi) or (b == ECE_BINS - 1 and p == 1.0)
        ]
        if not members:
            continue
        confidence = statistics.fmean(p for p, _ in members)
        accuracy = sum(1 for _, h in members if h) / len(members)
        ece += len(members) / n * abs(accuracy - confidence)
        bins.append(
            {"lower": lo, "upper": hi, "count": float(len(members)), "confidence": confidence, "observed": accuracy}
        )
    return {"brier": brier, "ece": ece, "bins": bins}


def _bootstrap_metadata(groups: Sequence[str], seed: int, resamples: int) -> dict[str, Any]:
    """Report actual draws and explain when the scored groups cannot be resampled."""
    count = len(set(groups))
    block = {"groups": count, "seed": seed, "confidence": BOOTSTRAP_CONFIDENCE, "resamples": resamples}
    if count < 2:
        reason = "a single group" if count else "no scored groups"
        return {**block, "resamples": 0, "reliable": False, "reason": f"{reason}: no interval can be estimated"}
    return block


def _group_draws(groups: Sequence[str], seed: int, resamples: int) -> Iterator[list[int]]:
    """The record indices of each bootstrap draw: as many groups as the sample holds, drawn with replacement.

    Every interval of a report is computed on this one sequence of draws
    (same groups, same seed), so the figures of a draw belong together.
    """
    units: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        units.setdefault(g, []).append(i)
    unit_list = list(units.values())
    if len(unit_list) < 2:
        return
    rng = random.Random(seed)
    for _ in range(resamples):
        yield [i for _ in range(len(unit_list)) for i in unit_list[rng.randrange(len(unit_list))]]


def _percentile_interval(values: list[float]) -> list[float] | None:
    """The 2.5th and 97.5th percentiles of *values* (``BOOTSTRAP_CONFIDENCE``, 95%); ``None`` without any."""
    if not values:
        return None
    values = sorted(values)
    return [
        values[max(0, math.floor(0.025 * (len(values) - 1)))],
        values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))],
    ]


def grouped_bootstrap(
    judge_pass: Sequence[bool],
    human_pass: Sequence[bool],
    groups: Sequence[str],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """95% percentile intervals of F1 (negative class), balanced accuracy and kappa, resampling by group.

    Draws with a single human class are excluded and counted; the intervals
    are flagged unreliable above 10% exclusions. A draw whose kappa is
    undefined (cannot happen after that exclusion) would be counted too.
    """
    samples: dict[str, list[float]] = {"f1_not_supported": [], "balanced_accuracy": [], "kappa": []}
    excluded = 0
    for drawn in _group_draws(groups, seed, resamples):
        h = [human_pass[i] for i in drawn]
        if all(h) or not any(h):
            excluded += 1
            continue
        j = [judge_pass[i] for i in drawn]
        samples["f1_not_supported"].append(_f1_not_supported(j, h)[2])
        samples["balanced_accuracy"].append(_balanced_accuracy(j, h))
        kappa = _cohens_kappa(j, h)
        if kappa is None:
            excluded += 1
            continue
        samples["kappa"].append(kappa)

    return {
        "excluded_single_class_draws": excluded,
        "reliable": excluded <= 0.10 * resamples,
        **_bootstrap_metadata(groups, seed, resamples),
        "intervals": {k: _percentile_interval(v) for k, v in samples.items()},
    }


def recall_by_label(
    judge_pass: Sequence[bool],
    labels: Sequence[str],
    groups: Sequence[str],
    threshold: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Recall of each not-supported human label of a sentence-level set, with the specificity.

    ``binary`` pools CONTRADICTED and UNSUPPORTED into one "not supported"
    recall; a judge can detect one and miss the other. For each of the two
    labels: the scored records (``n``), those the judge counts as not
    supported (``detected``: score below *threshold*, the threshold of the
    report), their share (``recall``) and its 95% ``interval``. The
    specificity, the share of the SUPPORTED records the judge passes, is
    given next to them so the three rates are read together.

    A rate within one label does not depend on the class mix, so the block
    is not sample-dependent under class quotas; a quota only changes ``n``
    and the width of the intervals. The rates are those of
    ``prevalence_adjusted`` (``_label_agreement``).

    A label without any scored record keeps its key, with ``n`` 0 and
    ``None`` for the rate and the interval. The intervals come from the
    draws of ``grouped_bootstrap`` (same groups, seed and resamples); a
    draw without any record of a label is excluded for that label and
    counted (``excluded_draws``), and the block is flagged unreliable when
    a label loses more than 10% of the draws.
    """
    positive = next(label for label in SENTENCE_LABELS if label in PASS_LABELS)
    negatives = [label for label in SENTENCE_LABELS if label not in PASS_LABELS]
    agreement = _label_agreement(judge_pass, labels)
    rates: dict[str, list[float]] = {label: [] for label in SENTENCE_LABELS}
    excluded = dict.fromkeys(SENTENCE_LABELS, 0)
    for drawn in _group_draws(groups, seed, resamples):
        draw = _label_agreement([judge_pass[i] for i in drawn], [labels[i] for i in drawn])
        for label in SENTENCE_LABELS:
            if agreement.get(label) is None:
                continue  # no record at all: no interval, and nothing to count
            if label in draw:
                rates[label].append(draw[label][0] / draw[label][1])
            else:
                excluded[label] += 1

    def entry(label: str) -> tuple[int, int, float | None, list[float] | None]:
        agreeing, n = agreement.get(label, (0, 0))
        return n, agreeing, (agreeing / n) if n else None, _percentile_interval(rates[label])

    by_label: dict[str, dict[str, Any]] = {}
    for label in negatives:
        n, detected, recall, interval = entry(label)
        by_label[label] = {
            "n": n,
            "detected": detected,
            "recall": recall,
            "interval": interval,
            "excluded_draws": excluded[label],
        }
    n, recognized, specificity, interval = entry(positive)
    return {
        "threshold": threshold,
        # Rates within one label: a class quota changes n and the intervals, not what is estimated
        "sample_dependent": False,
        "labels": by_label,
        "specificity": {
            "label": positive,
            "n": n,
            "recognized": recognized,
            "value": specificity,
            "interval": interval,
            "excluded_draws": excluded[positive],
        },
        "bootstrap": {
            "reliable": all(count <= 0.10 * resamples for count in excluded.values()),
            **_bootstrap_metadata(groups, seed, resamples),
        },
    }


def prevalence_adjusted_intervals(
    judge_pass: Sequence[bool],
    human_pass: Sequence[bool],
    full_not_supported_prevalence: float,
    groups: Sequence[str],
    seed: int,
    resamples: int,
    *,
    labels: Sequence[str] | None = None,
    full_label_counts: Mapping[str, int] | None = None,
    probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    """95% percentile intervals of the ``prevalence_adjusted`` figures, resampling by group.

    The draws are those of ``grouped_bootstrap`` (``_group_draws``: same
    groups, seed and resamples). **The reweighting is done inside each
    draw**: ``prevalence_adjusted`` is called on the records of the draw,
    so its per-label rates are recomputed and combined with the full set's
    label counts, which are fixed (the full set is known, it is not
    resampled). Reweighting the point estimate and attaching an interval
    computed on pooled figures would bring the quota's mix back into the
    interval. With *probabilities*, ``brier_reweighted`` gets its interval
    the same way (``reweighted_brier`` on each draw).

    A draw in which a figure cannot be estimated (a label of the full set
    absent from the draw) is excluded for that figure and counted
    (``excluded_draws``); the block is flagged unreliable when a figure
    loses more than 10% of the draws. A figure whose point estimate is
    ``None`` has no interval and nothing is counted for it.

    Two limits, shared with the other intervals of the report. The quota is
    drawn per label without regard to groups, while the bootstrap resamples
    groups. And a label subsampled by the quota is drawn without
    replacement from a finite set (300 of 469, say): the bootstrap ignores
    the finite-population correction, so its interval is conservative for
    that label.

    Returns:
        ``intervals`` (one ``[low, high]`` or ``None`` per figure) and
        ``bootstrap`` (resamples, seed, confidence, groups, excluded draws
        per figure, reliability), to be merged into the block.
    """
    pi = full_not_supported_prevalence
    figures = [*ADJUSTED_FIGURES, *(["brier_reweighted"] if probabilities is not None else [])]

    def estimate(indices: Sequence[int] | None) -> dict[str, float | None]:
        def take(values: Sequence[Any] | None) -> Any:
            return values if values is None or indices is None else [values[i] for i in indices]

        jp, hp, lab, prob = take(judge_pass), take(human_pass), take(labels), take(probabilities)
        block = prevalence_adjusted(jp, hp, pi, labels=lab, full_label_counts=full_label_counts)
        out: dict[str, float | None] = {name: block[name] for name in ADJUSTED_FIGURES}
        if prob is not None:
            out["brier_reweighted"] = reweighted_brier(prob, hp, pi, labels=lab, full_label_counts=full_label_counts)
        return out

    point = estimate(None)
    values: dict[str, list[float]] = {name: [] for name in figures}
    excluded = dict.fromkeys(figures, 0)
    for drawn in _group_draws(groups, seed, resamples):
        draw = estimate(drawn)
        for name in figures:
            if point[name] is None:
                continue  # no point estimate: no interval, and nothing to count
            value = draw[name]
            if value is None:
                excluded[name] += 1
            else:
                values[name].append(value)
    return {
        "intervals": {name: _percentile_interval(values[name]) for name in figures},
        "bootstrap": {
            "excluded_draws": excluded,
            "reliable": all(count <= 0.10 * resamples for count in excluded.values()),
            **_bootstrap_metadata(groups, seed, resamples),
        },
    }


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB (ru_maxrss is bytes on macOS, KiB elsewhere)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / (1024 * 1024) if sys.platform == "darwin" else maxrss / 1024


async def ollama_model_memory(
    base_url: str, model: str, *, timeout: float = 5.0, transport: Any = None
) -> dict[str, Any]:
    """Resident size of *model* as Ollama reports it on ``/api/ps`` (``size`` and ``size_vram``, in MiB).

    The harness RSS never includes the model, which runs in the Ollama
    server. Called right after the run, while the model is still loaded.
    When the server does not answer or the model is not listed, the result
    says so in ``reason`` and ``available`` is false.
    """
    import httpx

    result: dict[str, Any] = {
        "source": "ollama /api/ps",
        "model": model,
        "available": False,
        "size_mb": None,
        "size_vram_mb": None,
        "reason": None,
    }
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport) as client:
            response = await client.get("/api/ps")
            response.raise_for_status()
            entries = response.json().get("models", [])
    except Exception as exc:  # the report must be written even when Ollama is gone
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    wanted = {model, f"{model}:latest"}
    names: list[str] = []
    for entry in entries:
        name = str(entry.get("name") or entry.get("model") or "")
        names.append(name)
        if name in wanted or str(entry.get("model")) in wanted:
            result.update(
                model=name,
                available=True,
                size_mb=round(float(entry.get("size", 0)) / (1024 * 1024), 1),
                size_vram_mb=round(float(entry.get("size_vram", 0)) / (1024 * 1024), 1),
            )
            return result
    result["reason"] = f"model not loaded (listed: {names or 'none'})"
    return result


# =============================================================================
# Run
# =============================================================================


@dataclass
class BenchmarkRun:
    """Everything a benchmark run produced."""

    report: dict[str, Any]
    attempts_path: Path
    report_json: Path
    report_md: Path


def _score_of(row: dict[str, Any]) -> float | None:
    result = row.get("result")
    return None if result is None else float(result["score"])


def _probability_of(row: dict[str, Any]) -> float | None:
    """P(pass) when the judge produced a probability (binary or over classes)."""
    result = row.get("result")
    if result is None or result.get("output") is None:
        return None
    output = result["output"]
    if output.get("type") == BinaryProbability.__name__:
        return float(output["p_positive"])
    if output.get("type") == ClassProbabilities.__name__:
        return float(output["probs"][output["labels"].index(output["positive_label"])])
    return None


def _predicted_label(row: dict[str, Any]) -> str | None:
    result = row.get("result")
    if result is None or result.get("output") is None:
        return None
    output = result["output"]
    if output.get("type") == ClassProbabilities.__name__:
        probs = output["probs"]
        return str(output["labels"][max(range(len(probs)), key=lambda i: (probs[i], -i))])
    return None


async def _judge_records(
    judge: JudgeProtocol,
    criterion: str,
    records: Sequence[BenchRecord],
    attempts_path: Path,
    *,
    header: dict[str, Any],
    concurrency: int,
    keep_raw_output: bool,
) -> dict[str, dict[str, Any]]:
    """Judge the records not yet in *attempts_path*, appending as they complete; return all rows by id.

    A new file starts with *header*; an existing one must match it.
    """
    done = read_attempts(attempts_path, header)
    has_header = any(_is_header(row) for _, row in _json_lines(attempts_path)[:1])
    pending = [r for r in records if r.id not in done]
    chunk = max(1, concurrency)
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    with attempts_path.open("a", encoding="utf-8") as handle:
        if not has_header:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            handle.flush()
        for start in range(0, len(pending), chunk):
            batch = pending[start : start + chunk]
            attempts = await judge_many(
                judge, criterion, batch, concurrency=concurrency, keep_raw_output=keep_raw_output
            )
            for attempt in attempts:
                row = attempt_to_dict(attempt)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done[attempt.sample_id] = row
    return done


async def run_benchmark(
    judge: JudgeProtocol,
    records: Sequence[BenchRecord],
    *,
    out_dir: str | Path,
    criterion: str = "faithfulness",
    dev_records: Sequence[BenchRecord] | None = None,
    sample: int | None = None,
    per_class: int | None = None,
    stratify: str | None = "label",
    seed: int = 0,
    by: Iterable[str] = (),
    concurrency: int = 1,
    keep_raw_output: bool = False,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    judge_name: str | None = None,
    data_sha256: str | None = None,
    dev_sha256: str | None = None,
    model_memory: Callable[[], Awaitable[dict[str, Any]]] | None = None,
) -> BenchmarkRun:
    """Run *judge* on *records*, write ``attempts.jsonl``, ``report.json`` and ``report.md`` in *out_dir*.

    Args:
        judge: Any ``JudgeProtocol`` judge; called with ``answer=sentence`` on a
            sentence-level set, never wrapped.
        records: The evaluated set (test).
        out_dir: Output directory; an existing ``attempts.jsonl`` is resumed by id.
        criterion: The judged criterion.
        dev_records: Optional development set used to choose the pass
            threshold of a probabilistic judge. The evaluated set never
            influences the threshold.
        sample: Evaluate only this many records, drawn with *seed*.
        per_class: Evaluate at most this many records of every label (class
            quotas), drawn with *seed*; exclusive with *sample*. The raw
            metrics, Brier and ECE included, are then flagged as
            sample-dependent and the report adds the recall, the
            specificity and the precision, accuracy and Brier reweighted to
            the full set (rates per human label on a sentence-level set,
            see ``prevalence_adjusted`` and ``reweighted_brier``).
        stratify: ``label`` (default), a metadata field, or ``None`` for a plain draw.
        seed: Seed of the sample and of the bootstrap.
        by: Metadata fields for per-value breakdowns (e.g. ``task``, ``generator``).
        concurrency: Passed to ``judge_many`` (1 keeps Ollama reproducible).
        keep_raw_output: Keep raw model text in the attempts file.
        bootstrap_resamples: Number of grouped bootstrap draws.
        judge_name: Label of the judge in the report.
        data_sha256: SHA-256 of the evaluated file, written in the header of
            ``attempts.jsonl`` and checked on resume. When ``None`` (Python
            API), the SHA-256 of the records' content replaces it
            (``records_sha256``), so the check applies in both cases.
        dev_sha256: Same for the dev file and ``dev_attempts.jsonl``.
        model_memory: Optional coroutine function called once after the run,
            e.g. ``ollama_model_memory``; its result is reported as is.

    Raises:
        AttemptsMismatchError: An existing attempts file was produced with
            another judge configuration, criterion, level, sample, seed,
            stratify or data file; nothing is resumed or mixed.
    """
    if sample is not None and per_class is not None:
        msg = "sample and per_class are exclusive: a proportional sample or class quotas, not both"
        raise ValueError(msg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = judge.config(criterion)
    if data_sha256 is None:
        data_sha256 = records_sha256(records)
    if dev_sha256 is None and dev_records:
        dev_sha256 = records_sha256(dev_records)
    if per_class is not None:
        evaluated = quota_sample(records, per_class, seed)
    elif sample is not None:
        evaluated = stratified_sample(records, sample, stratify, seed)
    else:
        evaluated = list(records)
    level = evaluated[0].level if evaluated else "answer"

    # Threshold: judge's policy, or chosen on dev for a probabilistic judge
    policy_threshold = float(config.verdict_policy.pass_threshold_on_score())
    threshold = policy_threshold
    threshold_source = "verdict_policy"
    dev_rows: dict[str, dict[str, Any]] = {}
    if dev_records:
        # The dev set is never sampled: its header carries no sample, seed or stratify
        dev_header = attempts_header(
            config.config_hash,
            criterion,
            dev_records[0].level,
            sample=None,
            seed=None,
            stratify=None,
            data_sha256=dev_sha256,
            source_commit=config.source_commit,
        )
        dev_rows = await _judge_records(
            judge,
            criterion,
            dev_records,
            out / "dev_attempts.jsonl",
            header=dev_header,
            concurrency=concurrency,
            keep_raw_output=keep_raw_output,
        )
        dev_scored = [(r, _score_of(dev_rows[r.id])) for r in dev_records if _score_of(dev_rows[r.id]) is not None]
        if dev_scored and any(_probability_of(dev_rows[r.id]) is not None for r in dev_records):
            threshold = choose_threshold(
                [s for _, s in dev_scored if s is not None], [r.human_pass for r, _ in dev_scored], policy_threshold
            )
            threshold_source = f"dev ({len(dev_scored)} scored records, F1 of {NEGATIVE_CLASS} maximized)"

    header = attempts_header(
        config.config_hash,
        criterion,
        level,
        sample=sample,
        seed=seed,
        stratify=stratify,
        data_sha256=data_sha256,
        per_class=per_class,
        source_commit=config.source_commit,
    )
    rows = await _judge_records(
        judge,
        criterion,
        evaluated,
        out / "attempts.jsonl",
        header=header,
        concurrency=concurrency,
        keep_raw_output=keep_raw_output,
    )
    memory = await model_memory() if model_memory is not None else None

    scored = [r for r in evaluated if _score_of(rows[r.id]) is not None]
    failed = [r for r in evaluated if _score_of(rows[r.id]) is None]
    judge_pass = [bool(_score_of(rows[r.id]) >= threshold) for r in scored]  # type: ignore[operator]
    human_pass = [r.human_pass for r in scored]
    groups = [r.group_id for r in scored]

    report: dict[str, Any] = {
        "judge": judge_name or config.judge_class,
        "criterion": criterion,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "level": level,
        "prev_sentence": {
            "records_with_prev_sentence": sum(1 for r in evaluated if r.has_prev_sentence),
            "transmitted_to_judge": False,
        },
        "records": {"total": len(records), "evaluated": len(evaluated), "scored": len(scored), "failed": len(failed)},
        "sample": (
            {"per_class": per_class, "taken": dict(sorted(Counter(r.label for r in evaluated).items())), "seed": seed}
            if per_class is not None
            else None
            if sample is None
            else {"size": sample, "stratify": stratify, "seed": seed}
        ),
        # Raw metrics (binary, three_class, calibration, by) follow the sample's class mix when it is set by quotas
        "sample_dependent_metrics": per_class is not None,
        "prevalence_adjusted": (
            prevalence_adjusted(
                judge_pass,
                human_pass,
                sum(1 for r in records if not r.human_pass) / len(records) if records else 0.0,
                # Quotas are drawn per label: on a sentence-level set the not-supported class holds two of
                # them, so the rates are taken per label. On an answer-level set the labels are the classes
                labels=[r.label for r in scored] if level == "sentence" else None,
                full_label_counts=Counter(r.label for r in records) if level == "sentence" else None,
            )
            if per_class is not None
            else None
        ),
        "threshold": {
            "value": threshold,
            "source": threshold_source,
            "policy_threshold": policy_threshold,
            "rule": (
                f"PASS if score >= {threshold:.2f} (source: {threshold_source}); "
                f"PARTIAL and FAIL counted as {NEGATIVE_CLASS}"
            ),
        },
        "binary": binary_metrics(judge_pass, human_pass),
        "bootstrap": grouped_bootstrap(judge_pass, human_pass, groups, seed, bootstrap_resamples) if scored else None,
        # Sentence-level sets only: an answer-level set (PASS / FAIL) has one not-supported label, nothing to split
        "recall_by_label": (
            recall_by_label(judge_pass, [r.label for r in scored], groups, threshold, seed, bootstrap_resamples)
            if level == "sentence"
            else None
        ),
        "failures": {},
        "truncations": sum(1 for r in evaluated if (rows[r.id].get("trace") or {}).get("context_truncated") is True),
        "latency_ms": None,
        "input_length": None,
        # The harness process only: an Ollama model runs in another process (see model_memory)
        "peak_rss_mb_harness": round(_peak_rss_mb(), 1),
        "model_memory": memory,
        "duration_s": None,
        "by": {},
    }

    failures: dict[str, int] = {}
    for r in failed:
        kind = (rows[r.id].get("error") or {}).get("kind", "unknown")
        failures[kind] = failures.get(kind, 0) + 1
    report["failures"] = failures

    # Share of inputs beyond the judge's training length, when the judge declares one
    # (judge_specific "training_length", e.g. 512 for HHEM) and its traces count prompt tokens
    specific = dict(config.judge_specific)
    training_length = specific.get("training_length")
    if isinstance(training_length, int) and not isinstance(training_length, bool):
        counts = [(rows[r.id].get("trace") or {}).get("prompt_tokens") for r in evaluated]
        known = [int(c) for c in counts if c is not None]
        over = sum(1 for c in known if c > training_length)
        report["input_length"] = {
            "training_length": training_length,
            # What the number means for this judge: a training length unless the judge says otherwise
            # (MiniCheck declares its official truncation length)
            "kind": str(specific.get("training_length_kind") or "training length"),
            "with_token_count": len(known),
            "over_training_length": over,
            "share_over_training_length": (over / len(known)) if known else None,
            "max_prompt_tokens": max(known) if known else None,
        }

    raw_latencies = [(rows[r.id].get("trace") or {}).get("latency_ms") for r in evaluated]
    latencies = sorted(float(v) for v in raw_latencies if v is not None)
    if latencies:
        report["latency_ms"] = {
            "p50": latencies[len(latencies) // 2],
            "p90": latencies[min(len(latencies) - 1, int(0.9 * (len(latencies) - 1)))],
            "n": len(latencies),
        }

    predicted = [_predicted_label(rows[r.id]) for r in scored]
    if level == "sentence" and predicted and all(p is not None for p in predicted):
        report["three_class"] = three_class_metrics([p for p in predicted if p is not None], [r.label for r in scored])
    probabilities = [_probability_of(rows[r.id]) for r in scored]
    if probabilities and all(p is not None for p in probabilities):
        calibrated = [p for p in probabilities if p is not None]
        # Brier and ECE are computed on the evaluated sample: under class quotas they follow its mix
        report["calibration"] = {
            **calibration_metrics(calibrated, human_pass),
            "sample_dependent": per_class is not None,
        }
        if report["prevalence_adjusted"] is not None:
            # Same strata as the block: the human labels on a sentence-level set, the two classes otherwise
            report["prevalence_adjusted"]["brier_reweighted"] = reweighted_brier(
                calibrated,
                human_pass,
                report["prevalence_adjusted"]["full_set_not_supported_prevalence"],
                labels=[r.label for r in scored] if level == "sentence" else None,
                full_label_counts=Counter(r.label for r in records) if level == "sentence" else None,
            )
    if report["prevalence_adjusted"] is not None:
        # Intervals on the draws of the main bootstrap, the reweighting redone in each draw
        report["prevalence_adjusted"].update(
            prevalence_adjusted_intervals(
                judge_pass,
                human_pass,
                report["prevalence_adjusted"]["full_set_not_supported_prevalence"],
                groups,
                seed,
                bootstrap_resamples,
                labels=[r.label for r in scored] if level == "sentence" else None,
                full_label_counts=Counter(r.label for r in records) if level == "sentence" else None,
                probabilities=(
                    [p for p in probabilities if p is not None]
                    if "brier_reweighted" in report["prevalence_adjusted"]
                    else None
                ),
            )
        )

    for field_name in by:
        values: dict[str, dict[str, float | None]] = {}
        for value in sorted({str(r.metadata.get(field_name)) for r in scored}):
            idx = [i for i, r in enumerate(scored) if str(r.metadata.get(field_name)) == value]
            values[value] = binary_metrics([judge_pass[i] for i in idx], [human_pass[i] for i in idx])
        report["by"][field_name] = values

    report["duration_s"] = round(time.perf_counter() - started, 3)
    report_json = out / "report.json"
    report_md = out / "report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md.write_text(render_report(report), encoding="utf-8")
    return BenchmarkRun(
        report=report, attempts_path=out / "attempts.jsonl", report_json=report_json, report_md=report_md
    )


def _num(value: float | None, digits: int = 3) -> str:
    """A number for the Markdown report; ``undefined`` for a ``None`` kappa."""
    return "undefined" if value is None else f"{value:.{digits}f}"


def _adjusted_strata_lines(adjusted: dict[str, Any]) -> list[str]:
    """How the prevalence-adjusted rates were estimated: the per-label table, and the labels with no record."""
    by_label = adjusted.get("labels")
    if not by_label:
        return []
    lines = [
        "Rates are estimated per human label and combined with the labels' shares in the full set "
        "(the quota's mix between labels does not enter).",
        "",
        "| Human label | Class | Full set | Scored in the sample | Rate |",
        "|---|---|---|---|---|",
    ]
    for label, v in by_label.items():
        rate = "n/a" if v["rate"] is None else f"{v['rate']:.3f}"
        lines.append(
            f"| {label} | {v['human_class']} | {v['full_set_count']} ({v['full_set_share']:.1%}) | {v['n']} | {rate} |"
        )
    if adjusted.get("missing_labels"):
        lines += [
            "",
            f"No scored record for {', '.join(adjusted['missing_labels'])}: the figures that need its rate "
            "are not estimated (n/a).",
        ]
    return lines


def _recall_by_label_lines(block: dict[str, Any] | None) -> list[str]:
    """The "Recall by human label" section; nothing on an answer-level set."""
    if not block:
        return []

    def row(name: str, n: int, count: int, rate: float | None, interval: list[float] | None) -> str:
        shown = "n/a" if rate is None else f"{rate:.3f}"
        ci = "n/a" if interval is None else f"{interval[0]:.3f} to {interval[1]:.3f}"
        return f"| {name} | {n} | {count} | {shown} | {ci} |"

    lines = [
        "",
        "## Recall by human label",
        "",
        f"Share of the sentences of each human label that the judge counts as {NEGATIVE_CLASS} "
        f"(score < {block['threshold']:.2f}). Per-label rates: they do not depend on the class mix.",
        "",
        "| Human label | n | Detected | Rate | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for label, v in block["labels"].items():
        lines.append(row(label, v["n"], v["detected"], v["recall"], v["interval"]))
    s = block["specificity"]
    lines.append(row(f"{s['label']} (specificity, passed)", s["n"], s["recognized"], s["value"], s["interval"]))
    boot = block["bootstrap"]
    lost = {label: v["excluded_draws"] for label, v in block["labels"].items() if v["excluded_draws"]}
    if s["excluded_draws"]:
        lost[s["label"]] = s["excluded_draws"]
    if lost:
        lines += [
            "",
            "Draws without any record of the label, excluded from its interval: "
            + ", ".join(f"{label} {count}" for label, count in lost.items())
            + f" of {boot['resamples']}"
            + ("" if boot["reliable"] else "; intervals NOT reliable")
            + ".",
        ]
    return lines


def render_report(report: dict[str, Any]) -> str:
    """Markdown view of a report."""
    b = report["binary"]
    lines = [
        f"# Judge benchmark: {report['judge']} ({report['criterion']}, {report['level']} level)",
        "",
        f"Config hash `{report['config_hash']}`. Records: {report['records']['evaluated']} evaluated "
        f"of {report['records']['total']}, {report['records']['scored']} scored, {report['records']['failed']} failed.",
        f"Threshold {report['threshold']['value']:.2f} (source: {report['threshold']['source']}).",
        f"Prediction rule: {report['threshold']['rule']}.",
        *(
            [
                f"Class quotas: at most {report['sample']['per_class']} records per label, taken "
                + ", ".join(f"{k} {v}" for k, v in report["sample"]["taken"].items())
                + f" (seed {report['sample']['seed']}). The raw metrics below follow this class mix, not the set's: "
                "see the prevalence-adjusted figures."
            ]
            if report.get("sample_dependent_metrics")
            else []
        ),
        f"`prev_sentence` present in {report['prev_sentence']['records_with_prev_sentence']} records, not transmitted to the judge.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision ({NEGATIVE_CLASS}) | {b['precision_not_supported']:.3f} |",
        f"| Recall ({NEGATIVE_CLASS}) | {b['recall_not_supported']:.3f} |",
        f"| F1 ({NEGATIVE_CLASS}) | {b['f1_not_supported']:.3f} |",
        f"| Balanced accuracy | {b['balanced_accuracy']:.3f} |",
        f"| Accuracy | {b['accuracy']:.3f} |",
        f"| Kappa | {_num(b['kappa'])} |",
        f"| Minority human class (n) | {int(b['minority_count'])} |",
        f"| Kappa interpretable (minority >= {MIN_MINORITY_FOR_KAPPA}) | {'yes' if b['kappa_interpretable'] else 'no'} |",
        f"| Majority-class baseline (accuracy) | {b['majority_baseline_accuracy']:.3f} |",
        f"| Human PASS rate | {b['human_pass_rate']:.3f} |",
    ]
    boot = report.get("bootstrap")
    if boot:
        lines.append("")
        lines.append(
            f"Bootstrap: {boot['resamples']} draws grouped on {boot['groups']} groups (seed {boot['seed']}), "
            f"{boot['excluded_single_class_draws']} single-class draws excluded"
            + ("" if boot["reliable"] else ", intervals NOT reliable")
            + "."
        )
        if boot.get("reason"):
            lines.append(boot["reason"] + ".")
        for name, ci in boot["intervals"].items():
            if ci is not None:
                lines.append(f"- {name}: 95% CI {ci[0]:.3f} to {ci[1]:.3f}")
    lines += _recall_by_label_lines(report.get("recall_by_label"))
    if "three_class" in report:
        lines += ["", "## Three classes", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c, v in report["three_class"]["per_class"].items():
            lines.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {int(v['support'])} |")
        lines.append(
            f"\nMacro F1 {report['three_class']['macro_f1']:.3f}. Confusion (rows: truth, columns: predicted):"
        )
        classes = list(report["three_class"]["confusion"])
        lines.append("| | " + " | ".join(classes) + " |")
        lines.append("|---|" + "---|" * len(classes))
        for t in classes:
            lines.append(
                f"| {t} | " + " | ".join(str(report["three_class"]["confusion"][t][p]) for p in classes) + " |"
            )
    if "calibration" in report and report["calibration"]["brier"] is not None:
        qualifier = (
            " (sample-dependent, follows the class quotas)" if report["calibration"].get("sample_dependent") else ""
        )
        lines += [
            "",
            f"Calibration{qualifier}: Brier {report['calibration']['brier']:.4f}, "
            f"ECE ({ECE_BINS} bins) {report['calibration']['ece']:.4f}.",
        ]
    adjusted = report.get("prevalence_adjusted")
    if adjusted is not None:

        def row(name: str, key: str, digits: int = 3) -> str:
            value = adjusted.get(key)
            interval = (adjusted.get("intervals") or {}).get(key)
            shown = "n/a" if value is None else f"{value:.{digits}f}"
            ci = "n/a" if interval is None else f"{interval[0]:.{digits}f} to {interval[1]:.{digits}f}"
            return f"| {name} | {shown} | {ci} |"

        lines += [
            "",
            "## Prevalence-adjusted (independent of the quota)",
            "",
            f"Not-supported prevalence in the full set: {adjusted['full_set_not_supported_prevalence']:.1%}.",
            *_adjusted_strata_lines(adjusted),
            "",
            "| Metric | Value | 95% CI |",
            "|---|---|---|",
            row("Recall (not supported)", "recall_not_supported"),
            row("Specificity (supported recognized)", "specificity"),
            row("Precision (not supported), reweighted by full-set prevalence", "precision_not_supported_reweighted"),
            row("Accuracy, reweighted by full-set prevalence", "accuracy_reweighted"),
            row("F1 (not supported), reweighted by full-set prevalence", "f1_not_supported_reweighted"),
        ]
        if adjusted.get("brier_reweighted") is not None:
            lines.append(row("Brier, reweighted to the full set", "brier_reweighted", digits=4))
        boot = adjusted.get("bootstrap")
        if boot and boot["resamples"]:
            lost = {name: count for name, count in boot["excluded_draws"].items() if count}
            lines += [
                "",
                f"Intervals: the {boot['resamples']} draws of the bootstrap above (seed {boot['seed']}), "
                "the reweighting redone in each draw"
                + (
                    "; draws where a label of the full set is absent, excluded: "
                    + ", ".join(f"{name} {count}" for name, count in lost.items())
                    if lost
                    else ""
                )
                + ("" if boot["reliable"] else "; intervals NOT reliable")
                + ".",
            ]
        if adjusted.get("brier_reweighted") is not None:
            lines += ["", "ECE is not reweighted: it stays a figure of the sample."]
    lines += ["", "## Failures and cost", ""]
    lines.append(f"- failures by kind: {report['failures'] or 'none'}")
    lines.append(f"- probable truncations: {report['truncations']}")
    length = report.get("input_length")
    if length is not None:
        share = length["share_over_training_length"]
        lines.append(
            f"- inputs beyond the judge's declared {length.get('kind', 'training length')} "
            f"({length['training_length']} tokens): "
            f"{length['over_training_length']} of {length['with_token_count']} with a token count"
            + (f" ({share:.1%})" if share is not None else "")
            + (f", longest {length['max_prompt_tokens']} tokens" if length["max_prompt_tokens"] is not None else "")
        )
    if report.get("latency_ms"):
        lines.append(f"- latency p50 {report['latency_ms']['p50']:.0f} ms, p90 {report['latency_ms']['p90']:.0f} ms")
    lines.append(
        f"- peak RSS {report['peak_rss_mb_harness']} MiB (harness process only; an Ollama model runs in another "
        f"process), duration {report['duration_s']} s"
    )
    memory = report.get("model_memory")
    if memory is not None:
        if memory.get("available"):
            lines.append(
                f"- model memory ({memory['source']}): {memory['model']}, {memory['size_mb']} MiB resident, "
                f"{memory['size_vram_mb']} MiB in VRAM"
            )
        else:
            lines.append(f"- model memory ({memory['source']}): not available ({memory.get('reason')})")
    for field_name, values in report["by"].items():
        lines += [
            "",
            f"## By {field_name}",
            "",
            f"| Value | n | Minority class n | Kappa interpretable (>= {MIN_MINORITY_FOR_KAPPA}) | F1 (not supported) | Balanced accuracy | Kappa |",
            "|---|---|---|---|---|---|---|",
        ]
        for value, m in values.items():
            lines.append(
                f"| {value} | {int(m['n'])} | {int(m['minority_count'])} | {'yes' if m['kappa_interpretable'] else 'no'} | "
                f"{m['f1_not_supported']:.3f} | {m['balanced_accuracy']:.3f} | {_num(m['kappa'])} |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "ADJUSTED_FIGURES",
    "ANSWER_LABELS",
    "ATTEMPTS_HEADER_FIELDS",
    "BOOTSTRAP_CONFIDENCE",
    "PASS_LABELS",
    "SENTENCE_LABELS",
    "AttemptsMismatchError",
    "BenchRecord",
    "BenchmarkRun",
    "attempt_to_dict",
    "attempts_header",
    "binary_metrics",
    "calibration_metrics",
    "choose_threshold",
    "file_sha256",
    "grouped_bootstrap",
    "load_records",
    "ollama_model_memory",
    "parse_record",
    "prevalence_adjusted",
    "prevalence_adjusted_intervals",
    "quota_sample",
    "read_attempts",
    "recall_by_label",
    "records_sha256",
    "render_report",
    "reweighted_brier",
    "run_benchmark",
    "stratified_sample",
    "three_class_metrics",
]
