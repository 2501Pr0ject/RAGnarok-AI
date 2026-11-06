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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.calibration.calibrator import _cohens_kappa
from ragnarok_ai.judges.contract import (
    BinaryProbability,
    ClassProbabilities,
    JudgeAttempt,
    _jsonable,
)
from ragnarok_ai.judges.runner import judge_many

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from ragnarok_ai.core.protocols import JudgeProtocol

SENTENCE_LABELS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")
ANSWER_LABELS = ("PASS", "FAIL")
# The class of interest: a sentence the context does not support
NEGATIVE_CLASS = "not supported"
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
ECE_BINS = 10
ATTEMPTS_FORMAT = 1
# Fields of the attempts header; a run must match all of them to resume. Checked in this
# order so that the message names the plain field (criterion differs) before the hash it changes.
ATTEMPTS_HEADER_FIELDS = ("criterion", "level", "sample", "seed", "stratify", "data_sha256", "config_hash")


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
        human_pass=label in ("SUPPORTED", "PASS"),
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


def attempts_header(
    config_hash: str,
    criterion: str,
    level: str,
    *,
    sample: int | None,
    seed: int | None,
    stratify: str | None,
    data_sha256: str | None,
) -> dict[str, Any]:
    """The first line of an attempts file: what the run must match to resume from it."""
    return {
        "header": ATTEMPTS_FORMAT,
        "config_hash": config_hash,
        "criterion": criterion,
        "level": level,
        "sample": sample,
        "seed": seed,
        "stratify": stratify,
        "data_sha256": data_sha256,
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


def binary_metrics(judge_pass: Sequence[bool], human_pass: Sequence[bool]) -> dict[str, float]:
    precision, recall, f1 = _f1_not_supported(judge_pass, human_pass)
    n = len(human_pass)
    majority = max(sum(human_pass), n - sum(human_pass)) / n if n else 0.0
    return {
        "n": float(n),
        "accuracy": sum(1 for j, h in zip(judge_pass, human_pass, strict=True) if j == h) / n if n else 0.0,
        "balanced_accuracy": _balanced_accuracy(judge_pass, human_pass) if n else 0.0,
        "kappa": _cohens_kappa(judge_pass, human_pass) if n else 0.0,
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


def grouped_bootstrap(
    judge_pass: Sequence[bool],
    human_pass: Sequence[bool],
    groups: Sequence[str],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """95% percentile intervals of F1 (negative class), balanced accuracy and kappa, resampling by group.

    Draws with a single human class are excluded and counted; the intervals
    are flagged unreliable above 10% exclusions.
    """
    units: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        units.setdefault(g, []).append(i)
    unit_list = list(units.values())
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {"f1_not_supported": [], "balanced_accuracy": [], "kappa": []}
    excluded = 0
    for _ in range(resamples):
        drawn = [i for _ in range(len(unit_list)) for i in unit_list[rng.randrange(len(unit_list))]]
        h = [human_pass[i] for i in drawn]
        if all(h) or not any(h):
            excluded += 1
            continue
        j = [judge_pass[i] for i in drawn]
        samples["f1_not_supported"].append(_f1_not_supported(j, h)[2])
        samples["balanced_accuracy"].append(_balanced_accuracy(j, h))
        samples["kappa"].append(_cohens_kappa(j, h))

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        values = sorted(values)
        return [
            values[max(0, math.floor(0.025 * (len(values) - 1)))],
            values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))],
        ]

    return {
        "resamples": resamples,
        "seed": seed,
        "groups": len(unit_list),
        "excluded_single_class_draws": excluded,
        "reliable": excluded <= 0.10 * resamples,
        "intervals": {k: interval(v) for k, v in samples.items()},
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
        stratify: ``label`` (default), a metadata field, or ``None`` for a plain draw.
        seed: Seed of the sample and of the bootstrap.
        by: Metadata fields for per-value breakdowns (e.g. ``task``, ``generator``).
        concurrency: Passed to ``judge_many`` (1 keeps Ollama reproducible).
        keep_raw_output: Keep raw model text in the attempts file.
        bootstrap_resamples: Number of grouped bootstrap draws.
        judge_name: Label of the judge in the report.
        data_sha256: SHA-256 of the evaluated file, written in the header of
            ``attempts.jsonl`` and checked on resume.
        dev_sha256: Same for the dev file and ``dev_attempts.jsonl``.
        model_memory: Optional coroutine function called once after the run,
            e.g. ``ollama_model_memory``; its result is reported as is.

    Raises:
        AttemptsMismatchError: An existing attempts file was produced with
            another judge configuration, criterion, level, sample, seed,
            stratify or data file; nothing is resumed or mixed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = judge.config(criterion)
    evaluated = stratified_sample(records, sample, stratify, seed) if sample is not None else list(records)
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
        config.config_hash, criterion, level, sample=sample, seed=seed, stratify=stratify, data_sha256=data_sha256
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
        "sample": None if sample is None else {"size": sample, "stratify": stratify, "seed": seed},
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
        "failures": {},
        "truncations": sum(1 for r in evaluated if (rows[r.id].get("trace") or {}).get("context_truncated") is True),
        "latency_ms": None,
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
        report["calibration"] = calibration_metrics([p for p in probabilities if p is not None], human_pass)

    for field_name in by:
        values: dict[str, dict[str, float]] = {}
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
        f"`prev_sentence` present in {report['prev_sentence']['records_with_prev_sentence']} records, not transmitted to the judge.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision ({NEGATIVE_CLASS}) | {b['precision_not_supported']:.3f} |",
        f"| Recall ({NEGATIVE_CLASS}) | {b['recall_not_supported']:.3f} |",
        f"| F1 ({NEGATIVE_CLASS}) | {b['f1_not_supported']:.3f} |",
        f"| Balanced accuracy | {b['balanced_accuracy']:.3f} |",
        f"| Accuracy | {b['accuracy']:.3f} |",
        f"| Kappa | {b['kappa']:.3f} |",
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
        for name, ci in boot["intervals"].items():
            if ci is not None:
                lines.append(f"- {name}: 95% CI {ci[0]:.3f} to {ci[1]:.3f}")
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
        lines += [
            "",
            f"Calibration: Brier {report['calibration']['brier']:.4f}, ECE ({ECE_BINS} bins) {report['calibration']['ece']:.4f}.",
        ]
    lines += ["", "## Failures and cost", ""]
    lines.append(f"- failures by kind: {report['failures'] or 'none'}")
    lines.append(f"- probable truncations: {report['truncations']}")
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
            "| Value | n | F1 (not supported) | Balanced accuracy | Kappa |",
            "|---|---|---|---|---|",
        ]
        for value, m in values.items():
            lines.append(
                f"| {value} | {int(m['n'])} | {m['f1_not_supported']:.3f} | {m['balanced_accuracy']:.3f} | {m['kappa']:.3f} |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "ANSWER_LABELS",
    "ATTEMPTS_HEADER_FIELDS",
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
    "read_attempts",
    "render_report",
    "run_benchmark",
    "stratified_sample",
    "three_class_metrics",
]
