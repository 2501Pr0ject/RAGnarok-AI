"""Shared helpers for the judge calibration experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CRITERIA = ("faithfulness", "relevance", "hallucination", "completeness")

# Binary labeling guidelines, aligned with the study's benchmark/ANNOTATION.md
# (faithfulness and answer_relevance). The labeler sees only the question, the
# retrieved context and the answer: no reference answer, no case type, so each
# rule is decidable from those three alone. Stricter than the judge rubric
# (judge PASS = score >= 4/5): the judge threshold is tuned on the dev split.
# Frozen before labeling: bump GUIDELINES_VERSION for any change.
GUIDELINES_VERSION = "2.0"
GUIDELINES = {
    "faithfulness": (
        "PASS si chaque affirmation factuelle de la réponse est appuyée par le contexte fourni "
        "(pas par vos propres connaissances de l'outil). FAIL si au moins une affirmation n'est pas "
        "appuyée ou contredit le contexte. Paraphrase et résumé raisonnable sont appuyés ; une "
        "précision ajoutée absente du contexte (version, valeur par défaut, option) ne l'est pas = FAIL. "
        "Une réponse vraie dans le monde réel mais non appuyée ici = FAIL. Accepter une prémisse "
        "fausse que le contexte contredit = FAIL. Abstention ou réponse vide = PASS (aucune "
        "affirmation)."
    ),
    "relevance": (
        "PASS si la réponse traite la question effectivement posée. FAIL si elle est hors sujet, "
        "répond à une autre question ou reformule la question sans y répondre. Une abstention "
        "(« la documentation ne couvre pas ce point ») est PASS si le contexte fourni ne contient "
        "effectivement pas l'information, FAIL sinon. Accepter une prémisse fausse et y répondre = "
        "PASS ici (la réponse traite la question ; l'erreur relève de faithfulness). Réponse vide = "
        "FAIL. Ne pas juger l'exactitude ici."
    ),
    # Outside the current protocol (faithfulness and relevance only); kept for the judge API.
    "hallucination": (
        "PASS si aucune information factuelle significative n'est absente du contexte. "
        "Détails triviaux tolérés. Sinon FAIL."
    ),
    "completeness": (
        "PASS si la réponse couvre l'essentiel de ce que le contexte permet de répondre. Un point clé manquant = FAIL."
    ),
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, ignoring blank lines. Missing file -> empty list."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            msg = f"{p}:{line_no}: invalid JSON ({exc})"
            raise ValueError(msg) from exc
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row and flush immediately (safe to interrupt)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL file, replacing it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_samples(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load samples keyed by id and validate required fields."""
    samples: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        missing = [k for k in ("id", "question", "context", "answer") if not row.get(k)]
        if missing:
            msg = f"Sample {row.get('id', '?')} is missing fields: {missing}"
            raise ValueError(msg)
        sid = str(row["id"])
        if sid in samples:
            msg = f"Duplicate sample id: {sid}"
            raise ValueError(msg)
        row.setdefault("origin", "natural")
        samples[sid] = row
    return samples


def load_labels(path: str | Path) -> dict[str, dict[str, bool]]:
    """Load human labels keyed by sample id. Later lines override earlier ones."""
    labels: dict[str, dict[str, bool]] = {}
    for row in read_jsonl(path):
        labels[str(row["id"])] = {k: bool(v) for k, v in row.get("labels", {}).items()}
    return labels


def split_of(sample_id: str, dev_fraction: float = 0.5, salt: str = "calib-v1") -> str:
    """Deterministic dev/test assignment from the sample id.

    Stable across runs and independent of file order, so the threshold
    tuned on 'dev' is never measured on the samples it was tuned on.
    """
    digest = hashlib.sha256(f"{salt}:{sample_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "dev" if bucket < dev_fraction else "test"
