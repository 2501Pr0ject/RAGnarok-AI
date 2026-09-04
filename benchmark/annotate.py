"""Prepare and merge the blind human annotation of a pilot run.

Two subcommands, deliberately minimal (see ANNOTATION.md for the rules):

``prepare``
    Builds a seeded, shuffled, config-blind annotation sheet from a run's
    outputs. Each item shows: the question, the case's reference answer
    and supporting chunks (annotator ground truth), the retrieved
    context, and the answer — but NOT which RAG configuration produced
    it. The item→(case, config) mapping is written to a separate
    ``blind_map.json`` that the annotator must not open before merging.
    Timings are excluded from the sheet (they correlate with configs).

``merge``
    Joins the filled sheet with the blind map into schema-conformant
    annotation records (schemas/annotation.schema.json) and validates
    them. Refuses to merge while items remain unlabeled.

Usage:
    python benchmark/annotate.py prepare benchmark/pilot-run/<run_id>
    # ... human annotates annotation/<run_id>/sheet.yaml, in order ...
    python benchmark/annotate.py merge benchmark/pilot-run/<run_id> --round primary

Structural blindness caveat (documented limitation): the shape of the
retrieved context can hint at the defect class (a single chunk suggests
low k, tiny chunks suggest degraded chunking). The sheet still never
names the configuration, and items are shuffled across configs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SHUFFLE_SEED = 42  # recorded here; the sheet also records it

CRITERIA = ("retrieval_relevance", "faithfulness", "answer_relevance", "completeness")


def load_cases() -> dict[str, dict]:
    """Pilot cases by id."""
    cases = yaml.safe_load((HERE / "questions" / "pilot.yaml").read_text())["cases"]
    return {c["id"]: c for c in cases}


def prepare(run_dir: Path) -> int:
    """Build the blind, shuffled annotation sheet for a run."""
    cases = load_cases()
    outputs = sorted(run_dir.glob("outputs/*/*.json"))
    if not outputs:
        print(f"no outputs under {run_dir}", file=sys.stderr)
        return 1

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(outputs)

    ann_dir = HERE / "annotation" / run_dir.name
    ann_dir.mkdir(parents=True, exist_ok=True)

    blind_map: dict[str, dict[str, str]] = {}
    items = []
    for i, f in enumerate(outputs, start=1):
        d = json.loads(f.read_text())
        case = cases[d["case_id"]]
        blind_map[str(i)] = {"case_id": d["case_id"], "rag_config": d["config_name"]}
        items.append(
            {
                "item": i,
                "question": d["question"],
                "question_type": case["question_type"],
                "expected_behavior": case["expected_behavior"],
                "reference_answer": case["reference_answer"],
                "reference_chunks": case["supporting_chunks"],
                "retrieved_context": [{"id": r["id"], "text": r["text"]} for r in d["retrieved"]],
                "answer": d["answer"],
                # ── fill these in, per ANNOTATION.md ────────────────────
                "labels": dict.fromkeys(CRITERIA),
                "confidence": None,  # low | medium | high
                "note": "",
                "invalid": False,
            }
        )

    sheet = {
        "run_id": run_dir.name,
        "shuffle_seed": SHUFFLE_SEED,
        "instructions": "Annotate IN ORDER per ANNOTATION.md. Do not open blind_map.json.",
        "items": items,
    }
    (ann_dir / "sheet.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True, width=100))
    (ann_dir / "blind_map.json").write_text(json.dumps(blind_map, indent=2))
    (ann_dir / "REGISTRY.md").write_text(
        "# Ambiguity registry\n\n"
        "Log (do not fix) every ambiguity met while annotating: unclear\n"
        "rules, badly phrased cases, hard judgment calls. One bullet per\n"
        "entry with the item number.\n\n"
        "- \n"
    )
    print(f"{len(items)} items -> {ann_dir / 'sheet.yaml'}")
    print(f"blind map -> {ann_dir / 'blind_map.json'} (do not open before merge)")
    print(f"registry  -> {ann_dir / 'REGISTRY.md'}")
    return 0


def merge(run_dir: Path, annotator: str, round_name: str) -> int:
    """Join the filled sheet with the blind map into schema records."""
    import jsonschema

    ann_dir = HERE / "annotation" / run_dir.name
    sheet = yaml.safe_load((ann_dir / "sheet.yaml").read_text())
    blind_map = json.loads((ann_dir / "blind_map.json").read_text())
    schema = json.loads((HERE / "schemas" / "annotation.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)

    records = []
    incomplete = []
    for item in sheet["items"]:
        key = blind_map[str(item["item"])]
        if item.get("invalid"):
            rec: dict = {"invalid": True, "note": item.get("note", "")}
        else:
            labels = item.get("labels") or {}
            if any(labels.get(c) not in (0, 1) for c in CRITERIA) or item.get("confidence") not in (
                "low",
                "medium",
                "high",
            ):
                incomplete.append(item["item"])
                continue
            rec = {
                "labels": {c: int(labels[c]) for c in CRITERIA},
                "confidence": item["confidence"],
            }
            if item.get("note"):
                rec["note"] = item["note"]
        rec.update(
            {
                "case_id": key["case_id"],
                "rag_config": key["rag_config"],
                "annotator": annotator,
                "annotated_at": datetime.now(timezone.utc).isoformat(),
                "round": round_name,
            }
        )
        errors = list(validator.iter_errors(rec))
        if errors:
            print(f"item {item['item']}: {errors[0].message}", file=sys.stderr)
            return 1
        records.append(rec)

    if incomplete:
        print(f"{len(incomplete)} unlabeled item(s): {incomplete[:15]}{'...' if len(incomplete) > 15 else ''}")
        print("merge refused - finish the sheet first")
        return 1

    out = ann_dir / f"{round_name}.yaml"
    out.write_text(yaml.safe_dump({"records": records}, sort_keys=False, allow_unicode=True))
    print(f"{len(records)} annotation records -> {out}")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("run_dir", type=Path)
    p_merge = sub.add_parser("merge")
    p_merge.add_argument("run_dir", type=Path)
    p_merge.add_argument("--annotator", default="maintainer")
    p_merge.add_argument("--round", default="primary", choices=["primary", "reannotation", "second_annotator"])
    args = parser.parse_args()
    if args.cmd == "prepare":
        return prepare(args.run_dir)
    return merge(args.run_dir, args.annotator, args.round)


if __name__ == "__main__":
    sys.exit(main())
