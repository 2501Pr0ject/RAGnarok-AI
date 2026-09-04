"""Convert the webapp's raw export into schema-conformant annotation records.

The website never knows which configuration produced an item; the join
with ``key_map.json`` (gitignored, re-derivable via make_batches.py)
happens HERE, offline, on the operator's machine.

Input: the JSON dump from ``/api/export?key=...`` saved to a file.
Output: ``external_<round>.yaml`` with records validating against
``schemas/annotation.schema.json`` (round="external"), plus a summary
of per-item coverage. Smoke-batch annotations are written to a separate
file and never analyzed as results.

Usage:
    python benchmark/annotation/external/export_external.py dump.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BENCH = HERE.parent.parent
RUN_ID = "pilot-20260904T145740Z"


def main() -> int:
    """Convert, validate and summarize the webapp dump."""
    import jsonschema

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="JSON file saved from /api/export")
    args = parser.parse_args()

    dump = json.loads(args.dump.read_text())
    key_map = json.loads((HERE / "key_map.json").read_text())
    schema = json.loads((BENCH / "schemas" / "annotation.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    annotators = {a["id"]: a for a in dump["annotators"]}

    records: list[dict] = []
    smoke_records: list[dict] = []
    skipped = 0
    for a in dump["annotations"]:
        ann = annotators.get(a["annotator_id"])
        mapping = key_map.get(a["item_key"])
        if ann is None or mapping is None:
            skipped += 1
            continue
        rec = {
            "case_id": mapping["case_id"],
            "rag_config": mapping["rag_config"],
            "annotator": a["annotator_id"],
            "annotated_at": a["submitted_at"],
            "round": "external",
            "source": ann.get("source", "other"),
            "batch_id": ann["batch_id"],
            "study_version": RUN_ID,
            "labels": {k: int(v) for k, v in a["labels"].items()},
            "confidence": a["confidence"],
            "ambiguity_flag": bool(a.get("ambiguity")),
            "qualification": {"domain_level": ann.get("level") or "undisclosed"},
        }
        if a.get("note"):
            rec["note"] = a["note"]
        if a.get("started_at"):
            rec["started_at"] = a["started_at"]
        if a.get("submitted_at"):
            rec["completed_at"] = a["submitted_at"]
        errors = list(validator.iter_errors(rec))
        if errors:
            print(f"{a['annotator_id']}/{a['item_key']}: {errors[0].message}", file=sys.stderr)
            return 1
        (smoke_records if ann["batch_id"].startswith("smoke") else records).append(rec)

    out = HERE / "external_primary.yaml"
    out.write_text(yaml.safe_dump({"records": records}, sort_keys=False, allow_unicode=True))
    smoke_out = HERE / "external_smoke.yaml"
    smoke_out.write_text(yaml.safe_dump({"records": smoke_records}, sort_keys=False, allow_unicode=True))

    coverage = Counter((r["case_id"], r["rag_config"]) for r in records)
    print(f"{len(records)} external records -> {out}")
    print(f"{len(smoke_records)} smoke records  -> {smoke_out} (chain test only, never analyzed)")
    if skipped:
        print(f"{skipped} annotation(s) skipped (unknown annotator or item key)")
    if coverage:
        depth = Counter(coverage.values())
        print(f"item coverage: {dict(sorted(depth.items()))} (annotations per item: count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
