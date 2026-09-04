"""Mechanical checks on the external annotation batches.

Verifies, for every batch file under ``batches/``:

1. no forbidden field or value is exposed (configuration names or
   letters, question type, expected behavior, experimental target,
   timings, temperature, model names, run parameters);
2. items carry exactly the allowed payload fields;
3. no item appears twice inside one batch;
4. every non-smoke item appears in exactly COPIES batches, each item's
   copies in distinct batches;
5. every item's question/answer/context match the frozen raw outputs
   byte for byte (batches are a projection of the run, never an edit);
6. the batch inventory matches ``batches_manifest.json``.

Exit code 0 = safe to publish. Any failure prints the offending batch.

Usage:
    python benchmark/annotation/external/check_blindness.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BENCH = HERE.parent.parent

ALLOWED_ITEM_FIELDS = {
    "item_key",
    "case_id",
    "domain",
    "question",
    "reference_information",
    "reference_chunks",
    "retrieved_context",
    "answer",
}
ALLOWED_BATCH_FIELDS = {"batch_id", "domain", "smoke", "study_version", "items"}

# Values that must never appear anywhere in a batch file
FORBIDDEN_PATTERNS = [
    r"baseline",
    r"bad_chunking",
    r"weak_embedding",
    r"low_k",
    r"noisy_context",
    r"broken_retrieval",
    r"hallucination_prone",
    r'"config"',
    r'"config_name"',
    r'"rag_config"',
    r"question_type",
    r"expected_behavior",
    r"experimental_target",
    r"timings",
    r"temperature",
    r"qwen",
    r"nomic",
    r"all-minilm",
    r"top_k",
]


def main() -> int:
    """Run every check; print failures; non-zero exit on any."""
    errors: list[str] = []
    manifest = json.loads((HERE / "batches_manifest.json").read_text())
    copies = manifest["copies"]
    run_dir = BENCH / "pilot-run" / manifest["run_id"] / "outputs"
    cases = {c["id"]: c for c in yaml.safe_load((BENCH / "questions" / "pilot.yaml").read_text())["cases"]}

    # Raw outputs indexed by (case_id, answer-hash) for content matching
    raw_by_case: dict[str, list[dict]] = {}
    for f in run_dir.glob("*/*.json"):
        d = json.loads(f.read_text())
        raw_by_case.setdefault(d["case_id"], []).append(d)

    batch_files = sorted(HERE.glob("batches/*/*.json"))
    seen_inventory: dict[str, int] = {}
    item_batches: dict[str, list[str]] = {}

    for bf in batch_files:
        raw_text = bf.read_text()
        batch = json.loads(raw_text)
        bid = batch["batch_id"]
        seen_inventory[bid] = len(batch["items"])

        # 1. forbidden values anywhere in the file
        for pat in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                # tolerate matches inside corpus/document text is NOT ok either:
                # config names and parameter names simply never occur in the
                # docs; report every hit for human review.
                errors.append(f"{bid}: forbidden pattern {pat!r} at offset {m.start()}")
                break

        # 2. field allowlists
        extra = set(batch) - ALLOWED_BATCH_FIELDS
        if extra:
            errors.append(f"{bid}: unexpected batch fields {sorted(extra)}")
        keys_in_batch = Counter()
        for item in batch["items"]:
            extra = set(item) - ALLOWED_ITEM_FIELDS
            missing = ALLOWED_ITEM_FIELDS - set(item)
            if extra:
                errors.append(f"{bid}/{item.get('item_key')}: unexpected item fields {sorted(extra)}")
            if missing:
                errors.append(f"{bid}/{item.get('item_key')}: missing item fields {sorted(missing)}")
            keys_in_batch[item["item_key"]] += 1

            # 5. content matches exactly one frozen raw output
            matches = [
                r
                for r in raw_by_case.get(item["case_id"], [])
                if r["answer"] == item["answer"]
                and r["question"] == item["question"]
                and [c["text"] for c in item["retrieved_context"]] == [c["text"] for c in r["retrieved"]]
            ]
            if len(matches) != 1:
                errors.append(f"{bid}/{item['item_key']}: matches {len(matches)} raw outputs (expected exactly 1)")
            if item["reference_information"] != cases[item["case_id"]]["reference_answer"]:
                errors.append(f"{bid}/{item['item_key']}: reference_information differs from the frozen case")

            if not batch["smoke"]:
                item_batches.setdefault(item["item_key"], []).append(bid)

        # 3. intra-batch duplicates
        dupes = [k for k, n in keys_in_batch.items() if n > 1]
        if dupes:
            errors.append(f"{bid}: duplicated items {dupes}")

    # 4. copies across distinct batches
    for key, bids in item_batches.items():
        if len(bids) != copies:
            errors.append(f"item {key}: appears in {len(bids)} non-smoke batches (expected {copies})")
        if len(set(bids)) != len(bids):
            errors.append(f"item {key}: copies share a batch {bids}")

    # 6. manifest consistency
    if seen_inventory != manifest["batches"]:
        errors.append(f"batch inventory differs from manifest: {seen_inventory} vs {manifest['batches']}")

    print(f"{len(batch_files)} batch files checked, {sum(seen_inventory.values())} item slots")
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors[:30]:
            print(f"  - {e}")
        return 1
    print("all blindness and integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
