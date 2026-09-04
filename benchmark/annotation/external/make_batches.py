"""Generate blind, stratified annotation batches for external annotators.

Soft-launch design (PROTOCOL.md amendment of 2026-09-04, decisions D1-D4):

- 10 cases per RAG configuration (70 of the 210 pilot outputs), stratified
  by question type within each configuration, seeded;
- 3 independent annotations per selected output;
- domain-specific batches of at most 15 items, the three copies of an
  item always landing in three different batches (three different
  annotators);
- 3 identical cross-domain SMOKE batches of 10 items for the internal
  end-to-end chain test (their annotations are never analyzed as
  results).

Batch files contain ONLY what an annotator may see: question, reference
information, retrieved context, generated answer. Never: configuration,
question type, expected behavior, timings, model or run parameters.
``check_blindness.py`` enforces this mechanically.

The item_key -> (case_id, rag_config) mapping is written to
``key_map.json``, which is gitignored (server-side only) and fully
re-derivable by re-running this script (fixed seed).

Usage:
    python benchmark/annotation/external/make_batches.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BENCH = HERE.parent.parent
RUN_ID = "pilot-20260904T145740Z"
SEED = 42
CASES_PER_CONFIG = 10
COPIES = 3
BATCH_SIZE = 15
SMOKE_ITEMS = 10

CONFIGS = [
    "baseline",
    "bad_chunking",
    "weak_embedding",
    "low_k",
    "noisy_context",
    "broken_retrieval",
    "hallucination_prone",
]
CONFIG_DIRS = dict(zip("ABCDEFG", CONFIGS, strict=True))


def item_key(case_id: str, config: str) -> str:
    """Opaque, deterministic item identifier (raises casual-lookup cost)."""
    return hashlib.sha256(f"{RUN_ID}:{case_id}:{config}".encode()).hexdigest()[:10]


def select_cases(cases: list[dict], rng: random.Random) -> dict[str, list[str]]:
    """Per config: CASES_PER_CONFIG case ids, stratified by question type."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        by_type[c["question_type"]].append(c["id"])

    selected: dict[str, list[str]] = {}
    for config in CONFIGS:
        picked: list[str] = []
        # one per type first, then spread the remainder across types
        types = sorted(by_type)
        rng.shuffle(types)
        pools = {t: sorted(by_type[t]) for t in types}
        for t in types:
            picked.append(rng.choice(pools[t]))
        while len(picked) < CASES_PER_CONFIG:
            t = types[len(picked) % len(types)]
            remaining = [c for c in pools[t] if c not in picked]
            if remaining:
                picked.append(rng.choice(remaining))
        selected[config] = sorted(picked[:CASES_PER_CONFIG])
    return selected


def load_output(case_id: str, config: str) -> dict:
    """Raw pilot output for one (case, config)."""
    letter = next(k for k, v in CONFIG_DIRS.items() if v == config)
    return json.loads((BENCH / "pilot-run" / RUN_ID / "outputs" / letter / f"{case_id}.json").read_text())


def payload(case: dict, output: dict) -> dict:
    """The annotator-visible fields for one item — nothing else."""
    return {
        "item_key": item_key(case["id"], output["config_name"]),
        "case_id": case["id"],
        "domain": case["corpus"],
        "question": output["question"],
        "reference_information": case["reference_answer"],
        "reference_chunks": case["supporting_chunks"],
        "retrieved_context": [{"id": r["id"], "text": r["text"]} for r in output["retrieved"]],
        "answer": output["answer"],
    }


def distribute(items: list[dict], rng: random.Random) -> list[list[dict]]:
    """COPIES copies of each item into batches of <= BATCH_SIZE, no batch
    holding the same item twice."""
    n_batches = max(COPIES, -(-len(items) * COPIES // BATCH_SIZE))
    batches: list[list[dict]] = [[] for _ in range(n_batches)]
    order = list(items)
    rng.shuffle(order)
    for item in order:
        candidates = sorted(range(n_batches), key=lambda b: len(batches[b]))
        chosen = [b for b in candidates if item["item_key"] not in {x["item_key"] for x in batches[b]}][:COPIES]
        if len(chosen) < COPIES:
            msg = "cannot place item copies without duplication - increase batch count"
            raise RuntimeError(msg)
        for b in chosen:
            batches[b].append(item)
    for b in batches:
        rng.shuffle(b)
    return batches


def main() -> int:
    """Generate batches, smoke batches and the server-side key map."""
    rng = random.Random(SEED)
    cases = {c["id"]: c for c in yaml.safe_load((BENCH / "questions" / "pilot.yaml").read_text())["cases"]}
    selected = select_cases(list(cases.values()), rng)

    items: list[dict] = []
    key_map: dict[str, dict[str, str]] = {}
    for config, case_ids in selected.items():
        for cid in case_ids:
            out = load_output(cid, config)
            p = payload(cases[cid], out)
            items.append(p)
            key_map[p["item_key"]] = {"case_id": cid, "rag_config": config}

    # domain batches
    batches_dir = HERE / "batches"
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for p in items:
        by_domain[p["domain"]].append(p)

    manifest = {"run_id": RUN_ID, "seed": SEED, "copies": COPIES, "batch_size": BATCH_SIZE, "batches": {}}
    for domain, dom_items in sorted(by_domain.items()):
        dom_dir = batches_dir / domain
        dom_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(distribute(dom_items, rng), start=1):
            bid = f"{domain}-b{i:02d}"
            (dom_dir / f"{bid}.json").write_text(
                json.dumps(
                    {"batch_id": bid, "domain": domain, "smoke": False, "study_version": RUN_ID, "items": batch},
                    indent=2,
                )
            )
            manifest["batches"][bid] = len(batch)

    # smoke batches: three identical cross-domain sets of SMOKE_ITEMS
    smoke_pool = sorted(items, key=lambda p: p["item_key"])
    smoke_items = rng.sample(smoke_pool, SMOKE_ITEMS)
    smoke_dir = batches_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        bid = f"smoke-b{i:02d}"
        (smoke_dir / f"{bid}.json").write_text(
            json.dumps(
                {"batch_id": bid, "domain": "mixed", "smoke": True, "study_version": RUN_ID, "items": smoke_items},
                indent=2,
            )
        )
        manifest["batches"][bid] = len(smoke_items)

    (HERE / "key_map.json").write_text(json.dumps(key_map, indent=2))
    (HERE / "batches_manifest.json").write_text(json.dumps(manifest, indent=2))

    n_conf = Counter(km["rag_config"] for km in key_map.values())
    n_dom = Counter(p["domain"] for p in items)
    n_type = Counter(cases[p["case_id"]]["question_type"] for p in items)
    print(f"{len(items)} items selected ({COPIES} annotations each = {len(items) * COPIES} assignments)")
    print("per config:", dict(sorted(n_conf.items())))
    print("per domain:", dict(sorted(n_dom.items())))
    print("per type:  ", dict(sorted(n_type.items())))
    print("batches:", dict(manifest["batches"]))
    print("key_map.json written (server-side only, gitignored, re-derivable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
