#!/usr/bin/env python3
"""Pick the samples to label: a stratified subset of the pilot.

Stratified by RAG configuration (A-G, ~17 each) and, within each
configuration, by question type (direct, context, multi, trap, unans,
contra), so that degraded configurations provide the failures that a
judge must catch. Ids stay opaque and every sample keeps
origin="natural": the configuration is only injected after labeling
(reveal_origin.py), so the labeler stays blind to it.

Usage:
    python select_samples.py pilot.jsonl pilot_key.json samples.jsonl --n 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from common import load_samples, write_jsonl


def question_type(case_id: str) -> str:
    """'docker_trap_001' -> 'trap'."""
    return case_id.split("_")[1]


def spread(total: int, buckets: list[str], rng: random.Random) -> dict[str, int]:
    """Split total as evenly as possible; the remainder goes to randomly drawn buckets."""
    base, extra = divmod(total, len(buckets))
    lucky = set(rng.sample(buckets, extra))
    return {b: base + (b in lucky) for b in buckets}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", help="pilot JSONL (opaque ids)")
    ap.add_argument("key", help="pilot_key.json (id -> config, case)")
    ap.add_argument("out", help="output samples JSONL")
    ap.add_argument("--n", type=int, default=120, help="total number of samples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    samples = load_samples(args.pilot)
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))["ids"]
    rng = random.Random(args.seed)

    by_config: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid in sorted(samples):
        info = key[sid]
        by_config[info["config"]][question_type(info["case_id"])].append(sid)

    per_config = spread(args.n, sorted(by_config), rng)
    picked: list[str] = []
    for config in sorted(by_config):
        types = by_config[config]
        quota = spread(per_config[config], sorted(types), rng)
        for qtype in sorted(types):
            if quota[qtype] > len(types[qtype]):
                print(
                    f"Config {config}, type {qtype} : {quota[qtype]} demandés, {len(types[qtype])} disponibles.",
                    file=sys.stderr,
                )
                return 1
            picked += rng.sample(types[qtype], quota[qtype])

    write_jsonl(args.out, [samples[sid] for sid in sorted(picked)])
    print(f"{len(picked)} échantillons écrits dans {args.out} (configuration masquée : origin='natural').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
