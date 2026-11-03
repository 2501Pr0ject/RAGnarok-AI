#!/usr/bin/env python3
"""Inject the RAG configuration as `origin`, once labeling is finished.

Refuses to run while any sample still lacks a label for the chosen
criteria: revealing the configurations is what ends the blind phase.
The output feeds run_calibration.py, which then reports results per
configuration.

Usage:
    python reveal_origin.py samples.jsonl pilot_key.json labels.jsonl samples_revealed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import CRITERIA, load_labels, load_samples, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", help="samples JSONL (origin='natural')")
    ap.add_argument("key", help="pilot_key.json (id -> config)")
    ap.add_argument("labels", help="first-pass labels JSONL")
    ap.add_argument("out", help="output JSONL with origin = configuration")
    ap.add_argument("--criteria", nargs="+", default=["faithfulness", "relevance"], choices=CRITERIA)
    args = ap.parse_args()

    samples = load_samples(args.samples)
    labels = load_labels(args.labels)
    missing = [sid for sid in sorted(samples) if not all(c in labels.get(sid, {}) for c in args.criteria)]
    if missing:
        print(
            f"Annotation incomplète : {len(missing)} échantillon(s) sans label, configurations non révélées.",
            file=sys.stderr,
        )
        return 1

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))["ids"]
    rows = [{**s, "origin": f"{key[sid]['config']}-{key[sid]['config_name']}"} for sid, s in sorted(samples.items())]
    write_jsonl(args.out, rows)
    print(f"{len(rows)} échantillons écrits dans {args.out} avec leur configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
