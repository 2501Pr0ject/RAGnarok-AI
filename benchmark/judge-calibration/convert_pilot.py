#!/usr/bin/env python3
"""Convert the pilot run outputs to the calibration format (PROTOCOLE.md §3).

`context` is the exact string run_pilot.py inserted into the generator
prompt: the retrieved chunks, in retrieval order, joined with
"\\n\\n---\\n\\n". Each rebuilt prompt is checked against the
prompt_sha256 recorded by the run; any mismatch aborts.

Ids are opaque (p001..p210, seeded shuffle): they reveal neither the
configuration nor the question. The id -> configuration mapping goes to
a separate key file, which stays out of the repository until labeling
is finished.

Usage (from benchmark/judge-calibration/):
    python convert_pilot.py ../pilot-run/pilot-20260904T145740Z ../rag_configs/configs.yaml pilot.jsonl pilot_key.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml
from common import write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="pilot run directory (run_manifest.json + outputs/)")
    ap.add_argument("configs", help="rag_configs/configs.yaml (prompt templates)")
    ap.add_argument("out", help="output samples JSONL")
    ap.add_argument("key_out", help="output key JSON (id -> configuration, case)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    prompts = yaml.safe_load(Path(args.configs).read_text(encoding="utf-8"))["prompts"]
    outputs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((run_dir / "outputs").glob("*/*.json"))]

    random.Random(f"{manifest['run_id']}:calibration-ids").shuffle(outputs)
    samples, key = [], {}
    for n, d in enumerate(outputs, start=1):
        cfg = manifest["configs"][d["config"]]
        context = "\n\n---\n\n".join(c["text"] for c in d["retrieved"])
        prompt = prompts[cfg["prompt"]].format(context=context, question=d["question"])
        if hashlib.sha256(prompt.encode()).hexdigest() != d["prompt_sha256"]:
            print(f"Prompt reconstruit différent de l'original : {d['config']}/{d['case_id']}", file=sys.stderr)
            return 1
        sid = f"p{n:03d}"
        samples.append(
            {"id": sid, "question": d["question"], "context": context, "answer": d["answer"], "origin": "natural"}
        )
        key[sid] = {
            "config": d["config"],
            "config_name": cfg["name"],
            "case_id": d["case_id"],
            "run_id": d["run_id"],
            "prompt_sha256": d["prompt_sha256"],
            "retrieved_ids": [c["id"] for c in d["retrieved"]],
        }

    write_jsonl(args.out, samples)
    note = "Do not open before labeling is finished: maps opaque ids to config and question."
    Path(args.key_out).write_text(
        json.dumps(
            {"_note": note, "run_id": manifest["run_id"], "source_commit": "908ff6b", "ids": key},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"{len(samples)} échantillons écrits dans {args.out}, prompts vérifiés ; correspondance dans {args.key_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
