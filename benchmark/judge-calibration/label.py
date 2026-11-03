#!/usr/bin/env python3
"""Blind human labeling of (question, context, answer) samples.

Samples are shown in a shuffled order and their origin is never displayed,
so the labeler cannot guess which ones were deliberately degraded.
Each verdict is written immediately: quit at any time and resume later.

Usage:
    python label.py samples.jsonl labels.jsonl --criteria faithfulness relevance

Second pass (intra-annotator agreement, at least 14 days after the first pass):
    python label.py samples.jsonl labels_pass2.jsonl --second-pass 0.2 --from-labels labels.jsonl

Another annotator on a subset: same command and same --seed, written to
another file. The subset is deterministic, so every annotator gets the
same samples (run it only once the main labeling is complete).
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
import time

from common import CRITERIA, GUIDELINES, GUIDELINES_VERSION, append_jsonl, load_labels, load_samples

ANSWERS = {"o": True, "y": True, "n": False}


def show_block(title: str, text: str, limit: int | None) -> None:
    body = text if limit is None or len(text) <= limit else text[:limit]
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    for para in body.splitlines() or [""]:
        print(textwrap.fill(para, width=100) if para.strip() else "")
    if limit is not None and len(text) > limit:
        print(f"  […] (+{len(text) - limit} caractères — tapez 'c' pour tout afficher)")


def ask(criterion: str, sample: dict) -> bool | str:
    """Return True/False, 'skip' or 'quit'."""
    while True:
        raw = input(f"\n[{criterion}] PASS ? (o/n, c=contexte complet, g=consigne, s=passer, q=quitter) > ")
        choice = raw.strip().lower()
        if choice in ANSWERS:
            return ANSWERS[choice]
        if choice == "c":
            show_block("CONTEXTE (complet)", sample["context"], None)
        elif choice == "g":
            print(f"  {GUIDELINES[criterion]}")
        elif choice == "s":
            return "skip"
        elif choice == "q":
            return "quit"
        else:
            print("  Réponse non reconnue.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", help="samples JSONL (id, question, context, answer[, origin])")
    ap.add_argument("labels_out", help="labels JSONL to append to")
    ap.add_argument("--criteria", nargs="+", default=["faithfulness", "relevance"], choices=CRITERIA)
    ap.add_argument("--context-chars", type=int, default=1500, help="context preview length")
    ap.add_argument("--seed", type=int, default=13, help="shuffle seed")
    ap.add_argument("--second-pass", type=float, help="fraction of already-labeled samples to relabel")
    ap.add_argument("--from-labels", help="first-pass labels (required with --second-pass)")
    ap.add_argument("--break-minutes", type=int, default=50, help="suggest a break after this many minutes")
    args = ap.parse_args()

    samples = load_samples(args.samples)
    done = set(load_labels(args.labels_out))
    rng = random.Random(args.seed)

    if args.second_pass is not None:
        if not args.from_labels:
            ap.error("--second-pass requires --from-labels")
        first = sorted(load_labels(args.from_labels))
        k = max(1, round(len(first) * args.second_pass))
        ids = random.Random(args.seed + 1).sample(first, min(k, len(first)))
    else:
        ids = sorted(samples)
    rng.shuffle(ids)
    todo = [sid for sid in ids if sid not in done and sid in samples]

    print(f"Consignes v{GUIDELINES_VERSION} :")
    for c in args.criteria:
        print(f"  - {c} : {GUIDELINES[c]}")
    print(f"\n{len(done)} déjà annotés, {len(todo)} restants.")

    session_start = time.monotonic()
    next_break = session_start + args.break_minutes * 60
    for i, sid in enumerate(todo, start=1):
        if time.monotonic() >= next_break:
            print(
                f"\n[PAUSE] {args.break_minutes} min d'annotation : faites une pause (q pour quitter, reprise automatique)."
            )
            next_break = time.monotonic() + args.break_minutes * 60
        t0 = time.monotonic()
        s = samples[sid]
        print("\n" + "═" * 100)
        print(f"Échantillon {i}/{len(todo)}  (id {sid})")
        show_block("QUESTION", s["question"], None)
        show_block("CONTEXTE", s["context"], args.context_chars)
        show_block("RÉPONSE", s["answer"], None)

        labels: dict[str, bool] = {}
        for criterion in args.criteria:
            verdict = ask(criterion, s)
            if verdict == "quit":
                print("\nInterrompu. Reprise possible avec la même commande.")
                return 0
            if verdict == "skip":
                labels = {}
                break
            labels[criterion] = bool(verdict)
        if not labels:
            continue

        note = input("Note (facultative, Entrée pour ignorer) > ").strip()
        row = {
            "id": sid,
            "labels": labels,
            "guidelines_version": GUIDELINES_VERSION,
            "seconds": round(time.monotonic() - t0, 1),
        }
        if note:
            row["note"] = note
        append_jsonl(args.labels_out, row)

    print("\nTerminé.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nInterrompu. Reprise possible avec la même commande.")
        sys.exit(0)
