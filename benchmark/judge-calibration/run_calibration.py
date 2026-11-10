#!/usr/bin/env python3
"""Measure judge-vs-human agreement for one or more LLM judges.

Differences with ragnarok_ai.JudgeCalibrator, on purpose:
  * dev/test split: the pass threshold is tuned on 'dev' and measured on
    'test', so the reported agreement is not optimistic;
  * deterministic local calls (temperature 0, fixed seed, explicit num_ctx);
  * parse failures (an EvaluationError from the judge, ragnarok-ai >= 1.12.2),
    possible context truncation, latency and run-to-run stability are
    recorded instead of being silently absorbed;
  * every judge call is checkpointed: interrupt and rerun to resume.

Usage:
    python run_calibration.py --samples samples.jsonl --labels labels.jsonl --out results \\
        --judge prometheus=hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M \\
        --judge qwen=qwen2.5:7b-instruct \\
        --criteria faithfulness relevance

    # stability: 3 runs on 20 test samples
    python run_calibration.py ... --only-split test --limit 20 --repeats 3

    # recompute the report without calling any judge
    python run_calibration.py ... --analyze-only \\
        --second-pass labels_pass2.jsonl [--second-pass labels_other.jsonl]

Judge spec: NAME=MODEL (local Ollama) or NAME=openai:MODEL (remote API,
sends data outside the machine: only use it on public data).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from common import CRITERIA, append_jsonl, load_labels, load_samples, read_jsonl, split_of, write_jsonl

from ragnarok_ai.core.exceptions import EvaluationError

CANDIDATE_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)  # judge scores are (raw-1)/4 with raw in 1..5
RAW_KEEP_CHARS = 4000


# ── LLM clients ──────────────────────────────────────────────────────────


class OllamaJudgeLLM:
    """Minimal Ollama client with deterministic options (LLMProtocol-compatible)."""

    is_local = True

    def __init__(self, model: str, base_url: str, temperature: float, seed: int, num_ctx: int, timeout: float):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self.timeout = timeout
        self.last_meta: dict[str, Any] = {}

    async def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": self.options}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        prompt_tokens = data.get("prompt_eval_count")
        self.last_meta = {
            "prompt_tokens": prompt_tokens,
            "possibly_truncated": bool(prompt_tokens and prompt_tokens >= self.options["num_ctx"] - 8),
        }
        return data.get("response", "")

    async def embed(self, text: str) -> list[float]:  # pragma: no cover - unused by the judge
        raise NotImplementedError


class Recorder:
    """Wrap any LLMProtocol to capture the raw response and latency."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.is_local = getattr(llm, "is_local", False)
        self.model = getattr(llm, "model", "custom")
        self.last: dict[str, Any] = {}

    async def generate(self, prompt: str) -> str:
        t0 = time.perf_counter()
        out = await self.llm.generate(prompt)
        self.last = {"raw": out, "latency_s": time.perf_counter() - t0, **getattr(self.llm, "last_meta", {})}
        return out

    async def embed(self, text: str) -> list[float]:  # pragma: no cover
        return await self.llm.embed(text)


def build_judges(specs: list[str], args: argparse.Namespace) -> list[tuple[str, Recorder]]:
    judges = []
    for spec in specs:
        if "=" not in spec:
            msg = f"Invalid --judge '{spec}', expected NAME=MODEL"
            raise SystemExit(msg)
        name, model = spec.split("=", 1)
        if model.startswith("openai:"):
            from ragnarok_ai.adapters.llm.openai import OpenAILLM

            llm: Any = OpenAILLM(model=model.removeprefix("openai:"))
            print(f"ATTENTION : le juge '{name}' est distant, les échantillons quittent la machine.")
        else:
            model = model.removeprefix("ollama:")
            llm = OllamaJudgeLLM(model, args.base_url, args.temperature, args.seed, args.num_ctx, args.timeout)
        judges.append((name, Recorder(llm)))
    return judges


# ── Judging ──────────────────────────────────────────────────────────────


async def judge_once(judge: Any, criterion: str, s: dict[str, Any]) -> Any:
    if criterion == "faithfulness":
        return await judge.evaluate_faithfulness(context=s["context"], question=s["question"], answer=s["answer"])
    if criterion == "relevance":
        return await judge.evaluate_relevance(question=s["question"], answer=s["answer"])
    if criterion == "hallucination":
        return await judge.detect_hallucination(context=s["context"], answer=s["answer"])
    return await judge.evaluate_completeness(question=s["question"], answer=s["answer"], context=s["context"])


async def run_judges(args: argparse.Namespace, samples: dict, labels: dict, raw_path: Path) -> None:
    from ragnarok_ai import LLMJudge

    selected = sorted(sid for sid in samples if sid in labels)
    if args.only_split:
        selected = [sid for sid in selected if split_of(sid, args.dev_fraction) == args.only_split]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("Aucun échantillon annoté à évaluer.")

    # A parse failure is final (deterministic judge); only transport errors are retried
    done = {
        (r["judge"], r["criterion"], r["id"], r["repeat"])
        for r in read_jsonl(raw_path)
        if not r.get("error") or r.get("parse_error")
    }
    judges = build_judges(args.judge, args)
    plan = [
        (name, rec, c, rep, sid)
        for name, rec in judges
        for c in args.criteria
        for rep in range(args.repeats)
        for sid in selected
        if (name, c, sid, rep) not in done
    ]
    print(f"{len(selected)} échantillons, {len(plan)} appels de juge à faire ({len(done)} déjà en cache).")

    judge_objs = {name: LLMJudge(llm=rec) for name, rec in judges}
    t_start = time.perf_counter()
    for i, (name, rec, criterion, rep, sid) in enumerate(plan, start=1):
        s = samples[sid]
        row: dict[str, Any] = {
            "judge": name,
            "model": rec.model,
            "is_local": rec.is_local,
            "criterion": criterion,
            "id": sid,
            "repeat": rep,
        }
        result = None
        for attempt in range(2):
            try:
                result = await judge_once(judge_objs[name], criterion, s)
                break
            except EvaluationError as exc:
                row.update(
                    {
                        "error": f"parse: {exc}",
                        "parse_error": True,
                        "latency_s": round(rec.last.get("latency_s", 0.0), 3),
                        "raw": rec.last.get("raw", "")[:RAW_KEEP_CHARS],
                    }
                )
                break
            except (httpx.HTTPError, OSError) as exc:
                if attempt == 1:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, httpx.ConnectError):
                        append_jsonl(raw_path, row)
                        raise SystemExit(f"Serveur injoignable ({exc}). Relancez pour reprendre.") from exc
        if result is not None:
            raw = rec.last.get("raw", "")
            row.update(
                {
                    "raw_score": result.raw_score,
                    "score": result.score,
                    "parse_ok": True,
                    "latency_s": round(rec.last.get("latency_s", 0.0), 3),
                    "prompt_tokens": rec.last.get("prompt_tokens"),
                    "possibly_truncated": rec.last.get("possibly_truncated", False),
                    "explanation": result.explanation[:1500],
                    "raw": raw[:RAW_KEEP_CHARS],
                }
            )
        append_jsonl(raw_path, row)

        if i == 1 or i % 10 == 0 or i == len(plan):
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (len(plan) - i)
            print(f"  {i}/{len(plan)}  écoulé {elapsed / 60:.1f} min, restant ~{eta / 60:.1f} min", flush=True)


# ── Statistics ───────────────────────────────────────────────────────────


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - m), min(1.0, c + m))


def kappa_band(k: float) -> str:
    if math.isnan(k):
        return "n/a"
    for floor, label in (
        (0.8, "quasi parfait"),
        (0.6, "substantiel"),
        (0.4, "modéré"),
        (0.2, "faible"),
        (0.0, "léger"),
    ):
        if k >= floor:
            return label
    return "pire que le hasard"


def agreement(scores: list[float], human: list[bool], threshold: float) -> dict[str, Any]:
    judge = [s >= threshold for s in scores]
    n = len(judge)
    agree = sum(j == h for j, h in zip(judge, human, strict=True))
    fails = sum(not h for h in human)
    passes = n - fails
    fa = sum(j and not h for j, h in zip(judge, human, strict=True))
    fr = sum((not j) and h for j, h in zip(judge, human, strict=True))
    # Kappa is meaningless when the human labels are all identical (e.g. a subset of degraded samples)
    k = kappa(judge, human) if 0 < fails < n else float("nan")
    return {
        "n": n,
        "threshold": threshold,
        "accuracy": agree / n if n else float("nan"),
        "accuracy_ci": wilson(agree, n),
        "kappa": k,
        "kappa_band": kappa_band(k),
        "false_accept_rate": fa / fails if fails else None,
        "false_reject_rate": fr / passes if passes else None,
        "n_human_fail": fails,
    }


def best_threshold(scores: list[float], human: list[bool], default: float) -> float:
    best_t, best_k = default, kappa([s >= default for s in scores], human)
    for t in CANDIDATE_THRESHOLDS:
        k = kappa([s >= t for s in scores], human)
        if k > best_k + 1e-9 or (abs(k - best_k) <= 1e-9 and abs(t - default) < abs(best_t - default)):
            best_t, best_k = t, k
    return best_t


def analyze(args: argparse.Namespace, samples: dict, labels: dict, raw_path: Path) -> dict[str, Any]:
    rows = [r for r in read_jsonl(raw_path) if r["id"] in samples]
    by_jc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_jc[(r["judge"], r["criterion"])].append(r)

    report: dict[str, Any] = {"judges": {}, "settings": vars(args).copy()}
    report["settings"].pop("func", None)
    disagreements: list[dict[str, Any]] = []

    for (judge, criterion), jrows in sorted(by_jc.items()):
        ok = [r for r in jrows if not r.get("error") and r.get("score") is not None]
        resolved = {(r["id"], r["repeat"]) for r in ok}
        jrows = [r for r in jrows if not (r.get("error") and (r["id"], r["repeat"]) in resolved)]
        first = {r["id"]: r for r in ok if r["repeat"] == 0 and criterion in labels.get(r["id"], {})}

        def subset(
            split: str, first: dict = first, criterion: str = criterion
        ) -> tuple[list[str], list[float], list[bool]]:
            ids = sorted(i for i in first if split_of(i, args.dev_fraction) == split)
            return ids, [first[i]["score"] for i in ids], [labels[i][criterion] for i in ids]

        dev_ids, dev_s, dev_h = subset("dev")
        test_ids, test_s, test_h = subset("test")
        tuned = best_threshold(dev_s, dev_h, args.threshold) if dev_ids else args.threshold

        # Second view: an unreadable verdict counted as the judge saying FAIL (what a CI gate does)
        unreadable = sorted(
            r["id"]
            for r in jrows
            if r.get("parse_error")
            and r["repeat"] == 0
            and r["id"] not in first
            and criterion in labels.get(r["id"], {})
            and split_of(r["id"], args.dev_fraction) == "test"
        )
        fail_s = test_s + [0.0] * len(unreadable)  # 0.0 is below every threshold
        fail_h = test_h + [labels[i][criterion] for i in unreadable]

        by_origin: dict[str, dict[str, Any]] = {}
        for origin in sorted({samples[i]["origin"] for i in test_ids}):
            idx = [k for k, i in enumerate(test_ids) if samples[i]["origin"] == origin]
            by_origin[origin] = agreement([test_s[k] for k in idx], [test_h[k] for k in idx], args.threshold)

        # Exploratory Q2: does the judge rank configurations like the human does?
        # Pass rates over all labeled samples (dev + test), default threshold.
        pass_rates: dict[str, dict[str, Any]] = {}
        for origin in sorted({samples[i]["origin"] for i in first}):
            ids = [i for i in first if samples[i]["origin"] == origin]
            scores = [first[i]["score"] for i in ids]
            pass_rates[origin] = {
                "n": len(ids),
                "human_pass_rate": sum(labels[i][criterion] for i in ids) / len(ids),
                "judge_pass_rate": sum(s >= args.threshold for s in scores) / len(ids),
                "judge_mean_score": statistics.mean(scores),
            }

        groups: dict[str, list[int]] = defaultdict(list)
        for r in ok:
            groups[r["id"]].append(r["raw_score"])
        multi = [v for v in groups.values() if len(v) >= 2]

        lat = [r["latency_s"] for r in ok if r.get("latency_s")]
        n_parse = sum(1 for r in jrows if r.get("parse_error"))
        entry = {
            "model": jrows[0].get("model"),
            "is_local": jrows[0].get("is_local"),
            "calls": len(jrows),
            "errors": sum(1 for r in jrows if r.get("error") and not r.get("parse_error")),
            "parse_failures": n_parse,
            "parse_failure_rate": (n_parse / (len(ok) + n_parse)) if (ok or n_parse) else None,
            "possibly_truncated": sum(1 for r in ok if r.get("possibly_truncated")),
            "latency_median_s": statistics.median(lat) if lat else None,
            "latency_total_min": sum(lat) / 60 if lat else None,
            "stability": {
                "n_samples": len(multi),
                "identical_rate": (sum(1 for v in multi if len(set(v)) == 1) / len(multi)) if multi else None,
            },
            "n_dev": len(dev_ids),
            "test_default": agreement(test_s, test_h, args.threshold) if test_ids else None,
            "test_parse_failures": len(unreadable),
            "test_parse_as_fail": agreement(fail_s, fail_h, args.threshold) if fail_h else None,
            "tuned_threshold": tuned,
            "test_tuned": agreement(test_s, test_h, tuned) if test_ids else None,
            "always_pass_accuracy": (sum(test_h) / len(test_h)) if test_h else None,
            "test_by_origin": by_origin,
            "pass_rates_by_origin": pass_rates,
        }
        report["judges"].setdefault(judge, {})[criterion] = entry

        for i in test_ids:
            r = first[i]
            if (r["score"] >= args.threshold) != labels[i][criterion]:
                disagreements.append(
                    {
                        "judge": judge,
                        "criterion": criterion,
                        "id": i,
                        "origin": samples[i]["origin"],
                        "human_pass": labels[i][criterion],
                        "judge_raw_score": r["raw_score"],
                        "parse_ok": r.get("parse_ok"),
                        "question": samples[i]["question"],
                        "answer": samples[i]["answer"],
                        "judge_explanation": r.get("explanation"),
                    }
                )

    # Agreement between the main labels and each other labeling (own second pass or another annotator)
    report["label_agreement"] = {}
    for path in args.second_pass:
        other = load_labels(path)
        per_criterion = {}
        for c in args.criteria:
            ids = [i for i in other if c in other[i] and c in labels.get(i, {})]
            a, b = [labels[i][c] for i in ids], [other[i][c] for i in ids]
            if ids:
                per_criterion[c] = {
                    "n": len(ids),
                    "accuracy": sum(x == y for x, y in zip(a, b, strict=True)) / len(ids),
                    "kappa": kappa(a, b),
                }
        report["label_agreement"][Path(path).name] = per_criterion

    out = Path(args.out)
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_jsonl(out / "disagreements.jsonl", disagreements)
    (out / "summary.md").write_text(render_markdown(report, args), encoding="utf-8")
    return report


# ── Report ───────────────────────────────────────────────────────────────


def pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def num(x: float | None, fmt: str = ".2f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def render_markdown(report: dict[str, Any], args: argparse.Namespace) -> str:
    lines = ["# Calibration du juge : résultats", ""]
    lines.append(
        f"Seuil par défaut {args.threshold} (score Prometheus ≥ 4/5). Seuil optimisé sur le split dev, "
        f"mesuré sur le split test. Température {args.temperature}, seed {args.seed}, num_ctx {args.num_ctx}."
    )
    for criterion in args.criteria:
        rows = [(j, c[criterion]) for j, c in report["judges"].items() if criterion in c]
        if not rows:
            continue
        lines += ["", f"## {criterion}", "", "Échecs de lecture exclus :", ""]
        lines.append(
            "| Juge | n test | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés | Seuil optimisé (dev) | Kappa test au seuil optimisé |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for judge, e in rows:
            d, t = e["test_default"], e["test_tuned"]
            if not d:
                continue
            ci = d["accuracy_ci"]
            lines.append(
                f"| {judge} | {d['n']} | {pct(d['accuracy'])} ({pct(ci[0])}-{pct(ci[1])}) | "
                f"{num(d['kappa'])} ({d['kappa_band']}) | {pct(d['false_accept_rate'])} | "
                f"{pct(d['false_reject_rate'])} | {e['tuned_threshold']} | {num(t['kappa'])} |"
            )
        lines += [
            "",
            "Échecs de lecture comptés comme FAIL (seuil par défaut) :",
            "",
            "| Juge | n test | dont échecs de lecture | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés |",
            "|---|---|---|---|---|---|---|",
        ]
        for judge, e in rows:
            f = e["test_parse_as_fail"]
            if not f:
                continue
            ci = f["accuracy_ci"]
            lines.append(
                f"| {judge} | {f['n']} | {e['test_parse_failures']} | {pct(f['accuracy'])} ({pct(ci[0])}-{pct(ci[1])}) | "
                f"{num(f['kappa'])} ({f['kappa_band']}) | {pct(f['false_accept_rate'])} | {pct(f['false_reject_rate'])} |"
            )
        ref = rows[0][1]["always_pass_accuracy"]
        lines += [
            "",
            f"Référence : un juge qui répond toujours PASS obtiendrait {pct(ref)} d'accord sur ce split test.",
        ]
        lines += [
            "",
            "Par origine (seuil par défaut) :",
            "",
            "| Juge | Origine | n | Accord | Kappa |",
            "|---|---|---|---|---|",
        ]
        for judge, e in rows:
            for origin, o in e["test_by_origin"].items():
                lines.append(f"| {judge} | {origin} | {o['n']} | {pct(o['accuracy'])} | {num(o['kappa'])} |")
        lines += [
            "",
            "Analyse exploratoire Q2, taux de PASS par origine (tous splits, seuil par défaut) :",
            "",
            "| Juge | Origine | n | PASS humain | PASS juge | Score moyen du juge |",
            "|---|---|---|---|---|---|",
        ]
        for judge, e in rows:
            for origin, r in e["pass_rates_by_origin"].items():
                lines.append(
                    f"| {judge} | {origin} | {r['n']} | {pct(r['human_pass_rate'])} | "
                    f"{pct(r['judge_pass_rate'])} | {num(r['judge_mean_score'])} |"
                )

    lines += ["", "## Fonctionnement", ""]
    lines.append(
        "| Juge | Critère | Local | Appels | Erreurs | Échecs de parsing | Contexte possiblement tronqué | Latence médiane | Temps total | Stabilité (n) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for judge, crit in report["judges"].items():
        for criterion, e in crit.items():
            st = e["stability"]
            lines.append(
                f"| {judge} | {criterion} | {'oui' if e['is_local'] else 'non'} | {e['calls']} | {e['errors']} | "
                f"{pct(e['parse_failure_rate'])} | {e['possibly_truncated']} | {num(e['latency_median_s'], '.1f')} s | "
                f"{num(e['latency_total_min'], '.0f')} min | {pct(st['identical_rate'])} ({st['n_samples']}) |"
            )

    if report.get("label_agreement"):
        lines += [
            "",
            "## Accord entre annotations humaines (plafond atteignable par un juge)",
            "",
            "| Comparaison avec | Critère | n | Accord | Kappa |",
            "|---|---|---|---|---|",
        ]
        for name, per_c in report["label_agreement"].items():
            for c, v in per_c.items():
                lines.append(f"| {name} | {c} | {v['n']} | {pct(v['accuracy'])} | {num(v['kappa'])} |")

    small = [
        f"{j}/{c}"
        for j, cr in report["judges"].items()
        for c, e in cr.items()
        if e["test_default"] and e["test_default"]["n"] < 30
    ]
    if small:
        lines += ["", f"Échantillon test < 30 pour : {', '.join(small)}. Chiffres indicatifs, voir la largeur des IC."]
    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--judge", action="append", default=[], help="NAME=MODEL or NAME=openai:MODEL (repeatable)")
    ap.add_argument("--criteria", nargs="+", default=["faithfulness", "relevance"], choices=CRITERIA)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, help="judge only the first N labeled samples (by id)")
    ap.add_argument("--only-split", choices=["dev", "test"])
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument(
        "--second-pass",
        action="append",
        default=[],
        help="other labels JSONL (own second pass or another annotator), repeatable",
    )
    args = ap.parse_args()

    samples = load_samples(args.samples)
    labels = load_labels(args.labels)
    raw_path = Path(args.out) / "raw_judgments.jsonl"

    if not args.analyze_only:
        if not args.judge:
            ap.error("au moins un --judge est requis (ou --analyze-only)")
        asyncio.run(run_judges(args, samples, labels, raw_path))

    analyze(args, samples, labels, raw_path)
    print(f"\nRapport : {Path(args.out) / 'summary.md'}")
    print(f"Désaccords à relire : {Path(args.out) / 'disagreements.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
