"""Run the pilot: RAG configurations A-G over the pilot question set.

Experimental runner for the RAGnarok evaluation study (PROTOCOL.md §4).
It is deliberately self-contained study tooling — not part of the
framework — and produces a fully journaled run:

    benchmark/pilot-run/<run_id>/
    ├── run_manifest.json    # code version, model digests, hashes, configs
    ├── outputs/<CONFIG>/<case_id>.json
    └── logs/run.log

Before doing anything, the runner verifies the experimental locks:
corpus files must hash to the values frozen in each MANIFEST.yaml, and
the question set hash is recorded. On mismatch it refuses to run.

Determinism: pinned models (digests recorded), temperature and seed set
on generation, seeded RNG for the noisy/random retrieval defects, and
deterministic chunking. Embeddings are computed once per (embedding
model, chunking) combination and shared across configs.

Usage:
    python benchmark/run_pilot.py                  # full run, A-G x 30
    python benchmark/run_pilot.py --smoke          # 2 configs x 3 cases
    python benchmark/run_pilot.py --configs A B    # subset of configs

Requires a running Ollama with the models pinned in rag_configs/configs.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434"

log = logging.getLogger("pilot")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── Integrity locks ─────────────────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def verify_corpus(corpus_dir: Path) -> str:
    """Recompute a corpus hash and compare to its frozen manifest.

    Returns the corpus hash; raises if any file or the aggregate differs.
    """
    manifest = yaml.safe_load((corpus_dir / "MANIFEST.yaml").read_text())
    entries = []
    for entry in manifest["files"]:
        path = corpus_dir / "files" / entry["path"]
        if not path.is_file():
            msg = f"{corpus_dir.name}: missing frozen file {entry['path']}"
            raise RuntimeError(msg)
        digest = sha256_bytes(path.read_bytes())
        if digest != entry["sha256"]:
            msg = f"{corpus_dir.name}: {entry['path']} does not match its frozen hash"
            raise RuntimeError(msg)
        entries.append((entry["path"], digest))
    entries.sort()
    aggregate = sha256_bytes("\n".join(f"{d}  {p}" for p, d in entries).encode())
    if aggregate != manifest["corpus_sha256"]:
        msg = f"{corpus_dir.name}: corpus aggregate hash mismatch"
        raise RuntimeError(msg)
    return aggregate


# ── Chunking ────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """Deterministic character chunking with paragraph-boundary preference."""
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # Prefer to break at a blank line, else a newline, inside the window
            window = text[start:end]
            cut = window.rfind("\n\n")
            if cut < chunk_chars // 2:
                cut = window.rfind("\n")
            if cut >= chunk_chars // 2:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(corpora_dir: Path, chunk_chars: int, overlap: int) -> list[dict[str, str]]:
    """Chunk every frozen corpus file. Chunk ids are deterministic."""
    chunks: list[dict[str, str]] = []
    for corpus_dir in sorted(p for p in corpora_dir.iterdir() if (p / "files").is_dir()):
        for f in sorted((corpus_dir / "files").rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(corpus_dir / "files")
            for i, piece in enumerate(chunk_text(f.read_text(errors="replace"), chunk_chars, overlap)):
                chunks.append({"id": f"{corpus_dir.name}/{rel}::{i}", "text": piece})
    return chunks


# ── Ollama ──────────────────────────────────────────────────────────────


def model_digest(client: httpx.Client, model: str) -> str:
    """Digest of a local Ollama model (fails if the model is absent)."""
    r = client.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=30)
    r.raise_for_status()
    info = r.json()
    digest = info.get("details", {}).get("digest") or info.get("digest", "")
    if not digest:  # older ollama: fall back to the tags listing
        tags = client.get(f"{OLLAMA}/api/tags", timeout=30).json()["models"]
        digest = next(
            (m["digest"] for m in tags if m["name"].split(":latest")[0] in (model, model.split(":latest")[0])),
            "unknown",
        )
    return digest


EMBED_CACHE = HERE / "pilot-run" / "embed-cache.sqlite"


def embed_batch(client: httpx.Client, model: str, texts: list[str]) -> list[list[float]]:
    """Embed texts with an Ollama model, through a persistent cache.

    The cache (keyed by model + text hash) makes repeat runs cheap — the
    reproducibility requirement (PROTOCOL.md §7, two full runs) would
    otherwise pay the full embedding cost twice. Cached vectors are the
    model's deterministic output; caching does not alter the experiment.
    """
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(EMBED_CACHE)
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
    keys = [f"{model}:{sha256_bytes(t.encode())}" for t in texts]
    cached: dict[str, list[float]] = {}
    for i in range(0, len(keys), 500):
        batch = keys[i : i + 500]
        rows = db.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(batch))})", batch
        ).fetchall()
        cached.update({k: json.loads(v) for k, v in rows})

    missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in cached]
    for i in range(0, len(missing), 256):
        chunk = missing[i : i + 256]
        r = client.post(
            f"{OLLAMA}/api/embed",
            json={"model": model, "input": [t for _, t in chunk]},
            timeout=1200,
        )
        r.raise_for_status()
        vectors = r.json()["embeddings"]
        db.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(k, json.dumps(v)) for (k, _), v in zip(chunk, vectors, strict=True)],
        )
        db.commit()
        cached.update({k: v for (k, _), v in zip(chunk, vectors, strict=True)})
    db.close()
    return [cached[k] for k in keys]


def generate(client: httpx.Client, model: str, prompt: str, temperature: float, seed: int) -> str:
    """One deterministic-as-possible generation."""
    r = client.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed},
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["response"]


# ── Retrieval ───────────────────────────────────────────────────────────


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class Index:
    """In-memory embedding index over corpus chunks."""

    def __init__(self, chunks: list[dict[str, str]], vectors: list[list[float]]) -> None:
        self.chunks = chunks
        self.vectors = vectors

    def top_k(self, query_vec: list[float], k: int) -> list[tuple[dict[str, str], float]]:
        scored = [(chunk, cosine(query_vec, vec)) for chunk, vec in zip(self.chunks, self.vectors, strict=True)]
        scored.sort(key=lambda cs: -cs[1])
        return scored[:k]


def retrieve(
    index: Index,
    query_vec: list[float],
    cfg: dict[str, Any],
    rng: random.Random,
) -> list[tuple[dict[str, str], float]]:
    """Retrieval with the config's deliberate defect applied."""
    k = cfg["top_k"]
    if cfg["retrieval"] == "random":
        return [(c, 0.0) for c in rng.sample(index.chunks, k)]
    hits = index.top_k(query_vec, k)
    noise = cfg.get("noise_slots", 0)
    if noise:
        kept = hits[: k - noise]
        kept_ids = {c["id"] for c, _ in kept}
        pool = [c for c in index.chunks if c["id"] not in kept_ids]
        hits = kept + [(c, 0.0) for c in rng.sample(pool, noise)]
    return hits


# ── Runner ──────────────────────────────────────────────────────────────


def resolve_config(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Config = defaults overridden by the per-config block."""
    cfg = dict(defaults)
    cfg.update(raw)
    return cfg


def dry_run(
    spec: dict[str, Any],
    questions: list[dict[str, Any]],
    question_hash: str,
    wanted: list[str],
) -> int:
    """Verify every experimental lock and print the plan. Executes nothing."""
    print(f"{len(questions)} cases")
    print(f"{len(wanted)} configurations: {' '.join(wanted)}")
    print(f"{len(questions) * len(wanted)} evaluations planned")
    for letter in wanted:
        cfg = resolve_config(spec["configs"][letter], spec["defaults"])
        print(f"  {letter} ({cfg['name']}): {len(questions)} cases  [defect: {cfg['defect']}]")

    print()
    for d in sorted((HERE / "corpora").iterdir()):
        if (d / "MANIFEST.yaml").is_file():
            print(f"Corpus {d.name}: {verify_corpus(d)[:12]} OK")
    print(f"Question-set hash: {question_hash[:12]} OK")

    with httpx.Client() as client:
        models = {
            spec["defaults"]["generator_model"],
            *(resolve_config(c, spec["defaults"])["embedding_model"] for c in spec["configs"].values()),
        }
        for m in sorted(models):
            print(f"Model {m}: {model_digest(client, m)[:20]} OK")

    print()
    index_keys = {
        (cfg["embedding_model"], cfg["chunk_chars"], cfg["chunk_overlap"])
        for cfg in (resolve_config(spec["configs"][w], spec["defaults"]) for w in wanted)
    }
    for model, chars, overlap in sorted(index_keys):
        n = len(build_chunks(HERE / "corpora", chars, overlap))
        print(f"Index ({model}, {chars}/{overlap}): {n} chunks to embed")

    print()
    print("No evaluation executed.")
    return 0


def main() -> int:
    """Run the pilot and write the journaled outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", help="Subset of config letters (default: all)")
    parser.add_argument("--cases", type=int, default=0, help="Limit number of cases (0 = all)")
    parser.add_argument("--smoke", action="store_true", help="2 configs x 3 cases plumbing check")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify locks, digests and plan; execute nothing.",
    )
    args = parser.parse_args()

    spec = yaml.safe_load((HERE / "rag_configs" / "configs.yaml").read_text())
    questions = yaml.safe_load((HERE / "questions" / "pilot.yaml").read_text())["cases"]
    question_hash = sha256_bytes((HERE / "questions" / "pilot.yaml").read_bytes())

    wanted = args.configs or sorted(spec["configs"])
    if args.smoke:
        wanted, questions = ["A", "F"], questions[:3]
    elif args.cases:
        questions = questions[: args.cases]

    if args.dry_run:
        return dry_run(spec, questions, question_hash, wanted)

    run_id = datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%SZ")
    run_dir = HERE / "pilot-run" / run_id
    (run_dir / "logs").mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "logs" / "run.log")],
    )

    # ── Locks: corpus + questions integrity ─────────────────────────────
    log.info("verifying frozen corpora against manifests")
    corpus_hashes = {
        d.name: verify_corpus(d) for d in sorted((HERE / "corpora").iterdir()) if (d / "MANIFEST.yaml").is_file()
    }
    log.info("corpora verified: %s", {k: v[:12] for k, v in corpus_hashes.items()})

    client = httpx.Client()
    models = {
        spec["defaults"]["generator_model"],
        *(resolve_config(c, spec["defaults"])["embedding_model"] for c in spec["configs"].values()),
    }
    digests = {m: model_digest(client, m) for m in sorted(models)}
    log.info("model digests: %s", digests)

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=False
    ).stdout.strip()

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "platform": {
            "machine": platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "ollama_models": digests,
        "corpus_hashes": corpus_hashes,
        "question_set_hash": question_hash,
        "configs": {k: resolve_config(v, spec["defaults"]) for k, v in spec["configs"].items()},
        "configs_run": wanted,
        "cases_run": [c["id"] for c in questions],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Indexes: one per (embedding_model, chunking) combination ────────
    index_cache: dict[tuple[str, int, int], Index] = {}
    query_vec_cache: dict[tuple[str, str], list[float]] = {}

    def get_index(cfg: dict[str, Any]) -> Index:
        key = (cfg["embedding_model"], cfg["chunk_chars"], cfg["chunk_overlap"])
        if key not in index_cache:
            log.info("building index %s", key)
            chunks = build_chunks(HERE / "corpora", cfg["chunk_chars"], cfg["chunk_overlap"])
            t0 = time.perf_counter()
            vectors = embed_batch(client, cfg["embedding_model"], [c["text"] for c in chunks])
            log.info("embedded %d chunks in %.0fs", len(chunks), time.perf_counter() - t0)
            index_cache[key] = Index(chunks, vectors)
        return index_cache[key]

    def get_query_vec(model: str, question: str) -> list[float]:
        key = (model, question)
        if key not in query_vec_cache:
            query_vec_cache[key] = embed_batch(client, model, [question])[0]
        return query_vec_cache[key]

    # ── Execution ───────────────────────────────────────────────────────
    seed = spec["defaults"]["seed"]
    total = len(wanted) * len(questions)
    done = 0
    for letter in wanted:
        cfg = resolve_config(spec["configs"][letter], spec["defaults"])
        out_dir = run_dir / "outputs" / letter
        out_dir.mkdir(parents=True, exist_ok=True)
        index = get_index(cfg)
        prompt_template = spec["prompts"][cfg["prompt"]]

        for case in questions:
            rng = random.Random(f"{seed}:{letter}:{case['id']}")
            t0 = time.perf_counter()
            qvec = get_query_vec(cfg["embedding_model"], case["question"])
            hits = retrieve(index, qvec, cfg, rng)
            t_retrieve = time.perf_counter() - t0

            context = "\n\n---\n\n".join(c["text"] for c, _ in hits)
            prompt = prompt_template.format(context=context, question=case["question"])
            t1 = time.perf_counter()
            answer = generate(client, cfg["generator_model"], prompt, cfg["temperature"], seed)
            t_generate = time.perf_counter() - t1

            (out_dir / f"{case['id']}.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "config": letter,
                        "config_name": cfg["name"],
                        "case_id": case["id"],
                        "question": case["question"],
                        "retrieved": [{"id": c["id"], "score": round(s, 5), "text": c["text"]} for c, s in hits],
                        "answer": answer,
                        "prompt_sha256": sha256_bytes(prompt.encode()),
                        "timings_s": {"retrieve": round(t_retrieve, 2), "generate": round(t_generate, 2)},
                    },
                    indent=2,
                )
            )
            done += 1
            log.info("[%d/%d] %s %s (%.0fs)", done, total, letter, case["id"], t_retrieve + t_generate)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("run complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
