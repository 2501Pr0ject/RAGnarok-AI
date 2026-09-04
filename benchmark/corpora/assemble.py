"""Assemble the pinned study corpora from corpora.yaml.

Reproducible corpus assembly for the RAGnarok evaluation study
(see ../PROTOCOL.md §2). For each corpus this tool:

1. clones the source repository (shallow, blob-filtered, sparse) at the
   pinned commit;
2. copies the files selected by the include globs into
   ``<name>/files/``;
3. fetches each contradiction pair's file at its old ref, stores it as
   ``<path>@<old_ref>``, and fails if the two versions are identical;
4. extracts the upstream LICENSE next to the files;
5. writes ``MANIFEST.yaml``: source pin, license, rationale, selection,
   per-file SHA-256 + sizes, and a deterministic corpus-level SHA-256.

Usage:
    python benchmark/corpora/assemble.py            # all corpora
    python benchmark/corpora/assemble.py docker     # one corpus

Requires: git, network access, pyyaml. Re-running is deterministic for
a given corpora.yaml (commits are pinned).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
CONFIG = HERE / "corpora.yaml"


def run(args: list[str], cwd: Path | None = None) -> str:
    """Run a command, returning stdout; raise with stderr on failure."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = f"{' '.join(args)} failed: {result.stderr.strip()}"
        raise RuntimeError(msg)
    return result.stdout


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def safe_ref(ref: str) -> str:
    """Filesystem-safe form of a git ref for pair filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", ref)[:24]


def clone_pinned(spec: dict[str, Any], workdir: Path) -> Path:
    """Sparse-clone the corpus source at its pinned commit."""
    repo = workdir / spec["name"]
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            *(["--branch", spec["ref"]] if spec.get("ref") else []),
            spec["source_url"],
            str(repo),
        ]
    )
    head = run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    pinned = spec["pinned_commit"]
    if head != pinned:
        # The default branch moved since pinning: fetch the exact commit.
        run(["git", "fetch", "--depth", "1", "origin", pinned], cwd=repo)
        run(["git", "checkout", pinned], cwd=repo)

    sparse_dirs = sorted({glob.split("*", 1)[0].rstrip("/") for glob in spec["include"]})
    run(["git", "sparse-checkout", "set", *sparse_dirs], cwd=repo)
    return repo


def matches(path: str, glob: str) -> bool:
    """fnmatch with directory-recursive `**` semantics.

    ``dir/**/*.md`` matches files directly in ``dir/`` as well as at any
    depth below it (plain fnmatch would require at least one subdir).
    """
    return fnmatch(path, glob) or ("/**/" in glob and fnmatch(path, glob.replace("/**/", "/")))


def select_files(repo: Path, include: list[str], exclude: list[dict[str, str]]) -> list[Path]:
    """Repository-relative paths matching include globs minus documented excludes."""
    tracked = run(["git", "ls-files"], cwd=repo).splitlines()
    selected = [
        Path(p)
        for p in tracked
        if any(matches(p, glob) for glob in include) and not any(matches(p, e["glob"]) for e in exclude)
    ]
    if not selected:
        msg = f"No files matched include globs in {repo}"
        raise RuntimeError(msg)
    return sorted(selected)


def fetch_at_ref(repo: Path, ref: str, path: str) -> bytes:
    """Content of *path* at *ref* (fetched shallowly on demand)."""
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=repo)
    out = subprocess.run(
        ["git", "show", f"FETCH_HEAD:{path}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if out.returncode != 0:
        msg = f"{path} does not exist at ref {ref}: {out.stderr.decode().strip()}"
        raise RuntimeError(msg)
    return out.stdout


def assemble(spec: dict[str, Any], workdir: Path) -> None:
    """Assemble one corpus into benchmark/corpora/<name>/."""
    name = spec["name"]
    dest = HERE / name
    files_dir = dest / "files"
    print(f"[{name}] cloning {spec['source_url']} @ {spec['pinned_commit'][:12]}")
    repo = clone_pinned(spec, workdir)

    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True)

    strip = spec.get("strip_prefix", "")
    entries: list[dict[str, Any]] = []
    for rel in select_files(repo, spec["include"], spec.get("exclude", [])):
        data = (repo / rel).read_bytes()
        out_rel = str(rel)[len(strip) :] if str(rel).startswith(strip) else str(rel)
        out_path = files_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        entries.append({"path": out_rel, "sha256": sha256_bytes(data), "bytes": len(data)})
    print(f"[{name}] {len(entries)} files selected")

    pair_records = []
    for pair in spec.get("pairs", []):
        current = (repo / pair["path"]).read_bytes()
        old = fetch_at_ref(repo, pair["old_ref"], pair["path"])
        if sha256_bytes(old) == sha256_bytes(current):
            msg = f"[{name}] pair {pair['id']}: identical at {pair['old_ref']} and pinned commit — not a contradiction"
            raise RuntimeError(msg)
        rel = pair["path"]
        rel = rel[len(strip) :] if rel.startswith(strip) else rel
        rel_path = Path(rel)
        old_rel = str(rel_path.with_name(f"{rel_path.stem}@{safe_ref(pair['old_ref'])}{rel_path.suffix}"))
        old_path = files_dir / old_rel
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_bytes(old)
        entries.append({"path": old_rel, "sha256": sha256_bytes(old), "bytes": len(old)})
        pair_records.append(
            {
                "id": pair["id"],
                "files": [rel, old_rel],
                "old_ref": pair["old_ref"],
                "nature": pair["nature"],
            }
        )
        print(f"[{name}] pair {pair['id']}: versions differ ✓")

    license_data = fetch_at_ref(repo, spec["pinned_commit"], spec["license"]["path_in_repo"])
    (dest / "LICENSE").write_bytes(license_data)

    entries.sort(key=lambda e: e["path"])
    corpus_sha = sha256_bytes("\n".join(f"{e['sha256']}  {e['path']}" for e in entries).encode())

    manifest = {
        "name": name,
        "version": f"{spec.get('ref') or 'main'} @ {spec['pinned_commit'][:12]}",
        "source_url": spec["source_url"],
        "source_commit": spec["pinned_commit"],
        "retrieval_date": datetime.now(timezone.utc).date().isoformat(),
        "retrieved_by": "assemble.py",
        "format": spec["format"],
        "license": spec["license"]["spdx"],
        "license_url": spec["license"]["url"],
        "license_file": "LICENSE",
        "attribution": spec["license"]["attribution"],
        "subset_rationale": spec["subset_rationale"].strip(),
        "selection": {
            "include": spec["include"],
            "exclude": spec.get("exclude", []),
            "strip_prefix": strip,
        },
        "file_count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "corpus_sha256": corpus_sha,
        "contradiction_pairs": pair_records,
        "files": entries,
    }
    (dest / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    print(f"[{name}] manifest written: {len(entries)} files, {manifest['total_bytes']} bytes, sha {corpus_sha[:12]}")


def main() -> int:
    """Assemble all corpora, or only those named on the command line."""
    config = yaml.safe_load(CONFIG.read_text())
    wanted = set(sys.argv[1:])
    specs = [s for s in config["corpora"] if not wanted or s["name"] in wanted]
    if not specs:
        print(f"No corpus matches {sorted(wanted)}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="ragnarok-corpora-") as tmp:
        for spec in specs:
            assemble(spec, Path(tmp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
