"""Validate benchmark question files against the schema and the frozen corpora.

Checks, for every case in the given YAML files (default: all in this
directory):

1. JSON-Schema conformance (schemas/case.schema.json);
2. unique ids;
3. every supporting chunk's file exists in the frozen corpus copy
   (anchors are informative and not resolved);
4. contradiction pairs reference pair ids declared in the corpus manifest;
5. per-type and per-corpus counts, printed for review.

Usage:
    uv run --with jsonschema python benchmark/questions/validate.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BENCH = HERE.parent


def load_cases(paths: list[Path]) -> list[dict]:
    """All cases from the given question files."""
    cases: list[dict] = []
    for path in paths:
        data = yaml.safe_load(path.read_text())
        cases.extend(data["cases"])
    return cases


def main() -> int:
    """Validate question files; return non-zero on any failure."""
    import jsonschema

    schema = json.loads((BENCH / "schemas" / "case.schema.json").read_text())
    files = [Path(a) for a in sys.argv[1:]] or sorted(HERE.glob("*.yaml"))
    cases = load_cases(files)
    errors: list[str] = []

    # 1. Schema conformance
    validator = jsonschema.Draft202012Validator(schema)
    for case in cases:
        for err in validator.iter_errors(case):
            errors.append(f"{case.get('id', '<no id>')}: {err.message}")

    # 2. Unique ids
    ids = Counter(c["id"] for c in cases)
    errors.extend(f"duplicate id: {i}" for i, n in ids.items() if n > 1)

    # 3. Supporting chunk files exist in the frozen corpora
    for case in cases:
        for chunk in case.get("supporting_chunks", []):
            rel = chunk.split("#", 1)[0]
            f = BENCH / "corpora" / case["corpus"] / "files" / rel
            if not f.is_file():
                errors.append(f"{case['id']}: missing corpus file {case['corpus']}/files/{rel}")

    # 4. Contradiction pairs declared in the manifest
    for case in cases:
        pair = case.get("contradiction_pair")
        if not pair:
            continue
        manifest = yaml.safe_load((BENCH / "corpora" / case["corpus"] / "MANIFEST.yaml").read_text())
        declared = {p["id"] for p in manifest.get("contradiction_pairs", [])}
        if pair not in declared:
            errors.append(f"{case['id']}: pair {pair} not in {case['corpus']} manifest ({sorted(declared)})")

    # 5. Counts
    by_type = Counter(c["question_type"] for c in cases)
    by_corpus = Counter(c["corpus"] for c in cases)
    print(f"{len(cases)} cases from {len(files)} file(s)")
    print("by type:  ", dict(sorted(by_type.items())))
    print("by corpus:", dict(sorted(by_corpus.items())))

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
