"""Produce the sentence-segmentation reference fixture from the external converter.

The converter that built the judge-training data is kept outside this
repository. It is loaded from the path given by the environment variable
``RAGNAROK_SEGMENTATION_REFERENCE`` (never copied, never recorded in the
fixture), run on the answers below, and its offsets and exclusion reasons
are written to ``segmentation_reference.json``:

    RAGNAROK_SEGMENTATION_REFERENCE=/path/to/converter.py \\
        python tests/unit/text/fixtures/make_segmentation_reference.py \\
        > tests/unit/text/fixtures/segmentation_reference.json

``tests/unit/text/test_segmentation.py`` then requires
``ragnarok_ai.text.segment_sentences`` to reproduce them exactly.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

# Answers written for this fixture: line breaks, citation markers, fragments,
# abbreviations, numbers, lists, punctuation-only lines, non-ASCII text.
CASES: dict[str, str] = {
    "plain": "Docker restarts containers according to their restart policy. The default policy is no.",
    "citation_merged": (
        "The default restart policy is no (Passage 3). Use --restart to change it [Source 1]. "
        "Containers keep their policy across daemon restarts."
    ),
    "citation_alone_on_its_line": "Passage 1:\nThe daemon reads its configuration file at startup.\nSource 2 and 3.",
    "fragments": "Sure!\nIngredients:\nStep 2.\nMix 200 g of flour with the eggs. Bake for 25 minutes.",
    "punctuation_only": "---\n***\nA real sentence follows the separators.\n...",
    "abbreviations_and_decimals": (
        "Dr. Smith measured 3.5 kg, i.e. about 7.7 lb. The U.S. standard differs, e.g. in labelling. "
        "See Fig. 2 for details."
    ),
    "list_items": "Three steps are required:\n1. Pull the image.\n2. Create the network.\n3. Start the container with --rm.",
    "blank_lines_and_spaces": "First paragraph.  \n\n\n   Second paragraph, indented.   \nThird one.\n",
    "questions_and_quotes": (
        'Why does it fail? The log says "connection refused". Restart the service! Then retry the request.'
    ),
    "unicode": "L'utilisateur crée un réseau nommé « interne ». Les conteneurs s'y attachent ensuite. Ça fonctionne.",
    "code_and_paths": (
        "Run `kubectl apply -f deploy.yaml` from the repository root. "
        "The file lives in ./manifests/prod/ and references v1.2.3 of the image."
    ),
    "single_fragment_only": "Thanks!",
    "two_word_fragment_with_digit": "Version 2",
    "long_sentence": (
        "When the daemon starts it loads the configuration file, validates every key against the schema, "
        "resolves the storage driver, opens the API socket, registers the plugins that were enabled at "
        "build time and finally reports its readiness to the init system, which is why a misconfigured "
        "key delays the whole sequence."
    ),
}


def load_reference(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("segmentation_reference", path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load the reference converter from {path}"
        raise SystemExit(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["segmentation_reference"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    path = os.environ.get("RAGNAROK_SEGMENTATION_REFERENCE")
    if not path:
        msg = "Set RAGNAROK_SEGMENTATION_REFERENCE to the path of the reference converter"
        raise SystemExit(msg)
    ref = load_reference(path)
    splitter = ref.SentenceSplitter("en")
    cases = []
    for name, text in CASES.items():
        segments = []
        for index, (start, end) in enumerate(splitter.split(text)):
            sentence = text[start:end]
            record = ref.SentenceRecord(
                id=f"{name}:{index}",
                source_id=name,
                response_id=name,
                task="QA",
                source_dataset="fixture",
                generator="fixture",
                split="test",
                sent_index=index,
                char_start=start,
                char_end=end,
                sentence=sentence,
                prev_sentence=None,
            )
            # No annotation span: only the three segment-level exclusions can fire
            ref.label_sentence(record, text, [], 3, 0.25, 3)
            segments.append({"start": start, "end": end, "text": sentence, "skipped": record.exclusion_reason})
        cases.append({"name": name, "text": text, "segments": segments})
    fixture = {
        "reference": {
            "description": "offsets and exclusion reasons produced by the RAGTruth conversion script (kept outside the repository)",
            "pysbd_version": ref.pysbd.__version__,
            "min_words": 3,
            "generator": "tests/unit/text/fixtures/make_segmentation_reference.py",
        },
        "cases": cases,
    }
    json.dump(fixture, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
