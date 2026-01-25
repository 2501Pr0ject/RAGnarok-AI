"""I/O utilities for test sets.

This module provides functions for saving and loading test sets.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragnarok_ai.core.types import Query, TestSet


def save_testset(testset: TestSet, path: Path | str) -> None:
    """Save a test set to a JSON file.

    Args:
        testset: The test set to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": testset.name,
        "description": testset.description,
        "metadata": testset.metadata,
        "queries": [
            {
                "text": q.text,
                "ground_truth_docs": q.ground_truth_docs,
                "expected_answer": q.expected_answer,
                "metadata": q.metadata,
            }
            for q in testset.queries
        ],
    }

    path.write_text(json.dumps(data, indent=2))


def load_testset(path: Path | str) -> TestSet:
    """Load a test set from a JSON file.

    Args:
        path: Input file path.

    Returns:
        Loaded TestSet.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Test set file not found: {path}"
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text())

    queries = [
        Query(
            text=q["text"],
            ground_truth_docs=q.get("ground_truth_docs", []),
            expected_answer=q.get("expected_answer"),
            metadata=q.get("metadata", {}),
        )
        for q in data.get("queries", [])
    ]

    return TestSet(
        queries=queries,
        name=data.get("name"),
        description=data.get("description"),
        metadata=data.get("metadata", {}),
    )
