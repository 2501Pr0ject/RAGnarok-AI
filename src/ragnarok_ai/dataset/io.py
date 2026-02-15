"""Dataset I/O operations.

This module provides functions for loading and saving TestSet objects
from various file formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragnarok_ai.core.types import Query, TestSet


def load_testset(path: str | Path) -> TestSet:
    """Load a TestSet from a JSON or JSONL file.

    Supports:
    - JSON array of queries
    - JSON object with "queries" key
    - JSONL (one query per line)

    Args:
        path: Path to the dataset file.

    Returns:
        Loaded TestSet.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    content = path.read_text(encoding="utf-8")

    # Detect JSONL by trying to parse first line
    if path.suffix == ".jsonl" or _is_jsonl(content):
        return _load_jsonl(content, path.stem)

    # Parse as JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    return _parse_json_data(data, path.stem)


def _is_jsonl(content: str) -> bool:
    """Check if content is JSONL format (one JSON object per line)."""
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return False

    # Try to parse first two lines as separate JSON objects
    try:
        json.loads(lines[0])
        json.loads(lines[1])
        return True
    except json.JSONDecodeError:
        return False


def _load_jsonl(content: str, name: str) -> TestSet:
    """Load TestSet from JSONL content."""
    queries: list[Query] = []

    for i, line in enumerate(content.strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e

        queries.append(_dict_to_query(item, i))

    return TestSet(
        name=name,
        queries=queries,
    )


def _parse_json_data(data: Any, name: str) -> TestSet:
    """Parse JSON data into TestSet."""
    # Handle array of queries
    if isinstance(data, list):
        queries = [_dict_to_query(item, i + 1) for i, item in enumerate(data)]
        return TestSet(name=name, queries=queries)

    # Handle object with queries key
    if isinstance(data, dict):
        # Extract TestSet fields
        testset_name = data.get("name", name)
        description = data.get("description")
        schema_version = data.get("schema_version", 1)
        dataset_version = data.get("dataset_version", "1.0.0")
        author = data.get("author")
        source = data.get("source")
        metadata = data.get("metadata", {})

        # Parse queries
        queries_data = data.get("queries", [])
        if not isinstance(queries_data, list):
            raise ValueError("'queries' must be a list")

        queries = [_dict_to_query(item, i + 1) for i, item in enumerate(queries_data)]

        return TestSet(
            name=testset_name,
            description=description,
            queries=queries,
            schema_version=schema_version,
            dataset_version=dataset_version,
            author=author,
            source=source,
            metadata=metadata,
        )

    raise ValueError(f"Invalid dataset format: expected list or object, got {type(data).__name__}")


def _dict_to_query(item: Any, index: int) -> Query:
    """Convert a dict to a Query object."""
    if not isinstance(item, dict):
        raise ValueError(f"Query {index}: expected object, got {type(item).__name__}")

    # Extract text (support multiple field names)
    text = item.get("text") or item.get("question") or item.get("query")
    if not text:
        raise ValueError(f"Query {index}: missing 'text', 'question', or 'query' field")

    # Extract ground truth docs (support multiple field names)
    ground_truth = (
        item.get("ground_truth_docs")
        or item.get("ground_truth")
        or item.get("relevant_docs")
        or item.get("doc_ids")
        or []
    )

    # Ensure ground_truth is a list
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    # Extract expected answer
    expected_answer = item.get("expected_answer") or item.get("answer")

    # Extract metadata (exclude known fields)
    known_fields = {
        "text",
        "question",
        "query",
        "ground_truth_docs",
        "ground_truth",
        "relevant_docs",
        "doc_ids",
        "expected_answer",
        "answer",
        "id",
        "metadata",
    }
    metadata = item.get("metadata", {})

    # Add any extra fields to metadata
    for key, value in item.items():
        if key not in known_fields and key not in metadata:
            metadata[key] = value

    return Query(
        text=text,
        ground_truth_docs=ground_truth,
        expected_answer=expected_answer,
        metadata=metadata,
    )


def save_testset(testset: TestSet, path: str | Path) -> None:
    """Save a TestSet to a JSON file.

    Args:
        testset: TestSet to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": testset.name,
        "description": testset.description,
        "schema_version": testset.schema_version,
        "dataset_version": testset.dataset_version,
        "author": testset.author,
        "source": testset.source,
        "queries": [
            {
                "text": q.text,
                "ground_truth_docs": q.ground_truth_docs,
                "expected_answer": q.expected_answer,
                "metadata": q.metadata if q.metadata else None,
            }
            for q in testset.queries
        ],
        "metadata": testset.metadata if testset.metadata else None,
    }

    # Remove None values for cleaner output
    data = {k: v for k, v in data.items() if v is not None}
    for q in data.get("queries", []):
        if q.get("metadata") is None:
            del q["metadata"]
        if q.get("expected_answer") is None:
            del q["expected_answer"]

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
