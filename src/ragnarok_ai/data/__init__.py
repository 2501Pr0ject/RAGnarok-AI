"""Example datasets for ragnarok-ai.

This module provides pre-packaged datasets for quick evaluation and testing.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from pydantic import BaseModel, Field

from ragnarok_ai.core.types import Document, Query, TestSet


class ExampleDataset(BaseModel):
    """An example dataset with documents and queries.

    Attributes:
        name: Dataset name.
        description: Dataset description.
        version: Dataset version.
        license: Dataset license.
        documents: List of documents.
        queries: List of queries with ground truth.
    """

    name: str = Field(..., description="Dataset name")
    description: str = Field(default="", description="Dataset description")
    version: str = Field(default="1.0", description="Dataset version")
    license: str = Field(default="MIT", description="Dataset license")
    documents: list[Document] = Field(default_factory=list, description="Documents")
    queries: list[Query] = Field(default_factory=list, description="Queries")

    def to_testset(self) -> TestSet:
        """Convert to a TestSet for evaluation.

        Returns:
            TestSet ready for use with evaluate().
        """
        return TestSet(
            queries=self.queries,
            name=self.name,
            metadata={
                "description": self.description,
                "version": self.version,
                "license": self.license,
            },
        )

    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_query(self, query_id: str) -> Query | None:
        """Get a query by ID.

        Args:
            query_id: Query ID.

        Returns:
            Query if found, None otherwise.
        """
        for query in self.queries:
            if query.metadata.get("id") == query_id:
                return query
        return None

    def filter_by_category(self, category: str) -> list[Query]:
        """Filter queries by category.

        Args:
            category: Category to filter by.

        Returns:
            List of queries matching the category.
        """
        return [q for q in self.queries if q.metadata.get("category") == category]

    def filter_by_difficulty(self, difficulty: str) -> list[Query]:
        """Filter queries by difficulty.

        Args:
            difficulty: Difficulty level (easy, medium, hard).

        Returns:
            List of queries matching the difficulty.
        """
        return [q for q in self.queries if q.metadata.get("difficulty") == difficulty]


def _load_raw_dataset(filename: str) -> dict[str, Any]:
    """Load raw JSON dataset from package data.

    Args:
        filename: Name of the JSON file.

    Returns:
        Raw dataset dictionary.
    """
    data_files = resources.files("ragnarok_ai.data")
    with resources.as_file(data_files.joinpath(filename)) as path, path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def _parse_dataset(raw: dict[str, Any]) -> ExampleDataset:
    """Parse raw dataset into ExampleDataset.

    Args:
        raw: Raw dataset dictionary.

    Returns:
        Parsed ExampleDataset.
    """
    documents = [
        Document(
            id=doc["id"],
            content=doc["content"],
            metadata=doc.get("metadata", {}),
        )
        for doc in raw.get("documents", [])
    ]

    queries = [
        Query(
            text=q["text"],
            expected_answer=q.get("expected_answer"),
            ground_truth_docs=q.get("ground_truth_docs", []),
            metadata={
                "id": q.get("id", ""),
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
            },
        )
        for q in raw.get("queries", [])
    ]

    return ExampleDataset(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        version=raw.get("version", "1.0"),
        license=raw.get("license", "MIT"),
        documents=documents,
        queries=queries,
    )


def load_example_dataset(name: str = "novatech") -> ExampleDataset:
    """Load an example dataset by name.

    Args:
        name: Dataset name. Currently available: "novatech".

    Returns:
        ExampleDataset ready for evaluation.

    Raises:
        ValueError: If dataset name is not found.

    Example:
        >>> from ragnarok_ai.data import load_example_dataset
        >>> dataset = load_example_dataset()
        >>> print(f"Loaded {len(dataset.documents)} documents and {len(dataset.queries)} queries")
        >>> testset = dataset.to_testset()
    """
    datasets = {
        "novatech": "novatech_dataset.json",
    }

    if name not in datasets:
        available = ", ".join(datasets.keys())
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)

    raw = _load_raw_dataset(datasets[name])
    return _parse_dataset(raw)


def list_example_datasets() -> list[str]:
    """List available example datasets.

    Returns:
        List of dataset names.
    """
    return ["novatech"]


__all__ = [
    "ExampleDataset",
    "list_example_datasets",
    "load_example_dataset",
]
