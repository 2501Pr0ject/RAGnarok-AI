"""Core type definitions for ragnarok-ai.

This module defines the fundamental data structures used throughout
the library for documents, queries, and evaluation results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator


class Document(BaseModel):
    """A document in the knowledge base or retrieval result.

    Attributes:
        id: Unique identifier for the document.
        content: The text content of the document.
        metadata: Optional metadata associated with the document.

    Example:
        >>> doc = Document(
        ...     id="doc_001",
        ...     content="Paris is the capital of France.",
        ...     metadata={"source": "wikipedia", "page": 42},
        ... )
    """

    model_config = {"frozen": True}

    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Text content of the document")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the document",
    )


class Query(BaseModel):
    """A query with optional ground truth for evaluation.

    Attributes:
        text: The query text.
        ground_truth_docs: List of document IDs that are relevant to this query.
        expected_answer: Optional expected answer for generation evaluation.
        metadata: Optional metadata associated with the query.

    Example:
        >>> query = Query(
        ...     text="What is the capital of France?",
        ...     ground_truth_docs=["doc_001", "doc_042"],
        ...     expected_answer="Paris",
        ... )
    """

    model_config = {"frozen": True}

    text: str = Field(..., description="The query text")
    ground_truth_docs: list[str] = Field(
        default_factory=list,
        description="List of relevant document IDs (ground truth)",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Optional expected answer for generation evaluation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the query",
    )


class TestSet(BaseModel):
    """A collection of queries for evaluation.

    Attributes:
        queries: List of queries to evaluate.
        name: Optional name for the test set.
        description: Optional description of the test set.
        metadata: Optional metadata associated with the test set.

    Example:
        >>> testset = TestSet(
        ...     name="geography_questions",
        ...     queries=[
        ...         Query(text="What is the capital of France?"),
        ...         Query(text="What is the largest country?"),
        ...     ],
        ... )
        >>> len(testset)
        2
    """

    queries: list[Query] = Field(..., description="List of queries to evaluate")
    name: str | None = Field(default=None, description="Optional name for the test set")
    description: str | None = Field(default=None, description="Optional description")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the test set",
    )

    def __len__(self) -> int:
        """Return the number of queries in the test set."""
        return len(self.queries)

    def __iter__(self) -> Iterator[Query]:  # type: ignore[override]
        """Iterate over queries in the test set."""
        return iter(self.queries)


class RetrievalResult(BaseModel):
    """Result of a retrieval operation for a single query.

    Attributes:
        query: The original query.
        retrieved_docs: List of retrieved documents.
        scores: Relevance scores for each retrieved document.
        latency_ms: Time taken for retrieval in milliseconds.

    Example:
        >>> result = RetrievalResult(
        ...     query=Query(text="What is RAG?"),
        ...     retrieved_docs=[doc1, doc2, doc3],
        ...     scores=[0.95, 0.82, 0.71],
        ...     latency_ms=45.2,
        ... )
    """

    model_config = {"frozen": True}

    query: Query = Field(..., description="The original query")
    retrieved_docs: list[Document] = Field(..., description="List of retrieved documents")
    scores: list[float] = Field(
        default_factory=list,
        description="Relevance scores for each retrieved document",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Time taken for retrieval in milliseconds",
    )

    @model_validator(mode="after")
    def _validate_scores_length(self) -> Self:
        """Validate that scores and retrieved_docs have the same length."""
        if self.scores and len(self.scores) != len(self.retrieved_docs):
            msg = f"scores ({len(self.scores)}) and retrieved_docs ({len(self.retrieved_docs)}) must have same length"
            raise ValueError(msg)
        return self

    def __len__(self) -> int:
        """Return the number of retrieved documents."""
        return len(self.retrieved_docs)


class RAGResponse(BaseModel):
    """Response from a RAG pipeline query."""

    answer: str = Field(..., description="Generated answer from the RAG pipeline")
    retrieved_docs: list[Document] = Field(..., description="Retrieved documents")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
