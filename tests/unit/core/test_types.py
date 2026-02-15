"""Unit tests for core types module (Document, Query, TestSet, etc.)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import time_machine
from pydantic import ValidationError

from ragnarok_ai.core.types import (
    Document,
    Query,
    RAGResponse,
    RetrievalResult,
    TestSet,
)

# =============================================================================
# Document
# =============================================================================


class TestDocument:
    """Tests for Document model."""

    def test_create_with_required_fields(self) -> None:
        """Document should be created with id and content."""
        doc = Document(id="doc_001", content="Hello world")
        assert doc.id == "doc_001"
        assert doc.content == "Hello world"
        assert doc.metadata == {}

    def test_create_with_metadata(self) -> None:
        """Document should accept metadata."""
        doc = Document(
            id="doc_001",
            content="Hello",
            metadata={"source": "test", "page": 1},
        )
        assert doc.metadata["source"] == "test"
        assert doc.metadata["page"] == 1

    def test_document_is_frozen(self) -> None:
        """Document should be immutable."""
        doc = Document(id="doc_001", content="Hello")
        with pytest.raises(ValidationError):
            doc.id = "changed"  # type: ignore[misc]


# =============================================================================
# Query
# =============================================================================


class TestQuery:
    """Tests for Query model."""

    def test_create_with_text_only(self) -> None:
        """Query should be created with just text."""
        query = Query(text="What is Python?")
        assert query.text == "What is Python?"
        assert query.ground_truth_docs == []
        assert query.expected_answer is None

    def test_create_with_ground_truth(self) -> None:
        """Query should accept ground truth docs."""
        query = Query(
            text="What is Python?",
            ground_truth_docs=["doc_001", "doc_002"],
        )
        assert query.ground_truth_docs == ["doc_001", "doc_002"]

    def test_create_with_expected_answer(self) -> None:
        """Query should accept expected answer."""
        query = Query(
            text="What is the capital of France?",
            expected_answer="Paris",
        )
        assert query.expected_answer == "Paris"


# =============================================================================
# TestSet - Basic
# =============================================================================


class TestTestSetBasic:
    """Basic tests for TestSet model."""

    def test_create_with_queries(self) -> None:
        """TestSet should be created with list of queries."""
        queries = [
            Query(text="Question 1"),
            Query(text="Question 2"),
        ]
        testset = TestSet(queries=queries)
        assert len(testset) == 2

    def test_create_with_name(self) -> None:
        """TestSet should accept name."""
        testset = TestSet(
            name="geography_questions",
            queries=[Query(text="Test")],
        )
        assert testset.name == "geography_questions"

    def test_iter_returns_queries(self) -> None:
        """TestSet should be iterable."""
        queries = [Query(text="Q1"), Query(text="Q2")]
        testset = TestSet(queries=queries)
        result = list(testset)
        assert len(result) == 2
        assert result[0].text == "Q1"


# =============================================================================
# TestSet - Versioning
# =============================================================================


class TestTestSetVersioning:
    """Tests for TestSet versioning features."""

    def test_default_schema_version(self) -> None:
        """TestSet should have schema_version=1 by default."""
        testset = TestSet(queries=[Query(text="Test")])
        assert testset.schema_version == 1

    def test_default_dataset_version(self) -> None:
        """TestSet should have dataset_version='1.0.0' by default."""
        testset = TestSet(queries=[Query(text="Test")])
        assert testset.dataset_version == "1.0.0"

    def test_custom_dataset_version(self) -> None:
        """TestSet should accept custom dataset_version."""
        testset = TestSet(
            queries=[Query(text="Test")],
            dataset_version="2.1.0",
        )
        assert testset.dataset_version == "2.1.0"

    def test_created_at_auto_set(self) -> None:
        """TestSet should auto-set created_at timestamp."""
        testset = TestSet(queries=[Query(text="Test")])
        assert testset.created_at is not None
        assert isinstance(testset.created_at, datetime)

    @time_machine.travel("2024-01-15 10:30:00", tick=False)
    def test_created_at_uses_utc(self) -> None:
        """TestSet created_at should be in UTC."""
        testset = TestSet(queries=[Query(text="Test")])
        assert testset.created_at.tzinfo == timezone.utc

    def test_author_field(self) -> None:
        """TestSet should accept author."""
        testset = TestSet(
            queries=[Query(text="Test")],
            author="john.doe",
        )
        assert testset.author == "john.doe"

    def test_source_field(self) -> None:
        """TestSet should accept source."""
        testset = TestSet(
            queries=[Query(text="Test")],
            source="curated",
        )
        assert testset.source == "curated"


# =============================================================================
# TestSet - Hash Computation
# =============================================================================


class TestTestSetHash:
    """Tests for TestSet hash computation."""

    def test_compute_hash_returns_64_chars(self) -> None:
        """compute_hash() should return full 64-char SHA256."""
        testset = TestSet(queries=[Query(text="Test")])
        hash_value = testset.compute_hash()
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_short_returns_16_chars(self) -> None:
        """hash_short property should return 16-char prefix."""
        testset = TestSet(queries=[Query(text="Test")])
        assert len(testset.hash_short) == 16
        assert testset.hash_short == testset.compute_hash()[:16]

    def test_hash_is_deterministic(self) -> None:
        """Same content should produce same hash."""
        testset1 = TestSet(
            queries=[Query(text="Q1"), Query(text="Q2")],
            name="test",
            dataset_version="1.0.0",
        )
        testset2 = TestSet(
            queries=[Query(text="Q1"), Query(text="Q2")],
            name="test",
            dataset_version="1.0.0",
        )
        assert testset1.compute_hash() == testset2.compute_hash()

    def test_hash_changes_with_queries(self) -> None:
        """Different queries should produce different hash."""
        testset1 = TestSet(queries=[Query(text="Q1")])
        testset2 = TestSet(queries=[Query(text="Q2")])
        assert testset1.compute_hash() != testset2.compute_hash()

    def test_hash_changes_with_version(self) -> None:
        """Different dataset_version should produce different hash."""
        testset1 = TestSet(
            queries=[Query(text="Q1")],
            dataset_version="1.0.0",
        )
        testset2 = TestSet(
            queries=[Query(text="Q1")],
            dataset_version="2.0.0",
        )
        assert testset1.compute_hash() != testset2.compute_hash()

    def test_hash_excludes_created_at(self) -> None:
        """created_at should not affect hash."""
        with time_machine.travel("2024-01-01"):
            testset1 = TestSet(queries=[Query(text="Q1")])

        with time_machine.travel("2024-06-01"):
            testset2 = TestSet(queries=[Query(text="Q1")])

        assert testset1.compute_hash() == testset2.compute_hash()

    def test_hash_excludes_author(self) -> None:
        """author should not affect hash."""
        testset1 = TestSet(
            queries=[Query(text="Q1")],
            author="alice",
        )
        testset2 = TestSet(
            queries=[Query(text="Q1")],
            author="bob",
        )
        assert testset1.compute_hash() == testset2.compute_hash()

    def test_hash_excludes_metadata(self) -> None:
        """metadata should not affect hash."""
        testset1 = TestSet(
            queries=[Query(text="Q1")],
            metadata={"key": "value1"},
        )
        testset2 = TestSet(
            queries=[Query(text="Q1")],
            metadata={"key": "value2"},
        )
        assert testset1.compute_hash() == testset2.compute_hash()

    def test_hash_includes_source(self) -> None:
        """source should affect hash (it's part of dataset identity)."""
        testset1 = TestSet(
            queries=[Query(text="Q1")],
            source="curated",
        )
        testset2 = TestSet(
            queries=[Query(text="Q1")],
            source="generated",
        )
        assert testset1.compute_hash() != testset2.compute_hash()

    def test_hash_includes_name(self) -> None:
        """name should affect hash."""
        testset1 = TestSet(
            queries=[Query(text="Q1")],
            name="dataset_a",
        )
        testset2 = TestSet(
            queries=[Query(text="Q1")],
            name="dataset_b",
        )
        assert testset1.compute_hash() != testset2.compute_hash()


# =============================================================================
# RetrievalResult
# =============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_create_with_required_fields(self) -> None:
        """RetrievalResult should be created with query and docs."""
        query = Query(text="Test")
        docs = [Document(id="d1", content="Content")]
        result = RetrievalResult(query=query, retrieved_docs=docs)
        assert len(result) == 1

    def test_scores_must_match_docs_length(self) -> None:
        """scores and retrieved_docs must have same length."""
        query = Query(text="Test")
        docs = [Document(id="d1", content="C1"), Document(id="d2", content="C2")]
        with pytest.raises(ValueError, match="same length"):
            RetrievalResult(
                query=query,
                retrieved_docs=docs,
                scores=[0.9],  # Only 1 score for 2 docs
            )


# =============================================================================
# RAGResponse
# =============================================================================


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_create_with_required_fields(self) -> None:
        """RAGResponse should be created with answer and docs."""
        docs = [Document(id="d1", content="Content")]
        response = RAGResponse(answer="The answer is 42", retrieved_docs=docs)
        assert response.answer == "The answer is 42"
        assert len(response.retrieved_docs) == 1

    def test_metadata_default_empty(self) -> None:
        """RAGResponse metadata should default to empty dict."""
        response = RAGResponse(answer="Test", retrieved_docs=[])
        assert response.metadata == {}
