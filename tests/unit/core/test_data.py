"""Unit tests for the data module."""

from __future__ import annotations

import pytest

from ragnarok_ai.core.types import TestSet
from ragnarok_ai.data import (
    ExampleDataset,
    list_example_datasets,
    load_example_dataset,
)

# ============================================================================
# load_example_dataset Tests
# ============================================================================


class TestLoadExampleDataset:
    """Tests for load_example_dataset function."""

    def test_load_novatech(self) -> None:
        """Test loading the NovaTech dataset."""
        dataset = load_example_dataset("novatech")
        assert dataset.name == "novatech_quickstart"
        assert "NovaTech" in dataset.description
        assert dataset.version == "1.0"
        assert dataset.license == "MIT"

    def test_load_default(self) -> None:
        """Test loading with default name."""
        dataset = load_example_dataset()
        assert dataset.name == "novatech_quickstart"

    def test_load_unknown_raises(self) -> None:
        """Test that unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset 'nonexistent'"):
            load_example_dataset("nonexistent")

    def test_has_documents(self) -> None:
        """Test that dataset has documents."""
        dataset = load_example_dataset()
        assert len(dataset.documents) > 0
        assert all(doc.id for doc in dataset.documents)
        assert all(doc.content for doc in dataset.documents)

    def test_has_queries(self) -> None:
        """Test that dataset has queries."""
        dataset = load_example_dataset()
        assert len(dataset.queries) > 0
        assert all(q.text for q in dataset.queries)

    def test_queries_have_ground_truth(self) -> None:
        """Test that queries have ground truth documents."""
        dataset = load_example_dataset()
        for query in dataset.queries:
            assert query.ground_truth_docs, f"Query '{query.text}' has no ground truth docs"

    def test_queries_have_expected_answers(self) -> None:
        """Test that queries have expected answers."""
        dataset = load_example_dataset()
        for query in dataset.queries:
            assert query.expected_answer, f"Query '{query.text}' has no expected answer"


class TestListExampleDatasets:
    """Tests for list_example_datasets function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        datasets = list_example_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_contains_novatech(self) -> None:
        """Test that list contains novatech."""
        datasets = list_example_datasets()
        assert "novatech" in datasets


# ============================================================================
# ExampleDataset Tests
# ============================================================================


class TestExampleDataset:
    """Tests for ExampleDataset model."""

    @pytest.fixture
    def novatech_dataset(self) -> ExampleDataset:
        """Load NovaTech dataset for testing."""
        return load_example_dataset("novatech")

    def test_to_testset(self, novatech_dataset: ExampleDataset) -> None:
        """Test conversion to TestSet."""
        testset = novatech_dataset.to_testset()
        assert isinstance(testset, TestSet)
        assert len(testset.queries) == len(novatech_dataset.queries)
        assert testset.name == novatech_dataset.name
        assert testset.metadata["version"] == novatech_dataset.version

    def test_get_document(self, novatech_dataset: ExampleDataset) -> None:
        """Test getting document by ID."""
        doc = novatech_dataset.get_document("api_authentication")
        assert doc is not None
        assert "Authentication" in doc.content

    def test_get_document_not_found(self, novatech_dataset: ExampleDataset) -> None:
        """Test getting non-existent document."""
        doc = novatech_dataset.get_document("nonexistent")
        assert doc is None

    def test_get_query(self, novatech_dataset: ExampleDataset) -> None:
        """Test getting query by ID."""
        query = novatech_dataset.get_query("q001")
        assert query is not None
        assert "authenticate" in query.text.lower()

    def test_get_query_not_found(self, novatech_dataset: ExampleDataset) -> None:
        """Test getting non-existent query."""
        query = novatech_dataset.get_query("nonexistent")
        assert query is None

    def test_filter_by_category(self, novatech_dataset: ExampleDataset) -> None:
        """Test filtering queries by category."""
        api_queries = novatech_dataset.filter_by_category("api_reference")
        assert len(api_queries) > 0
        assert all(q.metadata.get("category") == "api_reference" for q in api_queries)

        security_queries = novatech_dataset.filter_by_category("security")
        assert len(security_queries) > 0
        assert all(q.metadata.get("category") == "security" for q in security_queries)

    def test_filter_by_difficulty(self, novatech_dataset: ExampleDataset) -> None:
        """Test filtering queries by difficulty."""
        easy_queries = novatech_dataset.filter_by_difficulty("easy")
        assert len(easy_queries) > 0
        assert all(q.metadata.get("difficulty") == "easy" for q in easy_queries)

        medium_queries = novatech_dataset.filter_by_difficulty("medium")
        assert len(medium_queries) > 0
        assert all(q.metadata.get("difficulty") == "medium" for q in medium_queries)


# ============================================================================
# NovaTech Dataset Content Tests
# ============================================================================


class TestNovaTechDatasetContent:
    """Tests for NovaTech dataset content quality."""

    @pytest.fixture
    def dataset(self) -> ExampleDataset:
        """Load NovaTech dataset."""
        return load_example_dataset("novatech")

    def test_has_all_categories(self, dataset: ExampleDataset) -> None:
        """Test that all expected categories are present."""
        categories = {q.metadata.get("category") for q in dataset.queries}
        expected = {"api_reference", "security", "billing", "troubleshooting", "best_practices"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"

    def test_has_all_difficulties(self, dataset: ExampleDataset) -> None:
        """Test that all difficulty levels are present."""
        difficulties = {q.metadata.get("difficulty") for q in dataset.queries}
        assert "easy" in difficulties
        assert "medium" in difficulties

    def test_ground_truth_docs_exist(self, dataset: ExampleDataset) -> None:
        """Test that all ground truth document IDs reference existing documents."""
        doc_ids = {doc.id for doc in dataset.documents}
        for query in dataset.queries:
            for gt_doc in query.ground_truth_docs:
                assert gt_doc in doc_ids, f"Ground truth doc '{gt_doc}' not found for query '{query.text}'"

    def test_document_content_quality(self, dataset: ExampleDataset) -> None:
        """Test that documents have meaningful content."""
        for doc in dataset.documents:
            assert len(doc.content) > 100, f"Document '{doc.id}' has very short content"
            assert "NovaTech" in doc.content, f"Document '{doc.id}' doesn't mention NovaTech"

    def test_query_answer_relevance(self, dataset: ExampleDataset) -> None:
        """Test that expected answers are relevant to questions."""
        for query in dataset.queries:
            # Basic check: answer should not be empty
            assert query.expected_answer
            assert len(query.expected_answer) > 10, f"Query '{query.text}' has very short answer"

    def test_minimum_dataset_size(self, dataset: ExampleDataset) -> None:
        """Test that dataset has minimum required size."""
        assert len(dataset.documents) >= 20, "Dataset should have at least 20 documents"
        assert len(dataset.queries) >= 30, "Dataset should have at least 30 queries"
