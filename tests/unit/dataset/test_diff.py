"""Unit tests for dataset diff module."""

from __future__ import annotations

import pytest

from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.dataset.diff import (
    build_index,
    diff_testsets,
)
from ragnarok_ai.dataset.models import DatasetDiffReport

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def query_a() -> Query:
    """Query with ID in metadata."""
    return Query(
        text="What is Python?",
        ground_truth_docs=["doc1", "doc2"],
        expected_answer="A programming language",
        metadata={"id": "q1"},
    )


@pytest.fixture
def query_b() -> Query:
    """Another query with ID."""
    return Query(
        text="What is JavaScript?",
        ground_truth_docs=["doc3"],
        expected_answer="A scripting language",
        metadata={"id": "q2"},
    )


@pytest.fixture
def query_c() -> Query:
    """Third query with ID."""
    return Query(
        text="What is Rust?",
        ground_truth_docs=["doc4", "doc5"],
        expected_answer="A systems language",
        metadata={"id": "q3"},
    )


@pytest.fixture
def testset_abc(query_a: Query, query_b: Query, query_c: Query) -> TestSet:
    """TestSet with queries a, b, c."""
    return TestSet(name="abc", queries=[query_a, query_b, query_c])


# =============================================================================
# build_index Tests
# =============================================================================


class TestBuildIndex:
    """Tests for build_index function."""

    def test_builds_index_with_metadata_id(self, testset_abc: TestSet) -> None:
        """Index uses metadata.id as key when available."""
        index = build_index(testset_abc)
        assert "q1" in index
        assert "q2" in index
        assert "q3" in index
        assert index["q1"].text == "What is Python?"

    def test_builds_index_with_content_hash(self) -> None:
        """Index uses content hash when no id in metadata."""
        testset = TestSet(
            queries=[
                Query(text="Question 1", ground_truth_docs=["d1"]),
                Query(text="Question 2", ground_truth_docs=["d2"]),
            ]
        )
        index = build_index(testset)
        assert len(index) == 2
        # Keys should be 16-char hashes
        for key in index:
            assert len(key) == 16

    def test_raises_on_duplicate_ids(self) -> None:
        """Raises ValueError when duplicate keys detected."""
        testset = TestSet(
            queries=[
                Query(text="Q1", metadata={"id": "dup"}),
                Query(text="Q2", metadata={"id": "dup"}),
            ]
        )
        with pytest.raises(ValueError, match="Duplicate keys"):
            build_index(testset)

    def test_custom_key_fn(self, testset_abc: TestSet) -> None:
        """Custom key function is used."""

        def custom_key(q: Query) -> str:
            # Use metadata id to ensure unique keys
            return f"custom_{q.metadata.get('id', '')}"

        index = build_index(testset_abc, key_fn=custom_key)
        assert "custom_q1" in index
        assert "custom_q2" in index
        assert "custom_q3" in index


# =============================================================================
# diff_testsets Tests - No Changes
# =============================================================================


class TestDiffTestsetsNoChanges:
    """Tests for diff_testsets when datasets are identical."""

    def test_same_dataset_no_changes(self, testset_abc: TestSet) -> None:
        """Same dataset produces empty diff."""
        report = diff_testsets(testset_abc, testset_abc)

        assert not report.has_changes
        assert len(report.added) == 0
        assert len(report.removed) == 0
        assert len(report.modified) == 0
        assert len(report.unchanged) == 3

    def test_identical_content_different_order(
        self, query_a: Query, query_b: Query, query_c: Query
    ) -> None:
        """Reordering items doesn't produce changes."""
        v1 = TestSet(queries=[query_a, query_b, query_c])
        v2 = TestSet(queries=[query_c, query_a, query_b])

        report = diff_testsets(v1, v2)

        assert not report.has_changes
        assert len(report.unchanged) == 3


# =============================================================================
# diff_testsets Tests - Added Items
# =============================================================================


class TestDiffTestsetsAdded:
    """Tests for detecting added items."""

    def test_detects_added_items(self, query_a: Query, query_b: Query, query_c: Query) -> None:
        """New items in v2 are detected."""
        v1 = TestSet(queries=[query_a])
        v2 = TestSet(queries=[query_a, query_b, query_c])

        report = diff_testsets(v1, v2)

        assert report.has_changes
        assert len(report.added) == 2
        assert "q2" in report.added
        assert "q3" in report.added
        assert len(report.unchanged) == 1


# =============================================================================
# diff_testsets Tests - Removed Items
# =============================================================================


class TestDiffTestsetsRemoved:
    """Tests for detecting removed items."""

    def test_detects_removed_items(self, query_a: Query, query_b: Query, query_c: Query) -> None:
        """Items missing from v2 are detected."""
        v1 = TestSet(queries=[query_a, query_b, query_c])
        v2 = TestSet(queries=[query_a])

        report = diff_testsets(v1, v2)

        assert report.has_changes
        assert len(report.removed) == 2
        assert "q2" in report.removed
        assert "q3" in report.removed
        assert len(report.unchanged) == 1


# =============================================================================
# diff_testsets Tests - Modified Items
# =============================================================================


class TestDiffTestsetsModified:
    """Tests for detecting modified items."""

    def test_detects_text_change(self, query_a: Query) -> None:
        """Modified text is detected."""
        v1 = TestSet(queries=[query_a])
        v2 = TestSet(
            queries=[
                Query(
                    text="What is Python programming language?",  # Changed
                    ground_truth_docs=["doc1", "doc2"],
                    expected_answer="A programming language",
                    metadata={"id": "q1"},
                )
            ]
        )

        report = diff_testsets(v1, v2)

        assert report.has_changes
        assert len(report.modified) == 1
        assert report.modified[0].key == "q1"
        assert "text" in report.modified[0].fields_changed

    def test_detects_ground_truth_change(self, query_a: Query) -> None:
        """Modified ground truth docs is detected."""
        v1 = TestSet(queries=[query_a])
        v2 = TestSet(
            queries=[
                Query(
                    text="What is Python?",
                    ground_truth_docs=["doc1", "doc3"],  # Changed
                    expected_answer="A programming language",
                    metadata={"id": "q1"},
                )
            ]
        )

        report = diff_testsets(v1, v2)

        assert report.has_changes
        assert len(report.modified) == 1
        assert "ground_truth_docs" in report.modified[0].fields_changed

    def test_detects_expected_answer_change(self, query_a: Query) -> None:
        """Modified expected answer is detected."""
        v1 = TestSet(queries=[query_a])
        v2 = TestSet(
            queries=[
                Query(
                    text="What is Python?",
                    ground_truth_docs=["doc1", "doc2"],
                    expected_answer="A high-level programming language",  # Changed
                    metadata={"id": "q1"},
                )
            ]
        )

        report = diff_testsets(v1, v2)

        assert report.has_changes
        assert len(report.modified) == 1
        assert "expected_answer" in report.modified[0].fields_changed

    def test_ground_truth_order_ignored(self, query_a: Query) -> None:
        """Ground truth doc order doesn't matter (canonicalized)."""
        v1 = TestSet(queries=[query_a])  # ["doc1", "doc2"]
        v2 = TestSet(
            queries=[
                Query(
                    text="What is Python?",
                    ground_truth_docs=["doc2", "doc1"],  # Reversed order
                    expected_answer="A programming language",
                    metadata={"id": "q1"},
                )
            ]
        )

        report = diff_testsets(v1, v2)

        assert not report.has_changes
        assert len(report.unchanged) == 1


# =============================================================================
# diff_testsets Tests - Metadata Handling
# =============================================================================


class TestDiffTestsetsMetadata:
    """Tests for metadata handling in diff."""

    def test_ignore_metadata_by_default(self) -> None:
        """Metadata changes are ignored by default."""
        v1 = TestSet(
            queries=[
                Query(
                    text="Q1",
                    metadata={"id": "q1", "author": "alice"},
                )
            ]
        )
        v2 = TestSet(
            queries=[
                Query(
                    text="Q1",
                    metadata={"id": "q1", "author": "bob"},  # Metadata changed
                )
            ]
        )

        report = diff_testsets(v1, v2, ignore_metadata=True)

        assert not report.has_changes

    def test_include_metadata_when_requested(self) -> None:
        """Metadata changes detected when ignore_metadata=False."""
        v1 = TestSet(
            queries=[
                Query(
                    text="Q1",
                    metadata={"id": "q1", "category": "general"},
                )
            ]
        )
        v2 = TestSet(
            queries=[
                Query(
                    text="Q1",
                    metadata={"id": "q1", "category": "specific"},
                )
            ]
        )

        report = diff_testsets(v1, v2, ignore_metadata=False)

        assert report.has_changes
        assert len(report.modified) == 1


# =============================================================================
# diff_testsets Tests - Mixed Changes
# =============================================================================


class TestDiffTestsetsMixed:
    """Tests for mixed changes (add + remove + modify)."""

    def test_mixed_changes(self, query_a: Query, query_b: Query, query_c: Query) -> None:
        """Handles add, remove, and modify simultaneously.

        v1: [a, b]
        v2: [a_modified, c] -> b removed, a modified, c added
        """
        v1 = TestSet(queries=[query_a, query_b])
        v2 = TestSet(
            queries=[
                Query(
                    text="What is Python 3?",  # Modified
                    ground_truth_docs=["doc1", "doc2"],
                    expected_answer="A programming language",
                    metadata={"id": "q1"},
                ),
                query_c,  # Added
            ]
        )

        report = diff_testsets(v1, v2)

        assert report.has_changes
        assert len(report.added) == 1
        assert "q3" in report.added
        assert len(report.removed) == 1
        assert "q2" in report.removed
        assert len(report.modified) == 1
        assert report.modified[0].key == "q1"


# =============================================================================
# DatasetDiffReport Tests
# =============================================================================


class TestDatasetDiffReport:
    """Tests for DatasetDiffReport model."""

    def test_summary(self) -> None:
        """summary() returns correct counts."""
        report = DatasetDiffReport(
            v1_hash="abc123",
            v2_hash="def456",
            added=["a", "b"],
            removed=["c"],
            modified=[],
            unchanged=["d", "e", "f"],
        )

        summary = report.summary()
        assert summary["added"] == 2
        assert summary["removed"] == 1
        assert summary["modified"] == 0
        assert summary["unchanged"] == 3

    def test_has_changes_true(self) -> None:
        """has_changes is True when any diff exists."""
        report = DatasetDiffReport(
            v1_hash="a",
            v2_hash="b",
            added=["new"],
        )
        assert report.has_changes

    def test_has_changes_false(self) -> None:
        """has_changes is False when no diff."""
        report = DatasetDiffReport(
            v1_hash="a",
            v2_hash="b",
            unchanged=["x", "y"],
        )
        assert not report.has_changes

    def test_to_dict(self) -> None:
        """to_dict() is JSON-serializable."""
        import json

        report = DatasetDiffReport(
            v1_hash="hash1",
            v2_hash="hash2",
            added=["new"],
            removed=["old"],
            unchanged=["same"],
        )

        data = report.to_dict()
        json_str = json.dumps(data)  # Should not raise
        assert "hash1" in json_str
