"""Unit tests for the golden set module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ragnarok_ai.core.types import TestSet
from ragnarok_ai.generators import (
    GoldenQuestion,
    GoldenSet,
    GoldenSetDiff,
)

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# GoldenQuestion Tests
# ============================================================================


class TestGoldenQuestion:
    """Tests for GoldenQuestion model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating a question with required fields."""
        question = GoldenQuestion(
            id="q001",
            question="What is the refund policy?",
            answer="Full refund within 30 days.",
        )
        assert question.id == "q001"
        assert question.question == "What is the refund policy?"
        assert question.answer == "Full refund within 30 days."

    def test_create_with_all_fields(self) -> None:
        """Test creating a question with all fields."""
        question = GoldenQuestion(
            id="q001",
            question="What is the refund policy?",
            answer="Full refund within 30 days.",
            tags=["policy", "critical"],
            source_docs=["policies/refund.md"],
            validated_by="john@example.com",
            validated_at="2024-01-15",
            difficulty="easy",
            priority="critical",
            metadata={"category": "support"},
        )
        assert question.tags == ["policy", "critical"]
        assert question.validated_by == "john@example.com"
        assert question.priority == "critical"

    def test_default_values(self) -> None:
        """Test default values."""
        question = GoldenQuestion(id="q001", question="Test?", answer="Yes")
        assert question.tags == []
        assert question.source_docs == []
        assert question.validated_by is None
        assert question.difficulty is None
        assert question.metadata == {}


# ============================================================================
# GoldenSetDiff Tests
# ============================================================================


class TestGoldenSetDiff:
    """Tests for GoldenSetDiff model."""

    def test_has_changes_true(self) -> None:
        """Test has_changes when there are changes."""
        diff = GoldenSetDiff(
            old_version="1.0.0",
            new_version="1.1.0",
            added=["q003"],
            removed=[],
            modified=[],
            unchanged=["q001", "q002"],
        )
        assert diff.has_changes is True

    def test_has_changes_false(self) -> None:
        """Test has_changes when no changes."""
        diff = GoldenSetDiff(
            old_version="1.0.0",
            new_version="1.0.0",
            added=[],
            removed=[],
            modified=[],
            unchanged=["q001", "q002"],
        )
        assert diff.has_changes is False

    def test_summary(self) -> None:
        """Test summary generation."""
        diff = GoldenSetDiff(
            old_version="1.0.0",
            new_version="1.1.0",
            added=["q003"],
            removed=["q004"],
            modified=["q002"],
            unchanged=["q001"],
        )
        assert "+1 added" in diff.summary
        assert "-1 removed" in diff.summary
        assert "~1 modified" in diff.summary
        assert "=1 unchanged" in diff.summary

    def test_summary_no_changes(self) -> None:
        """Test summary with no changes."""
        diff = GoldenSetDiff(
            old_version="1.0.0",
            new_version="1.0.0",
            added=[],
            removed=[],
            modified=[],
            unchanged=[],
        )
        assert diff.summary == "no changes"


# ============================================================================
# GoldenSet Tests
# ============================================================================


class TestGoldenSetBasic:
    """Basic tests for GoldenSet."""

    def test_create_empty(self) -> None:
        """Test creating an empty golden set."""
        golden = GoldenSet(version="1.0.0")
        assert golden.version == "1.0.0"
        assert len(golden) == 0

    def test_create_with_questions(self) -> None:
        """Test creating golden set with questions."""
        questions = [
            GoldenQuestion(id="q001", question="Q1?", answer="A1"),
            GoldenQuestion(id="q002", question="Q2?", answer="A2"),
        ]
        golden = GoldenSet(version="1.0.0", questions=questions)
        assert len(golden) == 2

    def test_iteration(self) -> None:
        """Test iterating over questions."""
        questions = [
            GoldenQuestion(id="q001", question="Q1?", answer="A1"),
            GoldenQuestion(id="q002", question="Q2?", answer="A2"),
        ]
        golden = GoldenSet(questions=questions)

        ids = [q.id for q in golden]
        assert ids == ["q001", "q002"]

    def test_get_by_id(self) -> None:
        """Test getting question by ID."""
        questions = [
            GoldenQuestion(id="q001", question="Q1?", answer="A1"),
            GoldenQuestion(id="q002", question="Q2?", answer="A2"),
        ]
        golden = GoldenSet(questions=questions)

        q = golden.get_by_id("q002")
        assert q is not None
        assert q.question == "Q2?"

    def test_get_by_id_not_found(self) -> None:
        """Test getting non-existent question."""
        golden = GoldenSet(questions=[])
        assert golden.get_by_id("nonexistent") is None


class TestGoldenSetFilter:
    """Tests for GoldenSet filtering."""

    @pytest.fixture
    def sample_golden_set(self) -> GoldenSet:
        """Create a sample golden set for testing."""
        return GoldenSet(
            version="1.0.0",
            questions=[
                GoldenQuestion(
                    id="q001",
                    question="Q1?",
                    answer="A1",
                    tags=["policy", "critical"],
                    difficulty="easy",
                    priority="high",
                ),
                GoldenQuestion(
                    id="q002",
                    question="Q2?",
                    answer="A2",
                    tags=["technical"],
                    difficulty="hard",
                    priority="low",
                ),
                GoldenQuestion(
                    id="q003",
                    question="Q3?",
                    answer="A3",
                    tags=["policy"],
                    difficulty="easy",
                    priority="critical",
                ),
            ],
        )

    def test_filter_by_tags(self, sample_golden_set: GoldenSet) -> None:
        """Test filtering by tags."""
        filtered = sample_golden_set.filter(tags=["policy"])
        assert len(filtered) == 2
        assert all("policy" in q.tags for q in filtered)

    def test_filter_by_multiple_tags(self, sample_golden_set: GoldenSet) -> None:
        """Test filtering by multiple tags (OR)."""
        filtered = sample_golden_set.filter(tags=["critical", "technical"])
        assert len(filtered) == 2

    def test_filter_by_difficulty(self, sample_golden_set: GoldenSet) -> None:
        """Test filtering by difficulty."""
        filtered = sample_golden_set.filter(difficulty="easy")
        assert len(filtered) == 2
        assert all(q.difficulty == "easy" for q in filtered)

    def test_filter_by_priority(self, sample_golden_set: GoldenSet) -> None:
        """Test filtering by priority."""
        filtered = sample_golden_set.filter(priority="critical")
        assert len(filtered) == 1
        assert filtered.questions[0].id == "q003"

    def test_filter_by_ids(self, sample_golden_set: GoldenSet) -> None:
        """Test filtering by IDs."""
        filtered = sample_golden_set.filter(ids=["q001", "q003"])
        assert len(filtered) == 2

    def test_filter_combined(self, sample_golden_set: GoldenSet) -> None:
        """Test combined filtering."""
        filtered = sample_golden_set.filter(tags=["policy"], difficulty="easy")
        assert len(filtered) == 2

    def test_filter_preserves_metadata(self, sample_golden_set: GoldenSet) -> None:
        """Test that filtering preserves set metadata."""
        filtered = sample_golden_set.filter(tags=["policy"])
        assert filtered.version == sample_golden_set.version


class TestGoldenSetMetadata:
    """Tests for GoldenSet metadata methods."""

    @pytest.fixture
    def sample_golden_set(self) -> GoldenSet:
        """Create a sample golden set."""
        return GoldenSet(
            questions=[
                GoldenQuestion(
                    id="q001", question="Q1?", answer="A1", tags=["a", "b"], difficulty="easy", priority="high"
                ),
                GoldenQuestion(
                    id="q002", question="Q2?", answer="A2", tags=["b", "c"], difficulty="hard", priority="low"
                ),
            ]
        )

    def test_get_tags(self, sample_golden_set: GoldenSet) -> None:
        """Test getting all tags."""
        tags = sample_golden_set.get_tags()
        assert tags == {"a", "b", "c"}

    def test_get_difficulties(self, sample_golden_set: GoldenSet) -> None:
        """Test getting all difficulties."""
        difficulties = sample_golden_set.get_difficulties()
        assert difficulties == {"easy", "hard"}

    def test_get_priorities(self, sample_golden_set: GoldenSet) -> None:
        """Test getting all priorities."""
        priorities = sample_golden_set.get_priorities()
        assert priorities == {"high", "low"}


class TestGoldenSetToTestSet:
    """Tests for converting GoldenSet to TestSet."""

    def test_to_testset(self) -> None:
        """Test conversion to TestSet."""
        golden = GoldenSet(
            version="1.0.0",
            name="Test Golden",
            description="Test description",
            questions=[
                GoldenQuestion(
                    id="q001",
                    question="What is X?",
                    answer="X is Y",
                    tags=["test"],
                    source_docs=["doc1.md"],
                    validated_by="tester",
                    difficulty="easy",
                    priority="high",
                ),
            ],
        )

        testset = golden.to_testset()

        assert isinstance(testset, TestSet)
        assert len(testset.queries) == 1
        assert testset.queries[0].text == "What is X?"
        assert testset.queries[0].expected_answer == "X is Y"
        assert testset.queries[0].ground_truth_docs == ["doc1.md"]
        assert testset.queries[0].metadata["golden_id"] == "q001"
        assert testset.metadata["golden_version"] == "1.0.0"


class TestGoldenSetDiffMethod:
    """Tests for GoldenSet.diff method."""

    def test_diff_no_changes(self) -> None:
        """Test diff with identical sets."""
        golden1 = GoldenSet(
            version="1.0.0",
            questions=[GoldenQuestion(id="q001", question="Q?", answer="A")],
        )
        golden2 = GoldenSet(
            version="1.0.0",
            questions=[GoldenQuestion(id="q001", question="Q?", answer="A")],
        )

        diff = golden1.diff(golden2)
        assert diff.added == []
        assert diff.removed == []
        assert diff.modified == []
        assert diff.unchanged == ["q001"]

    def test_diff_added(self) -> None:
        """Test diff with added questions."""
        golden1 = GoldenSet(
            version="1.0.0",
            questions=[GoldenQuestion(id="q001", question="Q1?", answer="A1")],
        )
        golden2 = GoldenSet(
            version="1.1.0",
            questions=[
                GoldenQuestion(id="q001", question="Q1?", answer="A1"),
                GoldenQuestion(id="q002", question="Q2?", answer="A2"),
            ],
        )

        diff = golden1.diff(golden2)
        assert diff.added == ["q002"]
        assert diff.removed == []
        assert diff.unchanged == ["q001"]

    def test_diff_removed(self) -> None:
        """Test diff with removed questions."""
        golden1 = GoldenSet(
            version="1.0.0",
            questions=[
                GoldenQuestion(id="q001", question="Q1?", answer="A1"),
                GoldenQuestion(id="q002", question="Q2?", answer="A2"),
            ],
        )
        golden2 = GoldenSet(
            version="1.1.0",
            questions=[GoldenQuestion(id="q001", question="Q1?", answer="A1")],
        )

        diff = golden1.diff(golden2)
        assert diff.added == []
        assert diff.removed == ["q002"]

    def test_diff_modified(self) -> None:
        """Test diff with modified questions."""
        golden1 = GoldenSet(
            version="1.0.0",
            questions=[GoldenQuestion(id="q001", question="Q?", answer="Old answer")],
        )
        golden2 = GoldenSet(
            version="1.1.0",
            questions=[GoldenQuestion(id="q001", question="Q?", answer="New answer")],
        )

        diff = golden1.diff(golden2)
        assert diff.modified == ["q001"]
        assert diff.unchanged == []


class TestGoldenSetIO:
    """Tests for GoldenSet save/load functionality."""

    @pytest.fixture
    def sample_golden_set(self) -> GoldenSet:
        """Create a sample golden set."""
        return GoldenSet(
            version="1.0.0",
            name="Test Set",
            description="Test description",
            questions=[
                GoldenQuestion(
                    id="q001",
                    question="What is X?",
                    answer="X is Y",
                    tags=["test", "example"],
                    source_docs=["doc1.md"],
                    validated_by="tester",
                    validated_at="2024-01-15",
                    difficulty="easy",
                    priority="high",
                ),
            ],
            metadata={"source": "test"},
        )

    def test_save_load_json(self, tmp_path: Path, sample_golden_set: GoldenSet) -> None:
        """Test saving and loading JSON."""
        path = tmp_path / "golden.json"
        sample_golden_set.save(path)

        loaded = GoldenSet.load(path)
        assert loaded.version == sample_golden_set.version
        assert loaded.name == sample_golden_set.name
        assert len(loaded) == len(sample_golden_set)
        assert loaded.questions[0].id == "q001"

    def test_save_load_yaml(self, tmp_path: Path, sample_golden_set: GoldenSet) -> None:
        """Test saving and loading YAML."""
        path = tmp_path / "golden.yaml"
        sample_golden_set.save(path)

        loaded = GoldenSet.load(path)
        assert loaded.version == sample_golden_set.version
        assert len(loaded) == len(sample_golden_set)

    def test_save_load_yml(self, tmp_path: Path, sample_golden_set: GoldenSet) -> None:
        """Test saving and loading .yml extension."""
        path = tmp_path / "golden.yml"
        sample_golden_set.save(path)

        loaded = GoldenSet.load(path)
        assert loaded.version == sample_golden_set.version

    def test_save_load_csv(self, tmp_path: Path, sample_golden_set: GoldenSet) -> None:
        """Test saving and loading CSV."""
        path = tmp_path / "golden.csv"
        sample_golden_set.save(path)

        loaded = GoldenSet.load(path)
        assert len(loaded) == len(sample_golden_set)
        assert loaded.questions[0].id == "q001"
        assert loaded.questions[0].tags == ["test", "example"]

    def test_save_explicit_format(self, tmp_path: Path, sample_golden_set: GoldenSet) -> None:
        """Test saving with explicit format."""
        path = tmp_path / "golden.txt"
        sample_golden_set.save(path, format="json")

        loaded = GoldenSet.load(path, format="json")
        assert loaded.version == sample_golden_set.version

    def test_save_creates_parent_dirs(self, tmp_path: Path, sample_golden_set: GoldenSet) -> None:
        """Test that save creates parent directories."""
        path = tmp_path / "subdir" / "nested" / "golden.json"
        sample_golden_set.save(path)
        assert path.exists()

    def test_csv_handles_empty_fields(self, tmp_path: Path) -> None:
        """Test CSV handles empty optional fields."""
        golden = GoldenSet(
            questions=[GoldenQuestion(id="q001", question="Q?", answer="A")],
        )
        path = tmp_path / "golden.csv"
        golden.save(path)

        loaded = GoldenSet.load(path)
        assert loaded.questions[0].validated_by is None
        assert loaded.questions[0].tags == []
