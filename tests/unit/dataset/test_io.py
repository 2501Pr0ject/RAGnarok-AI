"""Unit tests for dataset I/O module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from ragnarok_ai.dataset.io import load_testset, save_testset

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# load_testset Tests - JSON Array
# =============================================================================


class TestLoadTestsetJsonArray:
    """Tests for loading JSON array format."""

    def test_load_simple_array(self, tmp_path: Path) -> None:
        """Load array of query objects."""
        data = [
            {"text": "Question 1", "ground_truth_docs": ["d1"]},
            {"text": "Question 2", "ground_truth_docs": ["d2"]},
        ]
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert len(testset.queries) == 2
        assert testset.queries[0].text == "Question 1"
        assert testset.queries[1].ground_truth_docs == ["d2"]

    def test_load_with_question_field(self, tmp_path: Path) -> None:
        """Load using 'question' field instead of 'text'."""
        data = [{"question": "What is Python?"}]
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert testset.queries[0].text == "What is Python?"

    def test_load_with_query_field(self, tmp_path: Path) -> None:
        """Load using 'query' field instead of 'text'."""
        data = [{"query": "What is JavaScript?"}]
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert testset.queries[0].text == "What is JavaScript?"

    def test_load_with_alternative_ground_truth_fields(self, tmp_path: Path) -> None:
        """Load with various ground truth field names."""
        data = [
            {"text": "Q1", "ground_truth": ["d1"]},
            {"text": "Q2", "relevant_docs": ["d2"]},
            {"text": "Q3", "doc_ids": ["d3"]},
        ]
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert testset.queries[0].ground_truth_docs == ["d1"]
        assert testset.queries[1].ground_truth_docs == ["d2"]
        assert testset.queries[2].ground_truth_docs == ["d3"]

    def test_load_with_expected_answer(self, tmp_path: Path) -> None:
        """Load expected_answer field."""
        data = [{"text": "Q1", "expected_answer": "Answer 1"}]
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert testset.queries[0].expected_answer == "Answer 1"


# =============================================================================
# load_testset Tests - JSON Object
# =============================================================================


class TestLoadTestsetJsonObject:
    """Tests for loading JSON object format."""

    def test_load_with_queries_key(self, tmp_path: Path) -> None:
        """Load object with 'queries' key."""
        data = {
            "name": "my_testset",
            "queries": [
                {"text": "Q1", "ground_truth_docs": ["d1"]},
            ],
        }
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert testset.name == "my_testset"
        assert len(testset.queries) == 1

    def test_load_with_all_fields(self, tmp_path: Path) -> None:
        """Load object with all TestSet fields."""
        data = {
            "name": "full_testset",
            "description": "A complete test set",
            "schema_version": 2,
            "dataset_version": "2.0.0",
            "author": "test_author",
            "source": "generated",
            "queries": [{"text": "Q1"}],
            "metadata": {"key": "value"},
        }
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)

        assert testset.name == "full_testset"
        assert testset.description == "A complete test set"
        assert testset.schema_version == 2
        assert testset.dataset_version == "2.0.0"
        assert testset.author == "test_author"
        assert testset.source == "generated"
        assert testset.metadata == {"key": "value"}


# =============================================================================
# load_testset Tests - JSONL
# =============================================================================


class TestLoadTestsetJsonl:
    """Tests for loading JSONL format."""

    def test_load_jsonl_file(self, tmp_path: Path) -> None:
        """Load JSONL file (one JSON per line)."""
        content = '{"text": "Q1", "ground_truth_docs": ["d1"]}\n{"text": "Q2", "ground_truth_docs": ["d2"]}'
        path = tmp_path / "testset.jsonl"
        path.write_text(content)

        testset = load_testset(path)

        assert len(testset.queries) == 2
        assert testset.queries[0].text == "Q1"
        assert testset.queries[1].text == "Q2"

    def test_load_jsonl_with_empty_lines(self, tmp_path: Path) -> None:
        """JSONL with empty lines is handled."""
        content = '{"text": "Q1"}\n\n{"text": "Q2"}\n'
        path = tmp_path / "testset.jsonl"
        path.write_text(content)

        testset = load_testset(path)

        assert len(testset.queries) == 2


# =============================================================================
# load_testset Tests - Error Cases
# =============================================================================


class TestLoadTestsetErrors:
    """Tests for error handling in load_testset."""

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_testset("/nonexistent/path.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Raises ValueError for invalid JSON."""
        path = tmp_path / "invalid.json"
        path.write_text("not valid json {")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_testset(path)

    def test_missing_text_field(self, tmp_path: Path) -> None:
        """Raises ValueError when query has no text field."""
        data = [{"ground_truth_docs": ["d1"]}]  # Missing text
        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="missing.*text"):
            load_testset(path)


# =============================================================================
# save_testset Tests
# =============================================================================


class TestSaveTestset:
    """Tests for save_testset function."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Saved testset can be loaded back."""
        from ragnarok_ai.core.types import Query, TestSet

        original = TestSet(
            name="roundtrip_test",
            description="Test description",
            dataset_version="1.2.3",
            queries=[
                Query(
                    text="Question 1",
                    ground_truth_docs=["d1", "d2"],
                    expected_answer="Answer 1",
                ),
                Query(
                    text="Question 2",
                    ground_truth_docs=["d3"],
                ),
            ],
        )

        path = tmp_path / "output.json"
        save_testset(original, path)

        loaded = load_testset(path)

        assert loaded.name == "roundtrip_test"
        assert loaded.description == "Test description"
        assert loaded.dataset_version == "1.2.3"
        assert len(loaded.queries) == 2
        assert loaded.queries[0].text == "Question 1"
        assert loaded.queries[0].ground_truth_docs == ["d1", "d2"]
        assert loaded.queries[0].expected_answer == "Answer 1"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_testset creates parent directories."""
        from ragnarok_ai.core.types import Query, TestSet

        testset = TestSet(queries=[Query(text="Q1")])
        path = tmp_path / "nested" / "dir" / "testset.json"

        save_testset(testset, path)

        assert path.exists()

    def test_save_output_is_valid_json(self, tmp_path: Path) -> None:
        """Saved file is valid JSON."""
        from ragnarok_ai.core.types import Query, TestSet

        testset = TestSet(
            name="json_test",
            queries=[Query(text="Test")],
        )
        path = tmp_path / "testset.json"
        save_testset(testset, path)

        # Should not raise
        data: Any = json.loads(path.read_text())
        assert "name" in data
        assert "queries" in data
