"""Unit tests for the generators module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnarok_ai.core.types import Document, TestSet
from ragnarok_ai.generators import (
    GeneratedQuestion,
    GenerationConfig,
    QuestionGeneratorProtocol,
    QuestionValidator,
    SyntheticQuestionGenerator,
    TestSetGenerator,
    load_testset,
    save_testset,
)
from ragnarok_ai.generators.parsing import parse_json_array, parse_json_object

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content=(
                "Paris is the capital of France. It has a population of 2.1 million people. "
                "The city is known for its iconic landmarks including the Eiffel Tower, "
                "the Louvre Museum, and Notre-Dame Cathedral."
            ),
        ),
        Document(
            id="doc2",
            content=(
                "Python is a programming language created by Guido van Rossum in 1991. "
                "It is known for its simple syntax and readability, making it popular "
                "for beginners and experienced developers alike."
            ),
        ),
        Document(
            id="doc3",
            content=(
                "The Eiffel Tower is 330 meters tall and was built in 1889. "
                "It was designed by Gustave Eiffel and has become the most iconic "
                "symbol of Paris and France."
            ),
        ),
    ]


@pytest.fixture
def short_document() -> Document:
    """Create a document that's too short."""
    return Document(id="short", content="Too short")


# ============================================================================
# GeneratedQuestion Tests
# ============================================================================


class TestGeneratedQuestion:
    """Tests for GeneratedQuestion model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating a question with required fields."""
        question = GeneratedQuestion(
            question="What is the capital of France?",
            answer="Paris",
            source_doc_id="doc1",
        )
        assert question.question == "What is the capital of France?"
        assert question.answer == "Paris"
        assert question.source_doc_id == "doc1"
        assert question.question_type == "factual"
        assert question.is_valid is True

    def test_create_with_all_fields(self) -> None:
        """Test creating a question with all fields."""
        question = GeneratedQuestion(
            question="How does Python work?",
            answer="Python is an interpreted language.",
            source_doc_id="doc2",
            question_type="explanatory",
            is_valid=False,
        )
        assert question.question_type == "explanatory"
        assert question.is_valid is False

    def test_frozen_model(self) -> None:
        """Test that GeneratedQuestion is immutable."""
        question = GeneratedQuestion(
            question="Test?",
            answer="Test",
            source_doc_id="doc1",
        )
        with pytest.raises(Exception):  # ValidationError for frozen model
            question.question = "Modified?"  # type: ignore[misc]


# ============================================================================
# GenerationConfig Tests
# ============================================================================


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.num_questions == 50
        assert config.question_types == ["factual", "explanatory"]
        assert config.questions_per_chunk == 3
        assert config.validate_questions is True
        assert config.min_chunk_length == 100

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = GenerationConfig(
            num_questions=20,
            question_types=["factual", "comparative"],
            questions_per_chunk=5,
            validate_questions=False,
            min_chunk_length=50,
        )
        assert config.num_questions == 20
        assert config.question_types == ["factual", "comparative"]
        assert config.questions_per_chunk == 5
        assert config.validate_questions is False
        assert config.min_chunk_length == 50

    def test_validation_num_questions_positive(self) -> None:
        """Test that num_questions must be positive."""
        with pytest.raises(Exception):
            GenerationConfig(num_questions=0)

    def test_validation_min_chunk_length(self) -> None:
        """Test that min_chunk_length has minimum value."""
        with pytest.raises(Exception):
            GenerationConfig(min_chunk_length=5)


# ============================================================================
# SyntheticQuestionGenerator Tests
# ============================================================================


class TestSyntheticQuestionGeneratorInit:
    """Tests for SyntheticQuestionGenerator initialization."""

    def test_init_with_defaults(self, mock_llm: MagicMock) -> None:
        """Test initialization with default config."""
        generator = SyntheticQuestionGenerator(mock_llm)
        assert generator.llm is mock_llm
        assert generator.config.num_questions == 50
        assert generator.validator is not None

    def test_init_with_custom_config(self, mock_llm: MagicMock) -> None:
        """Test initialization with custom config."""
        config = GenerationConfig(num_questions=10)
        generator = SyntheticQuestionGenerator(mock_llm, config=config)
        assert generator.config.num_questions == 10


class TestSyntheticQuestionGeneratorGenerate:
    """Tests for SyntheticQuestionGenerator.generate method."""

    @pytest.mark.asyncio
    async def test_generate_empty_documents(self, mock_llm: MagicMock) -> None:
        """Test generation with empty document list."""
        generator = SyntheticQuestionGenerator(mock_llm)
        result = await generator.generate(documents=[])
        assert isinstance(result, TestSet)
        assert len(result.queries) == 0
        assert "no documents provided" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_documents_too_short(
        self, mock_llm: MagicMock, short_document: Document
    ) -> None:
        """Test generation when all documents are too short."""
        generator = SyntheticQuestionGenerator(mock_llm)
        result = await generator.generate(documents=[short_document])
        assert len(result.queries) == 0
        assert "no documents met minimum length" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_success(
        self, mock_llm: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test successful question generation."""
        # Mock LLM responses
        mock_llm.generate = AsyncMock(
            side_effect=[
                # First doc - question generation
                '["What is the capital of France?", "What is the population of Paris?"]',
                # First doc - answer for Q1
                "Paris",
                # First doc - answer for Q2
                "2.1 million people",
                # Validation for Q1
                '{"is_valid": true, "reason": "Good question"}',
                # Validation for Q2
                '{"is_valid": true, "reason": "Good question"}',
                # Second doc - question generation
                '["Who created Python?"]',
                # Second doc - answer
                "Guido van Rossum",
                # Validation
                '{"is_valid": true, "reason": "Good question"}',
                # Third doc - question generation
                '["How tall is the Eiffel Tower?"]',
                # Third doc - answer
                "330 meters",
                # Validation
                '{"is_valid": true, "reason": "Good question"}',
            ]
        )

        generator = SyntheticQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=4,
            validate=True,
        )

        assert isinstance(result, TestSet)
        assert len(result.queries) <= 4
        assert result.name == "generated_testset"

    @pytest.mark.asyncio
    async def test_generate_without_validation(
        self, mock_llm: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test generation without validation."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["What is the capital of France?"]',
                "Paris",
                '["Who created Python?"]',
                "Guido van Rossum",
                '["How tall is the Eiffel Tower?"]',
                "330 meters",
            ]
        )

        generator = SyntheticQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=3,
            validate=False,
        )

        assert isinstance(result, TestSet)
        assert result.metadata.get("validated") is False

    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(
        self, mock_llm: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test that generation continues when LLM fails for some docs."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                Exception("LLM error"),  # First doc fails
                '["Who created Python?"]',  # Second doc succeeds
                "Guido van Rossum",
                '["How tall is the Eiffel Tower?"]',  # Third doc succeeds
                "330 meters",
            ]
        )

        generator = SyntheticQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=2,
            validate=False,
        )

        # Should still have results from successful docs
        assert isinstance(result, TestSet)

    @pytest.mark.asyncio
    async def test_generate_unanswerable_question_skipped(
        self, mock_llm: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test that UNANSWERABLE questions are skipped."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["What is the weather in Paris?", "What is the capital of France?"]',
                "UNANSWERABLE",  # First question can't be answered
                "Paris",  # Second question answered
            ]
        )

        generator = SyntheticQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=[sample_documents[0]],
            num_questions=2,
            validate=False,
        )

        # Only the answerable question should be included
        assert len(result.queries) == 1
        assert result.queries[0].text == "What is the capital of France?"


class TestSyntheticQuestionGeneratorClassify:
    """Tests for question type classification."""

    def test_classify_explanatory(self, mock_llm: MagicMock) -> None:
        """Test classification of explanatory questions."""
        generator = SyntheticQuestionGenerator(mock_llm)
        q_type = generator._classify_question_type(
            "How does Python work?",
            ["factual", "explanatory"],
        )
        assert q_type == "explanatory"

    def test_classify_comparative(self, mock_llm: MagicMock) -> None:
        """Test classification of comparative questions."""
        generator = SyntheticQuestionGenerator(mock_llm)
        q_type = generator._classify_question_type(
            "What is the difference between Python and Java?",
            ["factual", "comparative"],
        )
        assert q_type == "comparative"

    def test_classify_definitional(self, mock_llm: MagicMock) -> None:
        """Test classification of definitional questions."""
        generator = SyntheticQuestionGenerator(mock_llm)
        q_type = generator._classify_question_type(
            "What is Python?",
            ["factual", "definitional"],
        )
        assert q_type == "definitional"

    def test_classify_multi_hop(self, mock_llm: MagicMock) -> None:
        """Test classification of multi-hop questions."""
        generator = SyntheticQuestionGenerator(mock_llm)
        q_type = generator._classify_question_type(
            "What is the capital of France and also its population?",
            ["factual", "multi_hop"],
        )
        assert q_type == "multi_hop"

    def test_classify_fallback_to_factual(self, mock_llm: MagicMock) -> None:
        """Test fallback to factual type."""
        generator = SyntheticQuestionGenerator(mock_llm)
        q_type = generator._classify_question_type(
            "What is the capital of France?",
            ["factual", "explanatory"],
        )
        assert q_type == "factual"

    def test_classify_fallback_to_first_type(self, mock_llm: MagicMock) -> None:
        """Test fallback to first allowed type."""
        generator = SyntheticQuestionGenerator(mock_llm)
        q_type = generator._classify_question_type(
            "What is the capital of France?",
            ["comparative", "explanatory"],
        )
        assert q_type == "comparative"


class TestJsonParsing:
    """Tests for JSON parsing utilities."""

    def test_parse_json_array_direct(self) -> None:
        """Test parsing a direct JSON array."""
        result = parse_json_array('["Q1", "Q2", "Q3"]')
        assert result == ["Q1", "Q2", "Q3"]

    def test_parse_json_array_with_text(self) -> None:
        """Test parsing JSON array embedded in text."""
        result = parse_json_array(
            'Here are the questions:\n["Q1", "Q2"]\nThat\'s all.'
        )
        assert result == ["Q1", "Q2"]

    def test_parse_json_array_invalid(self) -> None:
        """Test parsing invalid JSON returns empty list."""
        result = parse_json_array("not valid json")
        assert result == []

    def test_parse_json_object_direct(self) -> None:
        """Test parsing a direct JSON object."""
        result = parse_json_object('{"is_valid": true, "reason": "Good"}')
        assert result == {"is_valid": True, "reason": "Good"}

    def test_parse_json_object_with_text(self) -> None:
        """Test parsing JSON object embedded in text."""
        result = parse_json_object('Result:\n{"is_valid": false}\nDone.')
        assert result == {"is_valid": False}

    def test_parse_json_object_invalid(self) -> None:
        """Test parsing invalid JSON returns empty dict."""
        result = parse_json_object("not valid json")
        assert result == {}


# ============================================================================
# QuestionValidator Tests
# ============================================================================


class TestQuestionValidator:
    """Tests for QuestionValidator."""

    @pytest.mark.asyncio
    async def test_validate_valid_question(self, mock_llm: MagicMock) -> None:
        """Test validation of a valid question."""
        mock_llm.generate = AsyncMock(
            return_value='{"is_valid": true, "reason": "Good question"}'
        )

        validator = QuestionValidator(mock_llm)
        question = GeneratedQuestion(
            question="What is the capital of France?",
            answer="Paris",
            source_doc_id="doc1",
        )
        documents = [Document(id="doc1", content="Paris is the capital of France.")]

        result = await validator.validate(question, documents)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_invalid_question(self, mock_llm: MagicMock) -> None:
        """Test validation of an invalid question."""
        mock_llm.generate = AsyncMock(
            return_value='{"is_valid": false, "reason": "Question is ambiguous"}'
        )

        validator = QuestionValidator(mock_llm)
        question = GeneratedQuestion(
            question="What?",
            answer="Something",
            source_doc_id="doc1",
        )
        documents = [Document(id="doc1", content="Some content here.")]

        result = await validator.validate(question, documents)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_missing_source_doc(self, mock_llm: MagicMock) -> None:
        """Test validation when source document is not found."""
        validator = QuestionValidator(mock_llm)
        question = GeneratedQuestion(
            question="Test?",
            answer="Test",
            source_doc_id="missing_doc",
        )
        documents = [Document(id="doc1", content="Some content.")]

        result = await validator.validate(question, documents)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_llm_error_returns_true(self, mock_llm: MagicMock) -> None:
        """Test that LLM errors default to valid."""
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        validator = QuestionValidator(mock_llm)
        question = GeneratedQuestion(
            question="Test?",
            answer="Test",
            source_doc_id="doc1",
        )
        documents = [Document(id="doc1", content="Some content here.")]

        result = await validator.validate(question, documents)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_batch(self, mock_llm: MagicMock) -> None:
        """Test batch validation."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                '{"is_valid": true}',
                '{"is_valid": false}',
                '{"is_valid": true}',
            ]
        )

        validator = QuestionValidator(mock_llm)
        questions = [
            GeneratedQuestion(question="Q1?", answer="A1", source_doc_id="doc1"),
            GeneratedQuestion(question="Q2?", answer="A2", source_doc_id="doc1"),
            GeneratedQuestion(question="Q3?", answer="A3", source_doc_id="doc1"),
        ]
        documents = [Document(id="doc1", content="Some content here." * 10)]

        result = await validator.validate_batch(questions, documents)
        assert len(result) == 2  # Only valid questions

    @pytest.mark.asyncio
    async def test_validate_batch_with_max(self, mock_llm: MagicMock) -> None:
        """Test batch validation with max limit."""
        mock_llm.generate = AsyncMock(return_value='{"is_valid": true}')

        validator = QuestionValidator(mock_llm)
        questions = [
            GeneratedQuestion(question=f"Q{i}?", answer=f"A{i}", source_doc_id="doc1")
            for i in range(5)
        ]
        documents = [Document(id="doc1", content="Some content here." * 10)]

        result = await validator.validate_batch(questions, documents, max_questions=2)
        assert len(result) == 2


# ============================================================================
# I/O Tests
# ============================================================================


class TestSaveLoadTestset:
    """Tests for save_testset and load_testset functions."""

    def test_save_testset(self, tmp_path: Path) -> None:
        """Test saving a test set to file."""
        testset = TestSet(
            queries=[],
            name="test",
            description="Test set",
            metadata={"key": "value"},
        )

        path = tmp_path / "testset.json"
        save_testset(testset, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "test"
        assert data["description"] == "Test set"
        assert data["metadata"] == {"key": "value"}

    def test_save_testset_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save_testset creates parent directories."""
        testset = TestSet(queries=[], name="test")
        path = tmp_path / "subdir" / "nested" / "testset.json"
        save_testset(testset, path)
        assert path.exists()

    def test_load_testset(self, tmp_path: Path) -> None:
        """Test loading a test set from file."""
        data = {
            "name": "loaded_test",
            "description": "Loaded test set",
            "metadata": {"source": "test"},
            "queries": [
                {
                    "text": "What is Python?",
                    "ground_truth_docs": ["doc1"],
                    "expected_answer": "A programming language",
                    "metadata": {"type": "factual"},
                }
            ],
        }

        path = tmp_path / "testset.json"
        path.write_text(json.dumps(data))

        testset = load_testset(path)
        assert testset.name == "loaded_test"
        assert testset.description == "Loaded test set"
        assert len(testset.queries) == 1
        assert testset.queries[0].text == "What is Python?"
        assert testset.queries[0].expected_answer == "A programming language"

    def test_load_testset_file_not_found(self, tmp_path: Path) -> None:
        """Test loading from non-existent file raises error."""
        path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_testset(path)

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading preserves data."""
        from ragnarok_ai.core.types import Query

        original = TestSet(
            queries=[
                Query(
                    text="Test question?",
                    ground_truth_docs=["doc1", "doc2"],
                    expected_answer="Test answer",
                    metadata={"type": "test"},
                )
            ],
            name="roundtrip_test",
            description="Testing roundtrip",
            metadata={"version": "1.0"},
        )

        path = tmp_path / "roundtrip.json"
        save_testset(original, path)
        loaded = load_testset(path)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.metadata == original.metadata
        assert len(loaded.queries) == len(original.queries)
        assert loaded.queries[0].text == original.queries[0].text
        assert loaded.queries[0].expected_answer == original.queries[0].expected_answer


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_synthetic_generator_implements_protocol(self, mock_llm: MagicMock) -> None:
        """Test that SyntheticQuestionGenerator implements QuestionGeneratorProtocol."""
        generator = SyntheticQuestionGenerator(mock_llm)
        assert isinstance(generator, QuestionGeneratorProtocol)

    def test_testset_generator_alias(self, mock_llm: MagicMock) -> None:
        """Test that TestSetGenerator is an alias for SyntheticQuestionGenerator."""
        assert TestSetGenerator is SyntheticQuestionGenerator
        generator = TestSetGenerator(mock_llm)
        assert isinstance(generator, SyntheticQuestionGenerator)
