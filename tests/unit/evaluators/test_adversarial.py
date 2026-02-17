"""Unit tests for the adversarial question generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ragnarok_ai.core.types import Document, TestSet
from ragnarok_ai.generators import (
    AdversarialConfig,
    AdversarialQuestion,
    AdversarialQuestionGenerator,
    AdversarialType,
    ExpectedBehavior,
)

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
    """Create sample documents for adversarial testing."""
    return [
        Document(
            id="doc_paris",
            content=(
                "Paris is the capital of France. It has a population of 2.1 million people. "
                "The city is known for the Eiffel Tower, built in 1889 by Gustave Eiffel."
            ),
        ),
        Document(
            id="doc_london",
            content=(
                "London is the capital of the United Kingdom. It has a population of 8.8 million. "
                "Famous landmarks include Big Ben and the Tower of London."
            ),
        ),
    ]


# ============================================================================
# AdversarialType Tests
# ============================================================================


class TestAdversarialType:
    """Tests for AdversarialType enum."""

    def test_all_types_exist(self) -> None:
        """Test that all expected types are defined."""
        assert AdversarialType.OUT_OF_SCOPE == "out_of_scope"
        assert AdversarialType.AMBIGUOUS == "ambiguous"
        assert AdversarialType.MISLEADING == "misleading"
        assert AdversarialType.BOUNDARY == "boundary"

    def test_type_from_string(self) -> None:
        """Test creating type from string."""
        assert AdversarialType("out_of_scope") == AdversarialType.OUT_OF_SCOPE
        assert AdversarialType("ambiguous") == AdversarialType.AMBIGUOUS


# ============================================================================
# ExpectedBehavior Tests
# ============================================================================


class TestExpectedBehavior:
    """Tests for ExpectedBehavior enum."""

    def test_all_behaviors_exist(self) -> None:
        """Test that all expected behaviors are defined."""
        assert ExpectedBehavior.REFUSE == "refuse"
        assert ExpectedBehavior.CLARIFY == "clarify"
        assert ExpectedBehavior.CORRECT == "correct"
        assert ExpectedBehavior.HANDLE_GRACEFULLY == "handle_gracefully"


# ============================================================================
# AdversarialQuestion Tests
# ============================================================================


class TestAdversarialQuestion:
    """Tests for AdversarialQuestion model."""

    def test_create_question(self) -> None:
        """Test creating an adversarial question."""
        question = AdversarialQuestion(
            question="What is the population of Tokyo?",
            adversarial_type=AdversarialType.OUT_OF_SCOPE,
            expected_behavior=ExpectedBehavior.REFUSE,
            explanation="Tokyo is not mentioned in the documents",
        )
        assert question.question == "What is the population of Tokyo?"
        assert question.adversarial_type == AdversarialType.OUT_OF_SCOPE
        assert question.expected_behavior == ExpectedBehavior.REFUSE

    def test_question_frozen(self) -> None:
        """Test that AdversarialQuestion is immutable."""
        question = AdversarialQuestion(
            question="Test?",
            adversarial_type=AdversarialType.AMBIGUOUS,
            expected_behavior=ExpectedBehavior.CLARIFY,
        )
        with pytest.raises(ValidationError):
            question.question = "Modified?"  # type: ignore[misc]

    def test_question_default_explanation(self) -> None:
        """Test default explanation is empty string."""
        question = AdversarialQuestion(
            question="Test?",
            adversarial_type=AdversarialType.BOUNDARY,
            expected_behavior=ExpectedBehavior.HANDLE_GRACEFULLY,
        )
        assert question.explanation == ""


# ============================================================================
# AdversarialConfig Tests
# ============================================================================


class TestAdversarialConfig:
    """Tests for AdversarialConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AdversarialConfig()
        assert config.num_questions == 20
        assert len(config.adversarial_types) == 4  # All types
        assert config.min_chunk_length == 50

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AdversarialConfig(
            num_questions=10,
            adversarial_types=[AdversarialType.OUT_OF_SCOPE, AdversarialType.AMBIGUOUS],
            min_chunk_length=100,
        )
        assert config.num_questions == 10
        assert len(config.adversarial_types) == 2

    def test_validation_num_questions(self) -> None:
        """Test that num_questions must be positive."""
        with pytest.raises(ValidationError):
            AdversarialConfig(num_questions=0)


# ============================================================================
# AdversarialQuestionGenerator Tests
# ============================================================================


class TestAdversarialQuestionGeneratorInit:
    """Tests for AdversarialQuestionGenerator initialization."""

    def test_init_with_defaults(self, mock_llm: MagicMock) -> None:
        """Test initialization with default config."""
        generator = AdversarialQuestionGenerator(mock_llm)
        assert generator.llm is mock_llm
        assert generator.config.num_questions == 20

    def test_init_with_custom_config(self, mock_llm: MagicMock) -> None:
        """Test initialization with custom config."""
        config = AdversarialConfig(num_questions=10)
        generator = AdversarialQuestionGenerator(mock_llm, config=config)
        assert generator.config.num_questions == 10


class TestAdversarialQuestionGeneratorGenerate:
    """Tests for AdversarialQuestionGenerator.generate method."""

    @pytest.mark.asyncio
    async def test_generate_empty_documents(self, mock_llm: MagicMock) -> None:
        """Test generation with empty document list."""
        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(documents=[])
        assert isinstance(result, TestSet)
        assert len(result.queries) == 0
        assert "no documents provided" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_documents_too_short(self, mock_llm: MagicMock) -> None:
        """Test generation when documents are too short."""
        generator = AdversarialQuestionGenerator(mock_llm)
        short_docs = [Document(id="doc1", content="Short")]
        result = await generator.generate(documents=short_docs)
        assert len(result.queries) == 0
        assert "no documents met minimum length" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_out_of_scope(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test generating out-of-scope questions."""
        mock_llm.generate = AsyncMock(return_value='["What is the population of Tokyo?", "Who founded Berlin?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=2,
            adversarial_types=["out_of_scope"],
        )

        assert isinstance(result, TestSet)
        assert len(result.queries) == 2
        assert result.queries[0].metadata["adversarial_type"] == "out_of_scope"
        assert result.queries[0].metadata["expected_behavior"] == "refuse"

    @pytest.mark.asyncio
    async def test_generate_ambiguous(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test generating ambiguous questions."""
        mock_llm.generate = AsyncMock(return_value='["How big is the capital?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=1,
            adversarial_types=["ambiguous"],
        )

        assert len(result.queries) == 1
        assert result.queries[0].metadata["adversarial_type"] == "ambiguous"
        assert result.queries[0].metadata["expected_behavior"] == "clarify"

    @pytest.mark.asyncio
    async def test_generate_misleading(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test generating misleading questions."""
        mock_llm.generate = AsyncMock(return_value='["Why was the Eiffel Tower built in 1950?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=1,
            adversarial_types=["misleading"],
        )

        assert len(result.queries) == 1
        assert result.queries[0].metadata["adversarial_type"] == "misleading"
        assert result.queries[0].metadata["expected_behavior"] == "correct"

    @pytest.mark.asyncio
    async def test_generate_boundary(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test generating boundary/edge case questions."""
        mock_llm.generate = AsyncMock(return_value='["Paris?", "Tell me everything about..."]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=2,
            adversarial_types=["boundary"],
        )

        assert len(result.queries) == 2
        assert result.queries[0].metadata["adversarial_type"] == "boundary"
        assert result.queries[0].metadata["expected_behavior"] == "handle_gracefully"

    @pytest.mark.asyncio
    async def test_generate_multiple_types(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test generating multiple adversarial types."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["Out of scope question?"]',
                '["Ambiguous question?"]',
            ]
        )

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=2,
            adversarial_types=["out_of_scope", "ambiguous"],
        )

        assert len(result.queries) == 2
        types = {q.metadata["adversarial_type"] for q in result.queries}
        assert "out_of_scope" in types
        assert "ambiguous" in types

    @pytest.mark.asyncio
    async def test_generate_all_types_by_default(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test that all types are generated by default."""
        mock_llm.generate = AsyncMock(return_value='["Question?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=4,
        )

        assert result.metadata["adversarial_types"] == ["out_of_scope", "ambiguous", "misleading", "boundary"]

    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test that generation handles LLM errors gracefully."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                Exception("LLM error"),  # First type fails
                '["Ambiguous question?"]',  # Second succeeds
            ]
        )

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=2,
            adversarial_types=["out_of_scope", "ambiguous"],
        )

        # Should still have results from successful type
        assert isinstance(result, TestSet)

    @pytest.mark.asyncio
    async def test_generate_no_ground_truth_docs(self, mock_llm: MagicMock, sample_documents: list[Document]) -> None:
        """Test that adversarial questions have empty ground truth docs."""
        mock_llm.generate = AsyncMock(return_value='["Out of scope question?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=1,
            adversarial_types=["out_of_scope"],
        )

        assert result.queries[0].ground_truth_docs == []
        assert result.queries[0].expected_answer is None

    @pytest.mark.asyncio
    async def test_generate_metadata_includes_explanation(
        self, mock_llm: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test that metadata includes explanation."""
        mock_llm.generate = AsyncMock(return_value='["Out of scope question?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=sample_documents,
            num_questions=1,
            adversarial_types=["out_of_scope"],
        )

        assert "explanation" in result.queries[0].metadata
        assert "not present in the knowledge base" in result.queries[0].metadata["explanation"]


class TestAdversarialQuestionGeneratorHelpers:
    """Tests for helper methods."""

    def test_get_prompt_for_type(self, mock_llm: MagicMock) -> None:
        """Test getting prompts for each type."""
        generator = AdversarialQuestionGenerator(mock_llm)

        for adv_type in AdversarialType:
            prompt = generator._get_prompt_for_type(adv_type)
            assert "{num_questions}" in prompt
            assert "{documents}" in prompt

    def test_get_explanation_for_type(self, mock_llm: MagicMock) -> None:
        """Test getting explanations for each type."""
        generator = AdversarialQuestionGenerator(mock_llm)

        explanations = {
            AdversarialType.OUT_OF_SCOPE: "not present",
            AdversarialType.AMBIGUOUS: "multiple valid interpretations",
            AdversarialType.MISLEADING: "false premise",
            AdversarialType.BOUNDARY: "robustness",
        }

        for adv_type, expected_substr in explanations.items():
            explanation = generator._get_explanation_for_type(adv_type)
            assert expected_substr in explanation.lower()

    @pytest.mark.asyncio
    async def test_generate_for_type_invalid_json(self, mock_llm: MagicMock) -> None:
        """Test handling of invalid JSON response."""
        mock_llm.generate = AsyncMock(return_value="not valid json")

        generator = AdversarialQuestionGenerator(mock_llm)
        docs = [Document(id="doc1", content="Some content here for testing." * 5)]

        result = await generator._generate_for_type(docs, AdversarialType.OUT_OF_SCOPE, num_questions=1)

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_for_type_empty_questions(self, mock_llm: MagicMock) -> None:
        """Test handling of empty question strings."""
        mock_llm.generate = AsyncMock(return_value='["", "   ", "Valid question?"]')

        generator = AdversarialQuestionGenerator(mock_llm)
        docs = [Document(id="doc1", content="Some content here for testing." * 5)]

        result = await generator._generate_for_type(docs, AdversarialType.AMBIGUOUS, num_questions=3)

        # Should only include valid question
        assert len(result) == 1
        assert result[0].question == "Valid question?"
