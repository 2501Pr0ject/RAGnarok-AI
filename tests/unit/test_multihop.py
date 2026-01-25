"""Unit tests for the multi-hop question generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ragnarok_ai.core.types import Document, TestSet
from ragnarok_ai.generators import (
    DocumentRelationship,
    MultiHopConfig,
    MultiHopQuestion,
    MultiHopQuestionGenerator,
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
def related_documents() -> list[Document]:
    """Create documents with relationships for multi-hop testing."""
    return [
        Document(
            id="doc_alice",
            content="Alice is a software engineer who works at Acme Corporation. She has been with the company for 5 years.",
        ),
        Document(
            id="doc_acme",
            content="Acme Corporation is a technology company headquartered in Paris, France. It was founded in 2010.",
        ),
        Document(
            id="doc_paris",
            content="Paris is the capital of France with a population of over 2 million people. It is known for the Eiffel Tower.",
        ),
    ]


@pytest.fixture
def unrelated_documents() -> list[Document]:
    """Create documents without clear relationships."""
    return [
        Document(
            id="doc_weather",
            content="The weather forecast for tomorrow shows sunny skies with temperatures around 25 degrees Celsius.",
        ),
        Document(
            id="doc_recipe",
            content="To make chocolate cake, you need flour, sugar, cocoa powder, eggs, and butter. Mix all ingredients.",
        ),
    ]


# ============================================================================
# DocumentRelationship Tests
# ============================================================================


class TestDocumentRelationship:
    """Tests for DocumentRelationship model."""

    def test_create_relationship(self) -> None:
        """Test creating a document relationship."""
        rel = DocumentRelationship(
            doc_a_id="doc1",
            doc_b_id="doc2",
            relationship_type="employment",
            shared_entities=["Alice", "Acme Corp"],
            bridge_entity="Acme Corp",
        )
        assert rel.doc_a_id == "doc1"
        assert rel.doc_b_id == "doc2"
        assert rel.relationship_type == "employment"
        assert rel.shared_entities == ["Alice", "Acme Corp"]
        assert rel.bridge_entity == "Acme Corp"

    def test_relationship_frozen(self) -> None:
        """Test that DocumentRelationship is immutable."""
        rel = DocumentRelationship(
            doc_a_id="doc1",
            doc_b_id="doc2",
            relationship_type="test",
        )
        with pytest.raises(ValidationError):
            rel.doc_a_id = "modified"  # type: ignore[misc]


# ============================================================================
# MultiHopQuestion Tests
# ============================================================================


class TestMultiHopQuestion:
    """Tests for MultiHopQuestion model."""

    def test_create_question(self) -> None:
        """Test creating a multi-hop question."""
        question = MultiHopQuestion(
            question="In which city does Alice work?",
            answer="Paris",
            hop_chain=["doc_alice", "doc_acme"],
            reasoning_steps="Alice → Acme Corp → Paris",
        )
        assert question.question == "In which city does Alice work?"
        assert question.answer == "Paris"
        assert question.hop_chain == ["doc_alice", "doc_acme"]

    def test_question_frozen(self) -> None:
        """Test that MultiHopQuestion is immutable."""
        question = MultiHopQuestion(
            question="Test?",
            answer="Test",
            hop_chain=["doc1"],
            reasoning_steps="",
        )
        with pytest.raises(ValidationError):
            question.answer = "modified"  # type: ignore[misc]


# ============================================================================
# MultiHopConfig Tests
# ============================================================================


class TestMultiHopConfig:
    """Tests for MultiHopConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MultiHopConfig()
        assert config.num_questions == 20
        assert config.max_hops == 3
        assert config.min_hops == 2
        assert config.min_chunk_length == 50

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MultiHopConfig(
            num_questions=10,
            max_hops=4,
            min_hops=2,
            min_chunk_length=100,
        )
        assert config.num_questions == 10
        assert config.max_hops == 4

    def test_validation_max_hops(self) -> None:
        """Test that max_hops must be at least 2."""
        with pytest.raises(ValidationError):
            MultiHopConfig(max_hops=1)

    def test_validation_max_hops_upper_limit(self) -> None:
        """Test that max_hops has an upper limit."""
        with pytest.raises(ValidationError):
            MultiHopConfig(max_hops=10)


# ============================================================================
# MultiHopQuestionGenerator Tests
# ============================================================================


class TestMultiHopQuestionGeneratorInit:
    """Tests for MultiHopQuestionGenerator initialization."""

    def test_init_with_defaults(self, mock_llm: MagicMock) -> None:
        """Test initialization with default config."""
        generator = MultiHopQuestionGenerator(mock_llm)
        assert generator.llm is mock_llm
        assert generator.config.num_questions == 20
        assert generator.config.max_hops == 3

    def test_init_with_custom_config(self, mock_llm: MagicMock) -> None:
        """Test initialization with custom config."""
        config = MultiHopConfig(num_questions=10, max_hops=4)
        generator = MultiHopQuestionGenerator(mock_llm, config=config)
        assert generator.config.num_questions == 10
        assert generator.config.max_hops == 4


class TestMultiHopQuestionGeneratorGenerate:
    """Tests for MultiHopQuestionGenerator.generate method."""

    @pytest.mark.asyncio
    async def test_generate_insufficient_documents(self, mock_llm: MagicMock) -> None:
        """Test generation with less than 2 documents."""
        generator = MultiHopQuestionGenerator(mock_llm)
        result = await generator.generate(documents=[Document(id="doc1", content="Single doc")])
        assert isinstance(result, TestSet)
        assert len(result.queries) == 0
        assert "at least 2 documents" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_empty_documents(self, mock_llm: MagicMock) -> None:
        """Test generation with empty document list."""
        generator = MultiHopQuestionGenerator(mock_llm)
        result = await generator.generate(documents=[])
        assert len(result.queries) == 0

    @pytest.mark.asyncio
    async def test_generate_documents_too_short(self, mock_llm: MagicMock) -> None:
        """Test generation when documents are too short."""
        generator = MultiHopQuestionGenerator(mock_llm)
        short_docs = [
            Document(id="doc1", content="Short"),
            Document(id="doc2", content="Also short"),
        ]
        result = await generator.generate(documents=short_docs)
        assert len(result.queries) == 0
        assert "not enough valid documents" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_no_relationships_found(
        self, mock_llm: MagicMock, unrelated_documents: list[Document]
    ) -> None:
        """Test generation when no relationships are found."""
        mock_llm.generate = AsyncMock(return_value='{"has_relationship": false}')

        generator = MultiHopQuestionGenerator(mock_llm)
        result = await generator.generate(documents=unrelated_documents)
        assert len(result.queries) == 0
        assert "no relationships found" in result.description.lower()

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_llm: MagicMock, related_documents: list[Document]) -> None:
        """Test successful multi-hop question generation."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                # Relationship check doc_alice + doc_acme
                '{"has_relationship": true, "relationship_type": "employment", "shared_entities": ["Acme Corporation"], "bridge_entity": "Acme Corporation"}',
                # Relationship check doc_alice + doc_paris
                '{"has_relationship": false}',
                # Relationship check doc_acme + doc_paris
                '{"has_relationship": true, "relationship_type": "location", "shared_entities": ["Paris"], "bridge_entity": "Paris"}',
                # Generate questions for alice+acme relationship
                '[{"question": "Where does Alice work?", "reasoning_chain": "Alice → Acme Corporation"}]',
                # Generate answer
                "Alice works at Acme Corporation, which is headquartered in Paris.",
                # Generate questions for acme+paris relationship
                '[{"question": "In which country is Acme Corporation located?", "reasoning_chain": "Acme Corp → Paris → France"}]',
                # Generate answer
                "Acme Corporation is located in France, as it is headquartered in Paris.",
            ]
        )

        generator = MultiHopQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=related_documents,
            num_questions=2,
        )

        assert isinstance(result, TestSet)
        assert len(result.queries) <= 2
        assert result.name == "multihop_testset"

        if result.queries:
            query = result.queries[0]
            assert query.metadata.get("question_type") == "multi_hop"
            assert "hop_count" in query.metadata
            assert len(query.ground_truth_docs) >= 2

    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(self, mock_llm: MagicMock, related_documents: list[Document]) -> None:
        """Test that generation handles LLM errors gracefully."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                # First relationship check fails
                Exception("LLM error"),
                # Second succeeds
                '{"has_relationship": true, "relationship_type": "location", "shared_entities": ["Paris"], "bridge_entity": "Paris"}',
                # Third fails
                Exception("LLM error"),
                # Question generation
                '[{"question": "Where is Acme Corp?", "reasoning_chain": "Acme → Paris"}]',
                # Answer
                "Paris",
            ]
        )

        generator = MultiHopQuestionGenerator(mock_llm)
        result = await generator.generate(documents=related_documents, num_questions=1)

        # Should still produce results despite some errors
        assert isinstance(result, TestSet)

    @pytest.mark.asyncio
    async def test_generate_with_max_hops_override(
        self, mock_llm: MagicMock, related_documents: list[Document]
    ) -> None:
        """Test generation with max_hops parameter override."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                # Relationship found
                '{"has_relationship": true, "relationship_type": "test", "shared_entities": [], "bridge_entity": "test"}',
                '{"has_relationship": false}',
                '{"has_relationship": false}',
                # Question generation
                '[{"question": "Test?", "reasoning_chain": "A → B"}]',
                # Answer
                "Test answer",
            ]
        )

        generator = MultiHopQuestionGenerator(mock_llm)
        result = await generator.generate(
            documents=related_documents,
            num_questions=5,
            max_hops=4,
        )

        assert result.metadata.get("max_hops") == 4


class TestMultiHopQuestionGeneratorRelationships:
    """Tests for relationship identification."""

    @pytest.mark.asyncio
    async def test_check_relationship_found(self, mock_llm: MagicMock) -> None:
        """Test successful relationship identification."""
        mock_llm.generate = AsyncMock(
            return_value='{"has_relationship": true, "relationship_type": "employment", "shared_entities": ["Company"], "bridge_entity": "Company"}'
        )

        generator = MultiHopQuestionGenerator(mock_llm)
        doc_a = Document(id="doc1", content="Alice works at Company X.")
        doc_b = Document(id="doc2", content="Company X is in Paris.")

        result = await generator._check_relationship(doc_a, doc_b)

        assert result is not None
        assert result.doc_a_id == "doc1"
        assert result.doc_b_id == "doc2"
        assert result.relationship_type == "employment"

    @pytest.mark.asyncio
    async def test_check_relationship_not_found(self, mock_llm: MagicMock) -> None:
        """Test when no relationship is found."""
        mock_llm.generate = AsyncMock(return_value='{"has_relationship": false}')

        generator = MultiHopQuestionGenerator(mock_llm)
        doc_a = Document(id="doc1", content="Weather is sunny.")
        doc_b = Document(id="doc2", content="Chocolate cake recipe.")

        result = await generator._check_relationship(doc_a, doc_b)

        assert result is None

    @pytest.mark.asyncio
    async def test_find_relationships_limits_comparisons(self, mock_llm: MagicMock) -> None:
        """Test that relationship finding limits comparisons."""
        mock_llm.generate = AsyncMock(return_value='{"has_relationship": false}')

        generator = MultiHopQuestionGenerator(mock_llm)
        # Create many documents
        docs = [Document(id=f"doc{i}", content=f"Content for document {i}" * 10) for i in range(20)]

        await generator._find_relationships(docs)

        # Should limit comparisons (20 docs = 190 pairs, but limited to 50)
        assert mock_llm.generate.call_count <= 50


class TestMultiHopQuestionGeneratorAnswers:
    """Tests for answer generation."""

    @pytest.mark.asyncio
    async def test_generate_answer(self, mock_llm: MagicMock) -> None:
        """Test answer generation for multi-hop question."""
        mock_llm.generate = AsyncMock(return_value="  Paris is the answer.  ")

        generator = MultiHopQuestionGenerator(mock_llm)
        doc_a = Document(id="doc1", content="Alice works at Acme.")
        doc_b = Document(id="doc2", content="Acme is in Paris.")

        result = await generator._generate_answer(doc_a, doc_b, "Where does Alice work?")

        assert result == "Paris is the answer."  # Stripped


class TestMultiHopQuestionGeneratorFromRelationship:
    """Tests for generating questions from relationships."""

    @pytest.mark.asyncio
    async def test_generate_from_relationship_success(self, mock_llm: MagicMock) -> None:
        """Test successful question generation from relationship."""
        mock_llm.generate = AsyncMock(
            side_effect=[
                '[{"question": "Where does Alice work?", "reasoning_chain": "Alice → Acme → Paris"}]',
                "Paris",
            ]
        )

        generator = MultiHopQuestionGenerator(mock_llm)
        doc_a = Document(id="doc1", content="Alice works at Acme Corporation.")
        doc_b = Document(id="doc2", content="Acme Corporation is in Paris.")
        relationship = DocumentRelationship(
            doc_a_id="doc1",
            doc_b_id="doc2",
            relationship_type="employment-location",
            bridge_entity="Acme Corporation",
        )

        result = await generator._generate_from_relationship(doc_a, doc_b, relationship, num_questions=1)

        assert len(result) == 1
        assert result[0].question == "Where does Alice work?"
        assert result[0].answer == "Paris"
        assert result[0].hop_chain == ["doc1", "doc2"]

    @pytest.mark.asyncio
    async def test_generate_from_relationship_invalid_json(self, mock_llm: MagicMock) -> None:
        """Test handling of invalid JSON in question generation."""
        mock_llm.generate = AsyncMock(return_value="not valid json")

        generator = MultiHopQuestionGenerator(mock_llm)
        doc_a = Document(id="doc1", content="Content A")
        doc_b = Document(id="doc2", content="Content B")
        relationship = DocumentRelationship(
            doc_a_id="doc1",
            doc_b_id="doc2",
            relationship_type="test",
        )

        result = await generator._generate_from_relationship(doc_a, doc_b, relationship, num_questions=1)

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_from_relationship_empty_question(self, mock_llm: MagicMock) -> None:
        """Test handling of empty question in response."""
        mock_llm.generate = AsyncMock(return_value='[{"question": "", "reasoning_chain": "test"}]')

        generator = MultiHopQuestionGenerator(mock_llm)
        doc_a = Document(id="doc1", content="Content A")
        doc_b = Document(id="doc2", content="Content B")
        relationship = DocumentRelationship(
            doc_a_id="doc1",
            doc_b_id="doc2",
            relationship_type="test",
        )

        result = await generator._generate_from_relationship(doc_a, doc_b, relationship, num_questions=1)

        assert result == []
