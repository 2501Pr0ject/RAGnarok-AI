"""Multi-hop question generator for ragnarok-ai.

This module provides multi-hop question generation that creates
questions requiring reasoning across multiple documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.generators.parsing import parse_json_array, parse_json_object

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol
    from ragnarok_ai.core.types import Document


# Prompt to identify relationships between documents
RELATIONSHIP_IDENTIFICATION_PROMPT = """You are an expert at identifying relationships between documents.

Given the following two documents, identify if there is a meaningful relationship that could be used to create a multi-hop question.

Document A (ID: {doc_a_id}):
{doc_a_content}

Document B (ID: {doc_b_id}):
{doc_b_content}

A meaningful relationship exists if:
- One document mentions an entity that is described in the other
- The documents share common entities (people, places, organizations)
- Information from both documents is needed to answer a question

Return a JSON object with:
- "has_relationship": true/false
- "relationship_type": brief description of the relationship (or null if none)
- "shared_entities": list of shared entities (or empty list)
- "bridge_entity": the main entity connecting the documents (or null)

Only return the JSON object, nothing else."""


# Prompt to generate multi-hop questions
MULTIHOP_QUESTION_PROMPT = """You are an expert at creating multi-hop questions that require reasoning across multiple documents.

Given the following documents and their relationship, generate {num_questions} questions that:
- Require information from BOTH documents to answer
- Cannot be answered using only one document
- Have a clear reasoning chain from one document to another

Documents:
{documents}

Relationship: {relationship}
Bridge entity: {bridge_entity}

Example of a good multi-hop question:
- Documents: "Alice works at Acme Corp" + "Acme Corp is in Paris"
- Question: "In which city does Alice work?"
- Reasoning: Alice → Acme Corp → Paris

Generate questions that follow this pattern. Return a JSON array of objects with:
- "question": the multi-hop question
- "reasoning_chain": brief explanation of the hops required

Only return the JSON array, nothing else."""


# Prompt to generate answer for multi-hop question
MULTIHOP_ANSWER_PROMPT = """You are an expert at answering multi-hop questions using multiple documents.

Given the following documents and question, provide a concise answer by reasoning across the documents.

Documents:
{documents}

Question: {question}

Requirements:
- Use information from multiple documents to construct the answer
- Follow the reasoning chain across documents
- Be concise but complete

Provide your answer as a single paragraph."""


class DocumentRelationship(BaseModel):
    """Represents a relationship between two documents.

    Attributes:
        doc_a_id: ID of the first document.
        doc_b_id: ID of the second document.
        relationship_type: Description of the relationship.
        shared_entities: List of entities shared between documents.
        bridge_entity: Main entity connecting the documents.
    """

    model_config = {"frozen": True}

    doc_a_id: str = Field(..., description="First document ID")
    doc_b_id: str = Field(..., description="Second document ID")
    relationship_type: str = Field(..., description="Type of relationship")
    shared_entities: list[str] = Field(default_factory=list, description="Shared entities")
    bridge_entity: str | None = Field(default=None, description="Bridge entity")


class MultiHopQuestion(BaseModel):
    """A multi-hop question with its reasoning chain.

    Attributes:
        question: The multi-hop question text.
        answer: The generated answer.
        hop_chain: List of document IDs in the reasoning chain.
        reasoning_steps: Explanation of each reasoning hop.
    """

    model_config = {"frozen": True}

    question: str = Field(..., description="The multi-hop question")
    answer: str = Field(..., description="The generated answer")
    hop_chain: list[str] = Field(..., description="Document IDs in reasoning chain")
    reasoning_steps: str = Field(..., description="Explanation of reasoning hops")


class MultiHopConfig(BaseModel):
    """Configuration for multi-hop question generation.

    Attributes:
        num_questions: Total number of questions to generate.
        max_hops: Maximum number of hops (documents) per question.
        min_hops: Minimum number of hops per question.
        min_chunk_length: Minimum document length to consider.
    """

    num_questions: int = Field(default=20, ge=1, description="Total questions to generate")
    max_hops: int = Field(default=3, ge=2, le=5, description="Maximum hops per question")
    min_hops: int = Field(default=2, ge=2, description="Minimum hops per question")
    min_chunk_length: int = Field(default=50, ge=10, description="Minimum chunk length")


class MultiHopQuestionGenerator:
    """Generates multi-hop questions requiring reasoning across documents.

    This generator creates questions that require information from multiple
    documents, tracking the reasoning chain for evaluation.

    Attributes:
        llm: The LLM provider for generation.
        config: Generation configuration.

    Example:
        >>> from ragnarok_ai.adapters import OllamaLLM
        >>> async with OllamaLLM() as llm:
        ...     generator = MultiHopQuestionGenerator(llm)
        ...     testset = await generator.generate(
        ...         documents=docs,
        ...         num_questions=20,
        ...         max_hops=3,
        ...     )
    """

    def __init__(
        self,
        llm: LLMProtocol,
        config: MultiHopConfig | None = None,
    ) -> None:
        """Initialize MultiHopQuestionGenerator.

        Args:
            llm: The LLM provider implementing LLMProtocol.
            config: Optional generation configuration.
        """
        self.llm = llm
        self.config = config or MultiHopConfig()

    async def generate(
        self,
        documents: list[Document],
        num_questions: int | None = None,
        max_hops: int | None = None,
    ) -> TestSet:
        """Generate multi-hop questions from documents.

        Args:
            documents: Source documents to generate questions from.
            num_questions: Override for number of questions.
            max_hops: Override for maximum hops.

        Returns:
            TestSet with generated multi-hop queries.
        """
        target_questions = num_questions or self.config.num_questions
        max_hop_count = max_hops or self.config.max_hops

        if len(documents) < 2:
            return TestSet(
                queries=[],
                name="multihop_testset",
                description="Empty test set - need at least 2 documents for multi-hop",
            )

        # Filter documents by minimum length
        valid_docs = [doc for doc in documents if len(doc.content) >= self.config.min_chunk_length]

        if len(valid_docs) < 2:
            return TestSet(
                queries=[],
                name="multihop_testset",
                description="Empty test set - not enough valid documents",
            )

        # Find relationships between document pairs
        relationships = await self._find_relationships(valid_docs)

        if not relationships:
            return TestSet(
                queries=[],
                name="multihop_testset",
                description="Empty test set - no relationships found between documents",
            )

        # Generate multi-hop questions from relationships
        all_questions: list[MultiHopQuestion] = []
        questions_per_relationship = max(1, target_questions // len(relationships))

        for rel in relationships:
            if len(all_questions) >= target_questions:
                break

            try:
                # Get the documents for this relationship
                doc_a = next((d for d in valid_docs if d.id == rel.doc_a_id), None)
                doc_b = next((d for d in valid_docs if d.id == rel.doc_b_id), None)

                if not doc_a or not doc_b:
                    continue

                questions = await self._generate_from_relationship(
                    doc_a,
                    doc_b,
                    rel,
                    num_questions=min(questions_per_relationship, 3),
                )
                all_questions.extend(questions)
            except Exception:
                continue

        # Limit to target number
        all_questions = all_questions[:target_questions]

        # Convert to TestSet format
        queries = [
            Query(
                text=q.question,
                ground_truth_docs=q.hop_chain,
                expected_answer=q.answer,
                metadata={
                    "question_type": "multi_hop",
                    "hop_count": len(q.hop_chain),
                    "reasoning_steps": q.reasoning_steps,
                },
            )
            for q in all_questions
        ]

        return TestSet(
            queries=queries,
            name="multihop_testset",
            description=f"Multi-hop test set with {len(queries)} questions",
            metadata={
                "source_documents": len(documents),
                "relationships_found": len(relationships),
                "max_hops": max_hop_count,
            },
        )

    async def _find_relationships(
        self,
        documents: list[Document],
    ) -> list[DocumentRelationship]:
        """Find relationships between document pairs.

        Args:
            documents: List of documents to analyze.

        Returns:
            List of identified relationships.
        """
        relationships: list[DocumentRelationship] = []

        # Check pairs of documents for relationships
        # Limit to reasonable number of comparisons
        max_comparisons = min(50, len(documents) * (len(documents) - 1) // 2)
        comparisons_done = 0

        for i, doc_a in enumerate(documents):
            for doc_b in documents[i + 1 :]:
                if comparisons_done >= max_comparisons:
                    break

                try:
                    rel = await self._check_relationship(doc_a, doc_b)
                    if rel:
                        relationships.append(rel)
                except Exception:
                    continue

                comparisons_done += 1

            if comparisons_done >= max_comparisons:
                break

        return relationships

    async def _check_relationship(
        self,
        doc_a: Document,
        doc_b: Document,
    ) -> DocumentRelationship | None:
        """Check if two documents have a meaningful relationship.

        Args:
            doc_a: First document.
            doc_b: Second document.

        Returns:
            DocumentRelationship if found, None otherwise.
        """
        prompt = RELATIONSHIP_IDENTIFICATION_PROMPT.format(
            doc_a_id=doc_a.id,
            doc_a_content=doc_a.content,
            doc_b_id=doc_b.id,
            doc_b_content=doc_b.content,
        )

        response = await self.llm.generate(prompt)
        result = parse_json_object(response)

        if not result.get("has_relationship", False):
            return None

        return DocumentRelationship(
            doc_a_id=doc_a.id,
            doc_b_id=doc_b.id,
            relationship_type=result.get("relationship_type", "unknown"),
            shared_entities=result.get("shared_entities", []),
            bridge_entity=result.get("bridge_entity"),
        )

    async def _generate_from_relationship(
        self,
        doc_a: Document,
        doc_b: Document,
        relationship: DocumentRelationship,
        num_questions: int,
    ) -> list[MultiHopQuestion]:
        """Generate multi-hop questions from a document relationship.

        Args:
            doc_a: First document.
            doc_b: Second document.
            relationship: The relationship between documents.
            num_questions: Number of questions to generate.

        Returns:
            List of generated multi-hop questions.
        """
        documents_text = (
            f"Document 1 (ID: {doc_a.id}):\n{doc_a.content}\n\nDocument 2 (ID: {doc_b.id}):\n{doc_b.content}"
        )

        prompt = MULTIHOP_QUESTION_PROMPT.format(
            num_questions=num_questions,
            documents=documents_text,
            relationship=relationship.relationship_type,
            bridge_entity=relationship.bridge_entity or "shared context",
        )

        response = await self.llm.generate(prompt)
        questions_data = parse_json_array(response)

        generated: list[MultiHopQuestion] = []
        for q_data in questions_data[:num_questions]:
            if not isinstance(q_data, dict):
                continue

            question_text = q_data.get("question", "")
            if not question_text:
                continue

            try:
                # Generate answer
                answer = await self._generate_answer(doc_a, doc_b, question_text)
                if not answer:
                    continue

                generated.append(
                    MultiHopQuestion(
                        question=question_text,
                        answer=answer,
                        hop_chain=[doc_a.id, doc_b.id],
                        reasoning_steps=q_data.get("reasoning_chain", ""),
                    )
                )
            except Exception:
                continue

        return generated

    async def _generate_answer(
        self,
        doc_a: Document,
        doc_b: Document,
        question: str,
    ) -> str:
        """Generate answer for a multi-hop question.

        Args:
            doc_a: First document.
            doc_b: Second document.
            question: The multi-hop question.

        Returns:
            Generated answer.
        """
        documents_text = f"Document 1:\n{doc_a.content}\n\nDocument 2:\n{doc_b.content}"

        prompt = MULTIHOP_ANSWER_PROMPT.format(
            documents=documents_text,
            question=question,
        )

        response = await self.llm.generate(prompt)
        return response.strip()
