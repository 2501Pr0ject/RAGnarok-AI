"""Adversarial question generator for ragnarok-ai.

This module provides adversarial question generation designed to expose
RAG weaknesses through various types of challenging questions.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.generators.parsing import parse_json_array

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol
    from ragnarok_ai.core.types import Document


class AdversarialType(str, Enum):
    """Types of adversarial questions."""

    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"
    MISLEADING = "misleading"
    BOUNDARY = "boundary"


class ExpectedBehavior(str, Enum):
    """Expected RAG system behavior for adversarial questions."""

    REFUSE = "refuse"  # Should say "I don't know" or refuse to answer
    CLARIFY = "clarify"  # Should ask for clarification
    CORRECT = "correct"  # Should correct the false premise
    HANDLE_GRACEFULLY = "handle_gracefully"  # Should not crash or error


# Mapping of adversarial types to expected behaviors
ADVERSARIAL_EXPECTED_BEHAVIOR: dict[AdversarialType, ExpectedBehavior] = {
    AdversarialType.OUT_OF_SCOPE: ExpectedBehavior.REFUSE,
    AdversarialType.AMBIGUOUS: ExpectedBehavior.CLARIFY,
    AdversarialType.MISLEADING: ExpectedBehavior.CORRECT,
    AdversarialType.BOUNDARY: ExpectedBehavior.HANDLE_GRACEFULLY,
}


# Prompt for generating out-of-scope questions
OUT_OF_SCOPE_PROMPT = """You are an expert at creating adversarial test questions.

Given the following documents, generate {num_questions} questions that CANNOT be answered using the information in these documents. These questions should be related to the general topic but ask about information not present.

Documents:
{documents}

Requirements:
- Questions should be plausible and related to the topic
- Questions must NOT be answerable from the documents
- Questions should sound natural, not obviously unanswerable
- Avoid questions that are too general or philosophical

Example: If documents are about "Paris landmarks", an out-of-scope question could be "What is the population of Lyon?" (related to France but not in the documents)

Return a JSON array of questions only, nothing else."""


# Prompt for generating ambiguous questions
AMBIGUOUS_PROMPT = """You are an expert at creating adversarial test questions.

Given the following documents, generate {num_questions} ambiguous questions that have multiple valid interpretations or could refer to different things mentioned in the documents.

Documents:
{documents}

Requirements:
- Questions should have at least 2 valid interpretations
- The ambiguity should be natural, not forced
- Questions should be related to the document content
- A good RAG system should ask for clarification

Example: If documents mention both "Python the snake" and "Python the language", an ambiguous question could be "How long has Python been around?"

Return a JSON array of questions only, nothing else."""


# Prompt for generating misleading questions
MISLEADING_PROMPT = """You are an expert at creating adversarial test questions.

Given the following documents, generate {num_questions} misleading questions that contain false premises or incorrect assumptions about the content.

Documents:
{documents}

Requirements:
- Questions should contain a false statement or assumption
- The false premise should be subtle but clearly wrong based on the documents
- A good RAG system should correct the false premise
- Questions should be related to the actual content

Example: If a document says "Paris was founded in 3rd century BC", a misleading question could be "Why was Paris founded in the 15th century?"

Return a JSON array of questions only, nothing else."""


# Prompt for generating boundary/edge case questions
BOUNDARY_PROMPT = """You are an expert at creating adversarial test questions.

Generate {num_questions} edge case questions that test the boundaries of a RAG system. These should be challenging inputs that might cause issues.

Types of boundary questions to generate:
- Very short queries (1-2 words)
- Very long queries (50+ words with multiple clauses)
- Questions with special characters or formatting
- Questions mixing multiple topics
- Questions with typos or unusual phrasing

Context documents (for reference):
{documents}

Return a JSON array of questions only, nothing else."""


class AdversarialQuestion(BaseModel):
    """An adversarial question with its type and expected behavior.

    Attributes:
        question: The adversarial question text.
        adversarial_type: Type of adversarial question.
        expected_behavior: How the RAG system should respond.
        explanation: Why this question is adversarial.
    """

    model_config = {"frozen": True}

    question: str = Field(..., description="The adversarial question")
    adversarial_type: AdversarialType = Field(..., description="Type of adversarial question")
    expected_behavior: ExpectedBehavior = Field(..., description="Expected RAG behavior")
    explanation: str = Field(default="", description="Why this is adversarial")


class AdversarialConfig(BaseModel):
    """Configuration for adversarial question generation.

    Attributes:
        num_questions: Total number of questions to generate.
        adversarial_types: Types of adversarial questions to generate.
        min_chunk_length: Minimum document length to consider.
    """

    num_questions: int = Field(default=20, ge=1, description="Total questions to generate")
    adversarial_types: list[AdversarialType] = Field(
        default_factory=lambda: list(AdversarialType),
        description="Types of adversarial questions",
    )
    min_chunk_length: int = Field(default=50, ge=10, description="Minimum chunk length")


class AdversarialQuestionGenerator:
    """Generates adversarial questions to test RAG system weaknesses.

    This generator creates challenging questions designed to expose
    potential issues in RAG pipelines.

    Attributes:
        llm: The LLM provider for generation.
        config: Generation configuration.

    Example:
        >>> from ragnarok_ai.adapters import OllamaLLM
        >>> async with OllamaLLM() as llm:
        ...     generator = AdversarialQuestionGenerator(llm)
        ...     testset = await generator.generate(
        ...         documents=docs,
        ...         num_questions=20,
        ...         adversarial_types=["out_of_scope", "ambiguous"],
        ...     )
    """

    def __init__(
        self,
        llm: LLMProtocol,
        config: AdversarialConfig | None = None,
    ) -> None:
        """Initialize AdversarialQuestionGenerator.

        Args:
            llm: The LLM provider implementing LLMProtocol.
            config: Optional generation configuration.
        """
        self.llm = llm
        self.config = config or AdversarialConfig()

    async def generate(
        self,
        documents: list[Document],
        num_questions: int | None = None,
        adversarial_types: list[str] | None = None,
    ) -> TestSet:
        """Generate adversarial questions from documents.

        Args:
            documents: Source documents for context.
            num_questions: Override for number of questions.
            adversarial_types: Override for adversarial types (as strings).

        Returns:
            TestSet with generated adversarial queries.
        """
        target_questions = num_questions or self.config.num_questions

        # Parse adversarial types
        types = [AdversarialType(t) for t in adversarial_types] if adversarial_types else self.config.adversarial_types

        if not documents:
            return TestSet(
                queries=[],
                name="adversarial_testset",
                description="Empty test set - no documents provided",
            )

        # Filter documents by minimum length
        valid_docs = [doc for doc in documents if len(doc.content) >= self.config.min_chunk_length]

        if not valid_docs:
            return TestSet(
                queries=[],
                name="adversarial_testset",
                description="Empty test set - no documents met minimum length",
            )

        # Generate questions for each adversarial type
        all_questions: list[AdversarialQuestion] = []
        questions_per_type = max(1, target_questions // len(types))

        for adv_type in types:
            if len(all_questions) >= target_questions:
                break

            try:
                type_questions = await self._generate_for_type(
                    valid_docs,
                    adv_type,
                    num_questions=questions_per_type,
                )
                all_questions.extend(type_questions)
            except Exception:
                continue

        # Limit to target number
        all_questions = all_questions[:target_questions]

        # Convert to TestSet format
        queries = [
            Query(
                text=q.question,
                ground_truth_docs=[],  # Adversarial questions may not have ground truth
                expected_answer=None,
                metadata={
                    "question_type": "adversarial",
                    "adversarial_type": q.adversarial_type.value,
                    "expected_behavior": q.expected_behavior.value,
                    "explanation": q.explanation,
                },
            )
            for q in all_questions
        ]

        return TestSet(
            queries=queries,
            name="adversarial_testset",
            description=f"Adversarial test set with {len(queries)} questions",
            metadata={
                "source_documents": len(documents),
                "adversarial_types": [t.value for t in types],
            },
        )

    async def _generate_for_type(
        self,
        documents: list[Document],
        adversarial_type: AdversarialType,
        num_questions: int,
    ) -> list[AdversarialQuestion]:
        """Generate adversarial questions of a specific type.

        Args:
            documents: Source documents for context.
            adversarial_type: Type of adversarial questions to generate.
            num_questions: Number of questions to generate.

        Returns:
            List of generated adversarial questions.
        """
        # Build documents context
        docs_text = "\n\n".join([f"Document {i + 1} ({d.id}):\n{d.content}" for i, d in enumerate(documents[:5])])

        # Select prompt based on type
        prompt_template = self._get_prompt_for_type(adversarial_type)
        prompt = prompt_template.format(
            num_questions=num_questions,
            documents=docs_text,
        )

        response = await self.llm.generate(prompt)
        questions = parse_json_array(response)

        # Convert to AdversarialQuestion objects
        expected_behavior = ADVERSARIAL_EXPECTED_BEHAVIOR[adversarial_type]
        generated: list[AdversarialQuestion] = []

        for q_text in questions[:num_questions]:
            if not isinstance(q_text, str) or not q_text.strip():
                continue

            generated.append(
                AdversarialQuestion(
                    question=q_text.strip(),
                    adversarial_type=adversarial_type,
                    expected_behavior=expected_behavior,
                    explanation=self._get_explanation_for_type(adversarial_type),
                )
            )

        return generated

    def _get_prompt_for_type(self, adversarial_type: AdversarialType) -> str:
        """Get the prompt template for an adversarial type.

        Args:
            adversarial_type: The adversarial type.

        Returns:
            Prompt template string.
        """
        prompts = {
            AdversarialType.OUT_OF_SCOPE: OUT_OF_SCOPE_PROMPT,
            AdversarialType.AMBIGUOUS: AMBIGUOUS_PROMPT,
            AdversarialType.MISLEADING: MISLEADING_PROMPT,
            AdversarialType.BOUNDARY: BOUNDARY_PROMPT,
        }
        return prompts[adversarial_type]

    def _get_explanation_for_type(self, adversarial_type: AdversarialType) -> str:
        """Get explanation for why a question type is adversarial.

        Args:
            adversarial_type: The adversarial type.

        Returns:
            Explanation string.
        """
        explanations = {
            AdversarialType.OUT_OF_SCOPE: "Question asks about information not present in the knowledge base",
            AdversarialType.AMBIGUOUS: "Question has multiple valid interpretations requiring clarification",
            AdversarialType.MISLEADING: "Question contains a false premise that should be corrected",
            AdversarialType.BOUNDARY: "Edge case that tests system robustness",
        }
        return explanations[adversarial_type]
