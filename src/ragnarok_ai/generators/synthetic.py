"""Synthetic question generator for ragnarok-ai.

This module provides synthetic test set generation from documents
using local LLMs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.generators.models import GeneratedQuestion, GenerationConfig
from ragnarok_ai.generators.parsing import parse_json_array
from ragnarok_ai.generators.prompts import (
    ANSWER_GENERATION_PROMPT,
    QUESTION_GENERATION_PROMPT,
    get_question_types_description,
)
from ragnarok_ai.generators.validators import QuestionValidator

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol
    from ragnarok_ai.core.types import Document


class SyntheticQuestionGenerator:
    """Generates synthetic test sets from documents using LLMs.

    This generator creates diverse test questions and answers from
    provided documents, suitable for evaluating RAG pipelines.

    Attributes:
        llm: The LLM provider for generation.
        config: Generation configuration.
        validator: Question validator instance.

    Example:
        >>> from ragnarok_ai.adapters import OllamaLLM
        >>> async with OllamaLLM() as llm:
        ...     generator = SyntheticQuestionGenerator(llm)
        ...     testset = await generator.generate(
        ...         documents=docs,
        ...         num_questions=50,
        ...     )
        ...     save_testset(testset, "testset.json")
    """

    def __init__(
        self,
        llm: LLMProtocol,
        config: GenerationConfig | None = None,
    ) -> None:
        """Initialize SyntheticQuestionGenerator.

        Args:
            llm: The LLM provider implementing LLMProtocol.
            config: Optional generation configuration.
        """
        self.llm = llm
        self.config = config or GenerationConfig()
        self.validator = QuestionValidator(llm)

    async def generate(
        self,
        documents: list[Document],
        num_questions: int | None = None,
        question_types: list[str] | None = None,
        validate: bool | None = None,
    ) -> TestSet:
        """Generate a test set from documents.

        Args:
            documents: Source documents to generate questions from.
            num_questions: Override for number of questions.
            question_types: Override for question types.
            validate: Override for question validation.

        Returns:
            TestSet with generated queries.

        Raises:
            EvaluationError: If generation fails.
        """
        # Apply overrides
        target_questions = num_questions or self.config.num_questions
        types = question_types or self.config.question_types
        should_validate = validate if validate is not None else self.config.validate_questions

        if not documents:
            return TestSet(
                queries=[],
                name="generated_testset",
                description="Empty test set - no documents provided",
            )

        # Filter documents by minimum length
        valid_docs = [doc for doc in documents if len(doc.content) >= self.config.min_chunk_length]

        if not valid_docs:
            return TestSet(
                queries=[],
                name="generated_testset",
                description="Empty test set - no documents met minimum length",
            )

        # Generate questions from each document
        all_questions: list[GeneratedQuestion] = []
        questions_per_doc = max(1, target_questions // len(valid_docs))

        for doc in valid_docs:
            if len(all_questions) >= target_questions:
                break

            try:
                doc_questions = await self._generate_from_document(
                    doc,
                    num_questions=min(questions_per_doc, self.config.questions_per_chunk),
                    question_types=types,
                )
                all_questions.extend(doc_questions)
            except Exception:
                # Skip documents that fail, continue with others
                continue

        # Validate questions if enabled
        if should_validate:
            all_questions = await self.validator.validate_batch(
                all_questions,
                documents,
                max_questions=target_questions,
            )

        # Limit to target number
        all_questions = all_questions[:target_questions]

        # Convert to TestSet format
        queries = [
            Query(
                text=q.question,
                ground_truth_docs=[q.source_doc_id],
                expected_answer=q.answer,
                metadata={"question_type": q.question_type},
            )
            for q in all_questions
        ]

        return TestSet(
            queries=queries,
            name="generated_testset",
            description=f"Synthetic test set with {len(queries)} questions",
            metadata={
                "source_documents": len(documents),
                "question_types": types,
                "validated": should_validate,
            },
        )

    async def _generate_from_document(
        self,
        document: Document,
        num_questions: int,
        question_types: list[str],
    ) -> list[GeneratedQuestion]:
        """Generate questions from a single document.

        Args:
            document: Source document.
            num_questions: Number of questions to generate.
            question_types: Types of questions to generate.

        Returns:
            List of generated questions.
        """
        types_desc = get_question_types_description(question_types)

        # Generate questions
        prompt = QUESTION_GENERATION_PROMPT.format(
            document=document.content,
            num_questions=num_questions,
            question_types=types_desc,
        )

        try:
            response = await self.llm.generate(prompt)
            questions = parse_json_array(response)
        except Exception as e:
            msg = f"Failed to generate questions: {e}"
            raise EvaluationError(msg) from e

        if not questions:
            return []

        # Generate answers for each question
        generated: list[GeneratedQuestion] = []
        for q_text in questions[:num_questions]:
            if not isinstance(q_text, str) or not q_text.strip():
                continue

            try:
                answer = await self._generate_answer(document, q_text)
                if answer and answer != "UNANSWERABLE":
                    # Determine question type based on keywords
                    q_type = self._classify_question_type(q_text, question_types)
                    generated.append(
                        GeneratedQuestion(
                            question=q_text.strip(),
                            answer=answer.strip(),
                            source_doc_id=document.id,
                            question_type=q_type,
                        )
                    )
            except Exception:
                # Skip questions that fail answer generation
                continue

        return generated

    async def _generate_answer(self, document: Document, question: str) -> str:
        """Generate an answer for a question from a document.

        Args:
            document: Source document.
            question: The question to answer.

        Returns:
            Generated answer.
        """
        prompt = ANSWER_GENERATION_PROMPT.format(
            document=document.content,
            question=question,
        )

        response = await self.llm.generate(prompt)
        return response.strip()

    def _classify_question_type(self, question: str, allowed_types: list[str]) -> str:
        """Classify a question into one of the allowed types.

        Args:
            question: The question text.
            allowed_types: List of allowed question types.

        Returns:
            The classified question type.
        """
        question_lower = question.lower()

        # Multi-hop questions typically involve multiple entities or steps
        if "multi_hop" in allowed_types and any(
            phrase in question_lower for phrase in ["and also", "in addition", "furthermore", "both"]
        ):
            return "multi_hop"

        if "explanatory" in allowed_types and any(word in question_lower for word in ["how", "why", "explain"]):
            return "explanatory"

        if "comparative" in allowed_types and any(
            word in question_lower for word in ["difference", "compare", "versus", "vs"]
        ):
            return "comparative"

        if "definitional" in allowed_types and any(
            word in question_lower for word in ["what is", "define", "meaning of"]
        ):
            return "definitional"

        if "factual" in allowed_types:
            return "factual"

        return allowed_types[0] if allowed_types else "factual"
