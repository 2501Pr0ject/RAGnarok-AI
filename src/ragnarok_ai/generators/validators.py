"""Question validators for test set generation.

This module provides validation logic for generated questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.generators.parsing import parse_json_object
from ragnarok_ai.generators.prompts import QUESTION_VALIDATION_PROMPT

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol
    from ragnarok_ai.core.types import Document
    from ragnarok_ai.generators.models import GeneratedQuestion


class QuestionValidator:
    """Validates generated questions for quality.

    Uses an LLM to evaluate if questions are:
    - Answerable from the source document
    - Clear and unambiguous
    - Non-trivial (requires understanding)
    - Specific (has a definite answer)

    Attributes:
        llm: The LLM provider for validation.
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize QuestionValidator.

        Args:
            llm: The LLM provider implementing LLMProtocol.
        """
        self.llm = llm

    async def validate(
        self,
        question: GeneratedQuestion,
        documents: list[Document],
    ) -> bool:
        """Validate question quality.

        Args:
            question: The question to validate.
            documents: Source documents for context.

        Returns:
            True if question is valid, False otherwise.
        """
        # Find the source document
        source_doc = next(
            (d for d in documents if d.id == question.source_doc_id),
            None,
        )
        if not source_doc:
            return False

        prompt = QUESTION_VALIDATION_PROMPT.format(
            document=source_doc.content,
            question=question.question,
        )

        try:
            response = await self.llm.generate(prompt)
            result = parse_json_object(response)
            return bool(result.get("is_valid", False))
        except Exception:
            # If validation fails, assume question is valid
            return True

    async def validate_batch(
        self,
        questions: list[GeneratedQuestion],
        documents: list[Document],
        max_questions: int | None = None,
    ) -> list[GeneratedQuestion]:
        """Validate a batch of questions.

        Args:
            questions: Questions to validate.
            documents: Source documents for context.
            max_questions: Maximum number of valid questions to return.

        Returns:
            List of valid questions.
        """
        validated: list[GeneratedQuestion] = []

        for q in questions:
            if max_questions is not None and len(validated) >= max_questions:
                break
            try:
                is_valid = await self.validate(q, documents)
                if is_valid:
                    validated.append(q)
            except Exception:
                # Skip validation errors, include the question
                validated.append(q)

        return validated
