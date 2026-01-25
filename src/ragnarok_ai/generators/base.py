"""Base protocol for question generators.

This module defines the protocol that all question generators must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragnarok_ai.core.types import Document, TestSet


@runtime_checkable
class QuestionGeneratorProtocol(Protocol):
    """Protocol for question generators.

    Any question generator must implement this protocol to be compatible
    with the RAGnarok evaluation framework.
    """

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
            num_questions: Number of questions to generate.
            question_types: Types of questions to generate.
            validate: Whether to validate generated questions.

        Returns:
            TestSet with generated queries.
        """
        ...
