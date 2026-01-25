"""Data models for test set generation.

This module contains Pydantic models for generated questions
and generation configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedQuestion(BaseModel):
    """A generated question with its source and answer.

    Attributes:
        question: The generated question text.
        answer: The generated answer.
        source_doc_id: ID of the source document.
        question_type: Type of question (factual, explanatory, etc.).
        is_valid: Whether the question passed quality validation.
    """

    model_config = {"frozen": True}

    question: str = Field(..., description="The generated question")
    answer: str = Field(..., description="The generated answer")
    source_doc_id: str = Field(..., description="Source document ID")
    question_type: str = Field(default="factual", description="Type of question")
    is_valid: bool = Field(default=True, description="Quality validation result")


class GenerationConfig(BaseModel):
    """Configuration for test set generation.

    Attributes:
        num_questions: Total number of questions to generate.
        question_types: Types of questions to generate.
        questions_per_chunk: Max questions per document chunk.
        validate_questions: Whether to validate question quality.
        min_chunk_length: Minimum chunk length to consider.
    """

    num_questions: int = Field(default=50, ge=1, description="Total questions to generate")
    question_types: list[str] = Field(
        default_factory=lambda: ["factual", "explanatory"],
        description="Types of questions to generate",
    )
    questions_per_chunk: int = Field(default=3, ge=1, description="Max questions per chunk")
    validate_questions: bool = Field(default=True, description="Validate question quality")
    min_chunk_length: int = Field(default=100, ge=10, description="Minimum chunk length")
