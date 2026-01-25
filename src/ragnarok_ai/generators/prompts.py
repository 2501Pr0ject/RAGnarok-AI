"""Prompts for test set generation.

This module contains prompt templates for generating synthetic
questions and answers from documents.
"""

from __future__ import annotations

# Prompt for generating questions from a document chunk
QUESTION_GENERATION_PROMPT = """You are an expert at creating test questions from documents.

Given the following document chunk, generate {num_questions} diverse questions that can be answered using ONLY the information in this chunk.

Document:
{document}

Requirements:
- Questions must be answerable from the document content only
- Include a mix of question types: {question_types}
- Questions should be clear and unambiguous
- Avoid yes/no questions when possible

Example input document:
"Paris is the capital of France with a population of 2.1 million."

Example output:
["What is the capital of France?", "What is the population of Paris?"]

Now generate questions for the document above.
Return a JSON array only, nothing else."""

# Prompt for generating an answer from a document
ANSWER_GENERATION_PROMPT = """You are an expert at answering questions based on provided context.

Given the following document and question, provide a concise and accurate answer.

Document:
{document}

Question:
{question}

Requirements:
- Answer ONLY based on the information in the document
- Be concise but complete
- If the answer cannot be found in the document, respond with "UNANSWERABLE"

Provide your answer as a single paragraph."""

# Prompt for validating question quality
QUESTION_VALIDATION_PROMPT = """You are an expert at evaluating question quality.

Given the following question and its source document, evaluate if this is a high-quality test question.

Document:
{document}

Question:
{question}

Evaluate the question on these criteria:
1. Answerable: Can this question be answered from the document?
2. Clear: Is the question unambiguous and well-formed?
3. Non-trivial: Does the question require understanding the content?
4. Specific: Does the question have a definite answer?

Return a JSON object with:
- "is_valid": true if the question passes all criteria, false otherwise
- "reason": brief explanation of your decision

Only return the JSON object, nothing else."""

# Question type descriptions for the generation prompt
QUESTION_TYPE_DESCRIPTIONS: dict[str, str] = {
    "factual": "factual questions (who, what, when, where)",
    "explanatory": "explanatory questions (how, why)",
    "comparative": "comparative questions (differences, similarities)",
    "definitional": "definitional questions (what is X, define Y)",
    # v0.3 - Multi-hop
    "multi_hop": "multi-hop questions (requiring reasoning across multiple facts)",
}


def get_question_types_description(types: list[str]) -> str:
    """Get a description string for the requested question types.

    Args:
        types: List of question type keys.

    Returns:
        Formatted description string for the prompt.
    """
    descriptions = [
        QUESTION_TYPE_DESCRIPTIONS[t]
        for t in types
        if t in QUESTION_TYPE_DESCRIPTIONS
    ]
    return ", ".join(descriptions) if descriptions else "various types"
