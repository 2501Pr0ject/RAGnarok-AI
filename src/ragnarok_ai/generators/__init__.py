"""Generators module for ragnarok-ai.

This module provides test set generation capabilities for RAG evaluation,
including synthetic question generation from documents.
"""

from __future__ import annotations

from ragnarok_ai.generators.adversarial import (
    AdversarialConfig,
    AdversarialQuestion,
    AdversarialQuestionGenerator,
    AdversarialType,
    ExpectedBehavior,
)
from ragnarok_ai.generators.base import QuestionGeneratorProtocol
from ragnarok_ai.generators.golden import (
    GoldenQuestion,
    GoldenSet,
    GoldenSetDiff,
)
from ragnarok_ai.generators.io import load_testset, save_testset
from ragnarok_ai.generators.models import GeneratedQuestion, GenerationConfig
from ragnarok_ai.generators.multihop import (
    DocumentRelationship,
    MultiHopConfig,
    MultiHopQuestion,
    MultiHopQuestionGenerator,
)
from ragnarok_ai.generators.synthetic import SyntheticQuestionGenerator
from ragnarok_ai.generators.validators import QuestionValidator

# Backward compatibility alias
TestSetGenerator = SyntheticQuestionGenerator

__all__ = [
    "AdversarialConfig",
    "AdversarialQuestion",
    "AdversarialQuestionGenerator",
    "AdversarialType",
    "DocumentRelationship",
    "ExpectedBehavior",
    "GeneratedQuestion",
    "GenerationConfig",
    "GoldenQuestion",
    "GoldenSet",
    "GoldenSetDiff",
    "MultiHopConfig",
    "MultiHopQuestion",
    "MultiHopQuestionGenerator",
    "QuestionGeneratorProtocol",
    "QuestionValidator",
    "SyntheticQuestionGenerator",
    "TestSetGenerator",
    "load_testset",
    "save_testset",
]
