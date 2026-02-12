"""Medical domain evaluation utilities.

This package provides medical-specific evaluation tools including
abbreviation normalization and medical terminology handling.
"""

from ragnarok_ai.evaluators.medical.medical_normalizer import MedicalAbbreviationNormalizer

__all__ = [
    "MedicalAbbreviationNormalizer",
]
