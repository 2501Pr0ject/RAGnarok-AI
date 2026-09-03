"""Root-cause diagnosis of RAG evaluation failures.

Turn evaluation scores into an actionable answer to "why is quality low":
each failing query is classified (retrieval miss, ranking, hallucination,
insufficient context...) and the report aggregates causes, per-question-type
patterns, and ordered recommendations.
"""

from ragnarok_ai.diagnosis.diagnostician import RAGDiagnostician
from ragnarok_ai.diagnosis.models import (
    DiagnosisReport,
    DiagnosisThresholds,
    FailureCause,
    QueryDiagnosis,
)

__all__ = [
    "DiagnosisReport",
    "DiagnosisThresholds",
    "FailureCause",
    "QueryDiagnosis",
    "RAGDiagnostician",
]
