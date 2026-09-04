"""ragnarok-ai: A local-first RAG evaluation framework for LLM applications."""

from __future__ import annotations

from ragnarok_ai.ab import ABAnalyzer, ABTestConfig, ABTestReport, Experiment
from ragnarok_ai.alerts import AlertManager, AlertRule, AlertSeverity
from ragnarok_ai.calibration import CalibrationSample, CalibrationSet, JudgeCalibrator
from ragnarok_ai.core.batch import BatchConfig, BatchEvaluator, BatchProgress, BatchResult
from ragnarok_ai.core.compare import ComparisonResult, compare
from ragnarok_ai.core.evaluate import EvaluationResult, evaluate, evaluate_stream
from ragnarok_ai.cost.tracker import CostSummary, CostTracker
from ragnarok_ai.diagnosis import (
    DiagnosisReport,
    DiagnosisThresholds,
    FailureCause,
    RAGDiagnostician,
)
from ragnarok_ai.drift import DriftBaseline, DriftDetector, DriftReport, DriftThresholds, build_baseline
from ragnarok_ai.evaluators.judge import JudgeResult, JudgeResults, LLMJudge
from ragnarok_ai.loaders.forge_bundle import (
    ForgeLoadError,
    load_forge_bundle,
    load_forge_documents,
)
from ragnarok_ai.mining import TestsetMiner
from ragnarok_ai.monitor import MonitorClient
from ragnarok_ai.privacy import PiiMode, sanitize_dict, sanitize_value

__version__ = "1.11.0"
__all__ = [
    "ABAnalyzer",
    "ABTestConfig",
    "ABTestReport",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "BatchConfig",
    "BatchEvaluator",
    "BatchProgress",
    "BatchResult",
    "CalibrationSample",
    "CalibrationSet",
    "ComparisonResult",
    "CostSummary",
    "CostTracker",
    "DiagnosisReport",
    "DiagnosisThresholds",
    "DriftBaseline",
    "DriftDetector",
    "DriftReport",
    "DriftThresholds",
    "EvaluationResult",
    "Experiment",
    "FailureCause",
    "ForgeLoadError",
    "JudgeCalibrator",
    "JudgeResult",
    "JudgeResults",
    "LLMJudge",
    "MonitorClient",
    "PiiMode",
    "RAGDiagnostician",
    "TestsetMiner",
    "__version__",
    "build_baseline",
    "compare",
    "evaluate",
    "evaluate_stream",
    "load_forge_bundle",
    "load_forge_documents",
    "sanitize_dict",
    "sanitize_value",
]
