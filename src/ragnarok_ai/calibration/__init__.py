"""Judge calibration against human labels.

Measure how much you can trust your LLM judge: run it on a small
human-labeled sample set and get agreement statistics (Cohen's kappa,
false accept/reject rates, a recommended pass threshold) per criterion.
"""

from ragnarok_ai.calibration.calibrator import JudgeCalibrator
from ragnarok_ai.calibration.models import (
    CalibrationReport,
    CalibrationSample,
    CalibrationSet,
    CriterionCalibration,
)

__all__ = [
    "CalibrationReport",
    "CalibrationSample",
    "CalibrationSet",
    "CriterionCalibration",
    "JudgeCalibrator",
]
