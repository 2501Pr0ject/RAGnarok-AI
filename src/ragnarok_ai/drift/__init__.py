"""Production drift detection.

Compare current production traffic (monitor traces) against a recorded
baseline and raise alerts when behavior shifts: distribution drift via
the Population Stability Index, and metric drift on success rate and
latency.
"""

from ragnarok_ai.drift.detector import TRACKED_FIELDS, DriftDetector, build_baseline
from ragnarok_ai.drift.models import (
    DistributionSnapshot,
    DriftBaseline,
    DriftFinding,
    DriftReport,
    DriftThresholds,
)

__all__ = [
    "TRACKED_FIELDS",
    "DistributionSnapshot",
    "DriftBaseline",
    "DriftDetector",
    "DriftFinding",
    "DriftReport",
    "DriftThresholds",
    "build_baseline",
]
