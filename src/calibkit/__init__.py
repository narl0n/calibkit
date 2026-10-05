"""calibkit: measure and improve probability calibration of binary classifiers."""

from .metrics import (
    ReliabilityBins,
    brier_score,
    expected_calibration_error,
    max_calibration_error,
    reliability_bins,
)
from .scaling import TemperatureScaler

__version__ = "0.1.0"

__all__ = [
    "ReliabilityBins",
    "TemperatureScaler",
    "__version__",
    "brier_score",
    "expected_calibration_error",
    "max_calibration_error",
    "reliability_bins",
]
