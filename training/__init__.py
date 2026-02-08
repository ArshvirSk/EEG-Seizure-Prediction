"""
Training package for EEG seizure prediction.
Contains training logic and evaluation metrics.
"""

from .trainer import Trainer, EarlyStoppingCallback
from .metrics import (
    compute_metrics,
    compute_false_prediction_rate,
    print_classification_report,
)

__all__ = [
    "Trainer",
    "EarlyStoppingCallback",
    "compute_metrics",
    "compute_false_prediction_rate",
    "print_classification_report",
]
