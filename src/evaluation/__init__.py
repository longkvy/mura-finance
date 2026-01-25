"""Evaluation utilities for MURA-Finance Phase 2 baseline assessment."""

from .metrics import (
    compute_classification_metrics,
    confusion_matrix_dict,
    mcnemar_test_pair,
)
from .comparison import (
    build_model_comparison_report,
)

__all__ = [
    "compute_classification_metrics",
    "confusion_matrix_dict",
    "mcnemar_test_pair",
    "build_model_comparison_report",
]
