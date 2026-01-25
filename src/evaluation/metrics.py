"""
Classification metrics and statistical testing for MURA-Finance Phase 2.

Uses numeric labels: -1 (Negative), 0 (Neutral), 1 (Positive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
except ImportError:
    accuracy_score = precision_recall_fscore_support = confusion_matrix = None

try:
    from scipy.stats import chi2 as chi2_dist
except ImportError:
    chi2_dist = None


LABELS = (-1, 0, 1)
LABEL_NAMES = {-1: "Negative", 0: "Neutral", 1: "Positive"}


def _align_valid(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: Tuple[int, ...] = LABELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align and filter to valid (finite, in labels) pairs. Returns (y_t, y_p)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    mask = (
        np.isfinite(y_true) & np.isin(y_true, labels)
        & np.isfinite(y_pred) & np.isin(y_pred, labels)
    )
    return y_true[mask].astype(int), y_pred[mask].astype(int)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: Tuple[int, ...] = LABELS,
    label_names: Dict[int, str] | None = None,
) -> Dict:
    """
    Compute accuracy, macro/micro precision, recall, F1, and per-class metrics.

    Drops rows where either y_true or y_pred is missing/invalid.
    """
    if accuracy_score is None or precision_recall_fscore_support is None:
        raise ImportError("scikit-learn is required for compute_classification_metrics")

    y_t, y_p = _align_valid(y_true, y_pred, labels)

    if len(y_t) == 0:
        return {"error": "No valid (y_true, y_pred) pairs after filtering"}

    acc = float(accuracy_score(y_t, y_p))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_t, y_p, labels=list(labels), average=None, zero_division=0
    )
    prec_macro = float(np.nanmean(prec))
    rec_macro = float(np.nanmean(rec))
    f1_macro = float(np.nanmean(f1))
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_t, y_p, labels=list(labels), average="micro", zero_division=0
    )

    label_names = label_names or LABEL_NAMES
    per_class = {}
    for i, lb in enumerate(labels):
        name = label_names.get(lb, str(lb))
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        }

    return {
        "n": int(len(y_t)),
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "f1_micro": float(f1_micro),
        "per_class": per_class,
    }


def confusion_matrix_dict(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: Tuple[int, ...] = LABELS,
    label_names: Dict[int, str] | None = None,
) -> Dict:
    """Return confusion matrix as a dict with row/col keys and counts."""
    if confusion_matrix is None:
        raise ImportError("scikit-learn is required for confusion_matrix_dict")

    y_t, y_p = _align_valid(y_true, y_pred, labels)

    cm = confusion_matrix(y_t, y_p, labels=list(labels))
    label_names = label_names or LABEL_NAMES
    rows = [label_names.get(lb, str(lb)) for lb in labels]
    cols = list(rows)
    out = {"labels": cols, "matrix": cm.tolist()}
    for i, r in enumerate(rows):
        out[r] = {c: int(cm[i, j]) for j, c in enumerate(cols)}
    return out


def mcnemar_test_pair(
    y_true: np.ndarray | pd.Series,
    y_pred_a: np.ndarray | pd.Series,
    y_pred_b: np.ndarray | pd.Series,
) -> Dict:
    """
    McNemar's test for two paired classifiers (same instances).
    Uses chi-squared approximation: χ² = (b - c)² / (b + c), 1 df.
    Returns stat, p_value, and simple interpretation.
    """
    if chi2_dist is None:
        return {"error": "scipy is required for mcnemar_test_pair"}

    y_true = np.asarray(y_true).ravel()
    ya = np.asarray(y_pred_a).ravel()
    yb = np.asarray(y_pred_b).ravel()
    n = min(len(y_true), len(ya), len(yb))
    y_true, ya, yb = y_true[:n], ya[:n], yb[:n]
    valid = (
        np.isfinite(y_true) & np.isin(np.round(y_true), (-1, 0, 1))
        & np.isfinite(ya) & np.isin(np.round(ya), (-1, 0, 1))
        & np.isfinite(yb) & np.isin(np.round(yb), (-1, 0, 1))
    )
    y_t = y_true[valid]
    ya = np.round(ya[valid]).astype(int)
    yb = np.round(yb[valid]).astype(int)

    correct_a = (ya == y_t)
    correct_b = (yb == y_t)
    both_wrong = int(np.sum((~correct_a) & (~correct_b)))
    a_wrong_b_right = int(np.sum((~correct_a) & correct_b))
    a_right_b_wrong = int(np.sum(correct_a & (~correct_b)))
    both_right = int(np.sum(correct_a & correct_b))
    table = [[both_wrong, a_wrong_b_right], [a_right_b_wrong, both_right]]
    b, c = a_wrong_b_right, a_right_b_wrong
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "contingency": table, "interpretation": "No discordant pairs; cannot test."}
    try:
        chi2 = (b - c) ** 2 / (b + c)
        p = float(1 - chi2_dist.cdf(chi2, df=1))
        stat = float(chi2)
    except Exception as e:
        return {"error": str(e), "contingency": table}

    return {
        "statistic": stat,
        "p_value": p,
        "contingency": table,
        "interpretation": "A vs B differ significantly (p < 0.05)"
        if p < 0.05
        else "No significant difference between A and B (p >= 0.05)",
    }
