"""
Model comparison and evaluation runners for Phase 2 baseline evaluation.

Uses single_article and allday CSVs. All predictions normalized to -1/0/1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .metrics import (
    compute_classification_metrics,
    confusion_matrix_dict,
    mcnemar_test_pair,
)

# Model column -> (prediction column, is_continuous)
# Continuous _n models: we threshold with continuous_to_discrete
SINGLE_ARTICLE_MODELS = [
    ("finbert", "finbert_sentiment", False),
    ("finbert_a", "finbert_sentiment_a", False),
    ("gpt_p1", "gpt_sentiment_p1", False),
    ("gpt_p2", "gpt_sentiment_p2", False),
    ("gpt_p3", "gpt_sentiment_p3", False),
    ("gpt_p4", "gpt_sentiment_p4", False),
    ("gpt_p7", "gpt_sentiment_p7", False),
    ("gpt_p1n", "gpt_sentiment_p1n", True),
    ("gpt_p2n", "gpt_sentiment_p2n", True),
    ("gpt_p3n", "gpt_sentiment_p3n", True),
    ("gpt_p4n", "gpt_sentiment_p4n", True),
    ("gpt_p7n", "gpt_sentiment_p7n", True),
]

ALLDAY_MODELS = [
    ("gpt_p5", "gpt_sentiment_p5", False),
    ("gpt_p6", "gpt_sentiment_p6", False),
    ("gpt_p5n", "gpt_sentiment_p5n", True),
    ("gpt_p6n", "gpt_sentiment_p6n", True),
]

# Approx $/1K tokens (GPT-3.5 Turbo–like). Adjust as needed.
INPUT_COST_PER_1K = 0.0015
OUTPUT_COST_PER_1K = 0.002


def _continuous_to_discrete(x: float, neg: float = -0.33, pos: float = 0.33) -> int:
    if np.isnan(x) or not np.isfinite(x):
        return int(0)
    if x <= neg:
        return -1
    if x >= pos:
        return 1
    return 0


def _get_single_article_pred(
    df: pd.DataFrame,
    col: str,
    continuous: bool,
) -> np.ndarray:
    s = df[col].values
    if continuous:
        out = np.array([_continuous_to_discrete(float(v)) for v in s], dtype=float)
    else:
        out = np.asarray(s, dtype=float)
        valid = np.isfinite(out) & np.isin(np.round(out), (-1, 0, 1))
        out = np.where(valid, np.round(out).astype(int).astype(float), np.nan)
    return out


def evaluate_single_article_models(
    df: pd.DataFrame,
    *,
    y_true_col: str = "true_sentiment",
) -> Dict[str, Any]:
    """
    Evaluate all single-article models. Returns metrics + confusion matrices per model.
    """
    y_true = np.asarray(df[y_true_col], dtype=float)
    results = {}
    for name, col, continuous in SINGLE_ARTICLE_MODELS:
        if col not in df.columns:
            results[name] = {"error": f"Column {col} not found"}
            continue
        y_pred = _get_single_article_pred(df, col, continuous)
        metrics = compute_classification_metrics(y_true, y_pred)
        if "error" in metrics:
            results[name] = metrics
            continue
        cm = confusion_matrix_dict(y_true, y_pred)
        results[name] = {"metrics": metrics, "confusion_matrix": cm}
    return results


def _daily_ground_truth_from_single(sa: pd.DataFrame) -> pd.DataFrame:
    """Build (date, ticker) -> mode(true_sentiment) from single_article."""
    sa = sa.copy()
    sa["date"] = pd.to_datetime(sa["published_at"], errors="coerce").dt.date
    g = sa.groupby(["date", "ticker"])["true_sentiment"].apply(
        lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan
    )
    return g.reset_index().rename(columns={"true_sentiment": "daily_true"})


def evaluate_allday_models(
    sa: pd.DataFrame,
    allday: pd.DataFrame,
    *,
    y_true_source: str = "true_sentiment",
) -> Dict[str, Any]:
    """
    Evaluate p5/p6/p5n/p6n against daily-aggregated ground truth.
    Daily GT = mode of article-level true_sentiment per (date, ticker).
    """
    daily_gt = _daily_ground_truth_from_single(sa)
    ad = allday.copy()
    ad["date"] = pd.to_datetime(ad["published_at"], errors="coerce").dt.date
    merged = ad.merge(daily_gt, on=["date", "ticker"], how="inner")
    merged = merged.dropna(subset=["daily_true"])
    y_true = np.asarray(merged["daily_true"], dtype=float)

    results = {}
    for name, col, continuous in ALLDAY_MODELS:
        if col not in merged.columns:
            results[name] = {"error": f"Column {col} not found"}
            continue
        s = merged[col]
        if continuous:
            y_pred = np.array([_continuous_to_discrete(v) for v in s.values], dtype=float)
        else:
            y_pred = np.asarray(s, dtype=float)
            ok = np.isfinite(y_pred) & np.isin(np.round(y_pred), (-1, 0, 1))
            y_pred = np.where(ok, np.round(y_pred), np.nan)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t = y_true[valid].astype(int)
        y_p = np.round(y_pred[valid]).astype(int)
        if len(y_t) == 0:
            results[name] = {"error": "No valid pairs after merge"}
            continue
        metrics = compute_classification_metrics(y_t, y_p)
        if "error" in metrics:
            results[name] = metrics
            continue
        cm = confusion_matrix_dict(y_t, y_p)
        results[name] = {"metrics": metrics, "confusion_matrix": cm, "n_days": int(len(merged))}
    return results


def _token_columns_single(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    """(model_name, prompt_col, completion_col, time_col) for single-article."""
    out = []
    for name, pred_col, _ in SINGLE_ARTICLE_MODELS:
        if "gpt_" not in name:
            continue
        base = pred_col.replace("gpt_sentiment_", "")
        p = f"gpt_prompt_tokens_{base}"
        c = f"gpt_completion_tokens_{base}"
        t = f"gpt_time_{base}"
        if p in df.columns and c in df.columns and t in df.columns:
            out.append((name, p, c, t))
    return out


def _token_columns_allday(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    """(model_name, prompt_col, completion_col, time_col)."""
    out = []
    for name, pred_col, _ in ALLDAY_MODELS:
        base = pred_col.replace("gpt_sentiment_", "")
        p = f"gpt_prompt_tokens_{base}"
        c = f"gpt_completion_tokens_{base}"
        t = f"gpt_time_{base}"
        if p in df.columns and c in df.columns and t in df.columns:
            out.append((name, p, c, t))
    return out


def run_performance_analysis(
    sa: pd.DataFrame,
    allday: Optional[pd.DataFrame] = None,
    *,
    input_cost_per_1k: float = INPUT_COST_PER_1K,
    output_cost_per_1k: float = OUTPUT_COST_PER_1K,
) -> Dict[str, Any]:
    """
    Aggregate token usage, time, and estimated cost per model.
    """
    report: Dict[str, Any] = {"single_article": {}, "allday": {}}
    for name, pcol, ccol, tcol in _token_columns_single(sa):
        pt = sa[pcol].sum()
        ct = sa[ccol].sum()
        total_time = sa[tcol].sum()
        cost = (pt / 1000 * input_cost_per_1k) + (ct / 1000 * output_cost_per_1k)
        report["single_article"][name] = {
            "prompt_tokens_total": int(pt),
            "completion_tokens_total": int(ct),
            "total_time_sec": float(total_time),
            "estimated_cost_usd": round(cost, 4),
        }
    ad = allday if allday is not None else pd.DataFrame()
    for name, pcol, ccol, tcol in _token_columns_allday(ad):
        pt = ad[pcol].sum()
        ct = ad[ccol].sum()
        total_time = ad[tcol].sum()
        cost = (pt / 1000 * input_cost_per_1k) + (ct / 1000 * output_cost_per_1k)
        report["allday"][name] = {
            "prompt_tokens_total": int(pt),
            "completion_tokens_total": int(ct),
            "total_time_sec": float(total_time),
            "estimated_cost_usd": round(cost, 4),
        }
    return report


def run_error_analysis(
    df: pd.DataFrame,
    model_results: Dict[str, Any],
    *,
    y_true_col: str = "true_sentiment",
    top_n: int = 15,
) -> Dict[str, Any]:
    """
    Identify failure cases per model and overlap (commonly failed instances).
    """
    y_true = np.asarray(df[y_true_col], dtype=float)
    n = len(y_true)
    error_mask = {}
    for name, info in model_results.items():
        if "error" in info or "metrics" not in info:
            continue
        _, pred = _model_pred_column(df, name)
        if pred is None:
            continue
        wrong = (pred != y_true) & np.isfinite(pred) & np.isfinite(y_true)
        wrong = wrong & np.isin(y_true, (-1, 0, 1)) & np.isin(pred.astype(float), (-1, 0, 1))
        error_mask[name] = wrong
    common = np.zeros(n, dtype=bool)
    for m in error_mask.values():
        common |= m
    commonly_failed_idx = np.where(common)[0][:top_n]
    per_model_failures = {}
    for name, mask in error_mask.items():
        idx = np.where(mask)[0]
        per_model_failures[name] = {
            "n_errors": int(mask.sum()),
            "error_rate": float(mask.sum() / max(1, n)),
            "sample_indices": [int(i) for i in idx[:top_n]],
        }
    return {
        "per_model": per_model_failures,
        "sample_commonly_failed_indices": [int(i) for i in commonly_failed_idx],
        "n_total": n,
    }


def _model_pred_column(df: pd.DataFrame, name: str) -> Tuple[bool, Optional[np.ndarray]]:
    for mn, col, cont in SINGLE_ARTICLE_MODELS:
        if mn != name:
            continue
        if col not in df.columns:
            return False, None
        pred = _get_single_article_pred(df, col, cont)
        return True, pred
    return False, None


def build_model_comparison_report(
    base_path: Path,
    *,
    run_mcnemar: bool = True,
) -> Dict[str, Any]:
    """
    Load data, run all evaluations, performance, and error analysis.
    Optionally run McNemar between selected model pairs.
    """
    from ..utils.data_loader import load_all_dataframes

    data = load_all_dataframes(base_path)
    sa = data.get("single_article")
    ad = data.get("allday_articles")
    if sa is None:
        return {"error": "single_article CSV not loaded"}

    single_eval = evaluate_single_article_models(sa)
    report = {
        "single_article_evaluation": single_eval,
        "allday_evaluation": evaluate_allday_models(sa, ad) if ad is not None else {},
        "performance": run_performance_analysis(sa, ad),
        "error_analysis": run_error_analysis(sa, single_eval, top_n=20),
    }

    if run_mcnemar:
        single = report["single_article_evaluation"]
        y = np.asarray(sa["true_sentiment"], dtype=float)
        pairs = [("finbert", "gpt_p1"), ("gpt_p1", "gpt_p7"), ("finbert_a", "gpt_p1n")]
        mcnemar = {}
        for a, b in pairs:
            if a not in single or b not in single or "metrics" not in single[a] or "metrics" not in single[b]:
                continue
            pa = _get_single_article_pred(sa, next(c for n, c, _ in SINGLE_ARTICLE_MODELS if n == a), next(cont for n, _, cont in SINGLE_ARTICLE_MODELS if n == a))
            pb = _get_single_article_pred(sa, next(c for n, c, _ in SINGLE_ARTICLE_MODELS if n == b), next(cont for n, _, cont in SINGLE_ARTICLE_MODELS if n == b))
            mcnemar[f"{a}_vs_{b}"] = mcnemar_test_pair(y, pa, pb)
        report["statistical_tests"] = {"mcnemar": mcnemar}

    return report
