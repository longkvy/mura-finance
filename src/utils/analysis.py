"""
Analysis utilities for MURA-Finance project.
"""

import pandas as pd
from typing import Dict, Tuple, List


def analyze_sentiment_distribution(df: pd.DataFrame, sentiment_col: str) -> Dict:
    """
    Analyze sentiment label distribution.

    Args:
        df: DataFrame containing sentiment data
        sentiment_col: Name of the sentiment column

    Returns:
        Dictionary with distribution statistics
    """
    if sentiment_col not in df.columns:
        return {"error": f"Column '{sentiment_col}' not found"}

    counts = df[sentiment_col].value_counts()
    percentages = (df[sentiment_col].value_counts(normalize=True) * 100).round(2)

    return {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
        "total": len(df),
        "unique_values": df[sentiment_col].nunique(),
    }


def map_ground_truth_to_predictions(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    join_columns: List[str] = ["published_at", "ticker", "title"],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Map ground truth to predictions.

    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        join_columns: Columns to use for joining

    Returns:
        Tuple of (merged DataFrame, mapping statistics)
    """
    # Ensure date columns are in the same format
    gt = ground_truth.copy()
    pred = predictions.copy()

    # Convert dates if needed
    if "published_at" in gt.columns and "published_at" in pred.columns:
        gt["published_at"] = pd.to_datetime(gt["published_at"], errors="coerce")
        pred["published_at"] = pd.to_datetime(pred["published_at"], errors="coerce")

    # Merge on specified columns
    available_join_cols = [
        col for col in join_columns if col in gt.columns and col in pred.columns
    ]

    if not available_join_cols:
        return pd.DataFrame(), {"error": "No common columns found for joining"}

    merged = pd.merge(
        gt,
        pred,
        on=available_join_cols,
        how="outer",
        suffixes=("_gt", "_pred"),
        indicator=True,
    )

    mapping_stats = {
        "total_ground_truth": len(gt),
        "total_predictions": len(pred),
        "matched_records": int((merged["_merge"] == "both").sum()),
        "ground_truth_only": int((merged["_merge"] == "left_only").sum()),
        "predictions_only": int((merged["_merge"] == "right_only").sum()),
        "match_rate": round((merged["_merge"] == "both").sum() / len(gt) * 100, 2),
    }

    return merged, mapping_stats
