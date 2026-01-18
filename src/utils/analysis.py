"""
Analysis utilities for MURA-Finance project.
"""

import pandas as pd
import numpy as np
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


def analyze_ticker_distribution(df: pd.DataFrame, ticker_col: str = "ticker") -> Dict:
    """
    Analyze ticker distribution.

    Args:
        df: DataFrame containing ticker data
        ticker_col: Name of the ticker column

    Returns:
        Dictionary with distribution statistics
    """
    if ticker_col not in df.columns:
        return {"error": f"Column '{ticker_col}' not found"}

    counts = df[ticker_col].value_counts()
    percentages = (df[ticker_col].value_counts(normalize=True) * 100).round(2)

    return {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
        "total": len(df),
        "unique_tickers": df[ticker_col].nunique(),
        "most_common": counts.head(10).to_dict(),
    }


def analyze_temporal_distribution(
    df: pd.DataFrame, date_col: str = "published_at"
) -> Dict:
    """
    Analyze temporal distribution of data.

    Args:
        df: DataFrame containing date data
        date_col: Name of the date column

    Returns:
        Dictionary with temporal statistics
    """
    if date_col not in df.columns:
        return {"error": f"Column '{date_col}' not found"}

    # Convert to datetime if not already
    df_date = df.copy()
    df_date[date_col] = pd.to_datetime(df_date[date_col], errors="coerce")

    valid_dates = df_date[date_col].dropna()

    if len(valid_dates) == 0:
        return {"error": "No valid dates found"}

    # Convert Period to string for JSON serialization
    monthly_counts = df_date.groupby(df_date[date_col].dt.to_period("M")).size()
    monthly_counts_dict = {str(k): int(v) for k, v in monthly_counts.items()}

    return {
        "date_range": {
            "start": str(valid_dates.min()),
            "end": str(valid_dates.max()),
            "span_days": (valid_dates.max() - valid_dates.min()).days,
        },
        "articles_per_day": {
            "mean": float(df_date.groupby(df_date[date_col].dt.date).size().mean()),
            "median": float(df_date.groupby(df_date[date_col].dt.date).size().median()),
            "std": float(df_date.groupby(df_date[date_col].dt.date).size().std()),
            "min": int(df_date.groupby(df_date[date_col].dt.date).size().min()),
            "max": int(df_date.groupby(df_date[date_col].dt.date).size().max()),
        },
        "articles_per_month": {"counts": monthly_counts_dict},
    }


def check_duplicates(df: pd.DataFrame, key_columns: List[str] = None) -> Dict:
    """
    Check for duplicate rows.

    Args:
        df: DataFrame to check
        key_columns: Columns to use for duplicate detection (default: all columns)

    Returns:
        Dictionary with duplicate statistics
    """
    if key_columns is None:
        key_columns = df.columns.tolist()

    # Check for duplicate rows (all columns)
    total_duplicates = df.duplicated().sum()

    # Check for duplicates based on key columns
    key_duplicates = df.duplicated(subset=key_columns).sum() if key_columns else 0

    return {
        "total_duplicate_rows": int(total_duplicates),
        "duplicate_rows_percentage": round(total_duplicates / len(df) * 100, 2),
        "duplicates_by_key": int(key_duplicates),
        "unique_rows": int(len(df) - total_duplicates),
    }


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality analysis.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        "shape": df.shape,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "missing_values": {
            "total": int(df.isnull().sum().sum()),
            "columns_with_missing": df.columns[df.isnull().any()].tolist(),
            "missing_by_column": df.isnull().sum().to_dict(),
        },
        "duplicates": check_duplicates(df),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "text_columns": df.select_dtypes(include=["object"]).columns.tolist(),
    }

    return quality_report


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


def identify_anomalies(df: pd.DataFrame, numeric_columns: List[str] = None) -> Dict:
    """
    Identify anomalies in numeric columns.

    Args:
        df: DataFrame to analyze
        numeric_columns: Columns to check for anomalies (default: all numeric)

    Returns:
        Dictionary with anomaly information
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    anomalies = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        # Statistical outliers (using IQR method)
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

        anomalies[col] = {
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "outliers_count": len(outliers),
            "outliers_percentage": round(len(outliers) / len(col_data) * 100, 2),
            "extreme_values": outliers.head(10).tolist() if len(outliers) > 0 else [],
        }

    return anomalies
