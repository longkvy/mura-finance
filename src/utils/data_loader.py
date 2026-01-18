"""
Data loading utilities for MURA-Finance project.
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def load_all_dataframes(base_path: Path = Path(".")) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files into a dictionary of DataFrames.

    Args:
        base_path: Base path where CSV files are located

    Returns:
        Dictionary with keys: 'ground_truth', 'single_article', 'allday_articles'
    """
    data = {}

    # Ground truth data
    ground_truth_path = base_path / "sentiment_annotated_with_texts.csv"
    if ground_truth_path.exists():
        data["ground_truth"] = pd.read_csv(ground_truth_path)
        print(f"Loaded ground truth: {len(data['ground_truth'])} rows")

    # Single article predictions
    single_article_path = base_path / "sentiment_predictions_single_article.csv"
    if single_article_path.exists():
        data["single_article"] = pd.read_csv(single_article_path)
        print(f"Loaded single article predictions: {len(data['single_article'])} rows")

    # All day articles (aggregated)
    allday_path = base_path / "sentiment_predictions_allday_articles.csv"
    if allday_path.exists():
        data["allday_articles"] = pd.read_csv(allday_path)
        print(f"Loaded all-day articles: {len(data['allday_articles'])} rows")

    return data


def get_schema_info(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Get schema information for a DataFrame.

    Args:
        df: DataFrame to analyze
        name: Name of the DataFrame for display

    Returns:
        DataFrame with column information
    """
    schema_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values,
            "Null Percentage": (df.isnull().sum() / len(df) * 100).round(2).values,
            "Unique Values": [df[col].nunique() for col in df.columns],
        }
    )

    return schema_info


def print_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print a summary of a DataFrame.

    Args:
        df: DataFrame to summarize
        name: Name of the DataFrame for display
    """
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {name}")
    print(f"{'=' * 80}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn Information:")
    print(get_schema_info(df, name).to_string(index=False))
    print("\nFirst few rows:")
    print(df.head(3).to_string())
    print(f"\n{'=' * 80}\n")
