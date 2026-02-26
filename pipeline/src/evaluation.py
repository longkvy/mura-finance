"""
evaluation.py
-------------
Evaluation utilities: metrics, classification report, and confusion matrix plot.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import LABELS


def evaluate(
    df: pd.DataFrame,
    true_col: str = "true_sentiment",
    pred_col: str = "final_prediction",
    title: str = "Results",
    save_path: str = None,
) -> dict:
    """
    Print accuracy, macro-F1, classification report, and draw confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ground-truth and predicted columns.
    true_col, pred_col : str
        Column names for ground-truth and predictions.
    title : str
        Title shown on the confusion matrix plot.
    save_path : str, optional
        If provided, saves the confusion matrix figure to this path.

    Returns
    -------
    dict with keys: accuracy, macro_f1
    """
    y_true = df[true_col]
    y_pred = df[pred_col]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print()
    print(
        classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0)
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  [evaluation] Figure saved → {save_path}")

    plt.show()

    return {"accuracy": acc, "macro_f1": macro_f1}
