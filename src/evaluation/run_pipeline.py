import numpy as np
import pandas as pd
from .metrics import confusion_matrix_dict


def run_pipeline_on_sample(pipeline, sample):
    """
    Run the ReasoningPipeline on a sample DataFrame and return (y_true, y_pred).
    """
    y_true = (
        sample["true_sentiment"].apply(_sentiment_str_to_numeric).values.astype(float)
    )
    y_pred = np.full(len(sample), np.nan, dtype=float)

    if pipeline is None:
        print("Pipeline not initialized; skipping run.")
        return y_true, y_pred

    for idx in range(len(sample)):
        row = sample.iloc[idx]
        text = row.get("title")  # or row.get("text")
        ticker = row.get("ticker")
        if pd.isna(ticker):
            ticker = None
        else:
            ticker = str(ticker).strip() or None
        try:
            context = pipeline.run(text, ticker=ticker)
            print(context)
            sent = context.sentiment
            y_pred[idx] = _sentiment_str_to_numeric(sent)
        except Exception as e:
            print(f"Sample {idx}: {e}")
            y_pred[idx] = np.nan

    print(
        f"Completed. Valid predictions: {np.sum(np.isfinite(y_pred))} / {len(sample)}"
    )
    return y_true, y_pred


def pretty_print_metrics(name: str, metrics: dict, y_true, y_pred) -> None:
    """
    Print classification metrics + confusion matrix for a given model/pipeline.
    """
    if "error" in metrics:
        print(f"Error in {name}:", metrics["error"])
        return

    print(f"{name} — Metrics")
    print("-" * 50)
    print(f"  n (valid pairs): {metrics['n']}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")

    cm = confusion_matrix_dict(y_true, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]))


def _sentiment_str_to_numeric(s: str) -> float:
    """Map pipeline sentiment to dataset labels: Positive->1, Negative->-1, Neutral->0."""
    if s is None:
        return np.nan
    s = (s or "").strip().lower()
    if s == "positive":
        return 1.0
    if s == "negative":
        return -1.0
    if s == "neutral":
        return 0.0
    return np.nan
