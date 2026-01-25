"""
Sentiment encoding utilities for MURA-Finance project.

Unified conversion between:
- Text: 'Positive', 'Negative', 'Neutral'
- Numeric (discrete): 1, -1, 0
- Numeric (continuous): sentiment scores typically in [-1, 1]

Reference: Fatouros et al. 2023, arXiv:2308.07935
"""

from typing import Union, Optional
import pandas as pd
import numpy as np

# Canonical mapping (used consistently across the project)
TEXT_TO_NUMERIC = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1,
}

NUMERIC_TO_TEXT = {v: k for k, v in TEXT_TO_NUMERIC.items()}

# Allow case-insensitive and common variants
TEXT_ALIASES = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "pos": "Positive",
    "neg": "Negative",
    "neu": "Neutral",
}


def to_numeric(
    value: Union[str, int, float],
    *,
    strict: bool = False,
) -> Optional[int]:
    """
    Convert sentiment to numeric encoding: -1 (Negative), 0 (Neutral), 1 (Positive).

    Args:
        value: Text label, numeric label, or continuous score.
        strict: If True, only accept exact text/numeric. If False, round floats and normalize aliases.

    Returns:
        -1, 0, or 1, or None if invalid/strict and no match.

    Examples:
        >>> to_numeric("Positive")
        1
        >>> to_numeric(-1)
        -1
        >>> to_numeric(0.8)  # positive score
        1
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Already discrete numeric
    if isinstance(value, (int, np.integer)):
        v = int(value)
        if v in NUMERIC_TO_TEXT:
            return v
        if strict:
            return None
        # Clamp to valid range
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    # Text
    if isinstance(value, str):
        s = value.strip()
        if s in TEXT_TO_NUMERIC:
            return TEXT_TO_NUMERIC[s]
        key = s.lower() if s else ""
        if key in TEXT_ALIASES:
            return TEXT_TO_NUMERIC[TEXT_ALIASES[key]]
        if strict:
            return None
        return None

    # Continuous score (float): threshold around 0
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if strict:
            return None
        if f > 0:
            return 1
        if f < 0:
            return -1
        return 0

    return None


def to_text(
    value: Union[str, int, float],
    *,
    strict: bool = False,
) -> Optional[str]:
    """
    Convert sentiment to text: 'Positive', 'Neutral', 'Negative'.

    Args:
        value: Numeric (-1, 0, 1), text, or continuous score.
        strict: If True, only accept exact numeric or text.

    Returns:
        'Positive', 'Neutral', or 'Negative', or None if invalid.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, (int, np.integer)):
        v = int(value)
        if v in NUMERIC_TO_TEXT:
            return NUMERIC_TO_TEXT[v]
        if strict:
            return None
        if v > 0:
            return "Positive"
        if v < 0:
            return "Negative"
        return "Neutral"

    if isinstance(value, str):
        s = value.strip()
        if s in TEXT_TO_NUMERIC:
            return s
        key = s.lower() if s else ""
        if key in TEXT_ALIASES:
            return TEXT_ALIASES[key]
        return s if not strict else None

    if isinstance(value, (float, np.floating)):
        if strict:
            return None
        n = to_numeric(value, strict=False)
        return NUMERIC_TO_TEXT[n] if n is not None else None

    return None


def normalize_series(
    s: pd.Series,
    to: str = "numeric",
    *,
    strict: bool = False,
) -> pd.Series:
    """
    Convert a pandas Series of sentiment values to a uniform encoding.

    Args:
        s: Series of text, numeric, or continuous sentiment.
        to: 'numeric' (-1,0,1) or 'text' ('Positive','Neutral','Negative').
        strict: Passed through to to_numeric / to_text.

    Returns:
        Series with the requested encoding. Invalid values become None/NaN.
    """
    if to == "numeric":
        return s.map(lambda x: to_numeric(x, strict=strict))
    if to == "text":
        return s.map(lambda x: to_text(x, strict=strict))
    raise ValueError("to must be 'numeric' or 'text'")


__all__ = [
    "TEXT_TO_NUMERIC",
    "NUMERIC_TO_TEXT",
    "to_numeric",
    "to_text",
    "normalize_series",
]
