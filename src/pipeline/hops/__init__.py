"""
Hop modules for the 4-hop reasoning pipeline (FX Insight → Base/Quote sentiment → Final).

Plain-text prompts and responses; no JSON.
"""

from .base import BaseHop, parse_sentiment_token
from .fx_insight import FXInsightHop
from .base_currency_sentiment import BaseCurrencySentimentHop
from .quote_currency_sentiment import QuoteCurrencySentimentHop
from .final_classification import FinalClassificationHop

__all__ = [
    "BaseHop",
    "parse_sentiment_token",
    "FXInsightHop",
    "BaseCurrencySentimentHop",
    "QuoteCurrencySentimentHop",
    "FinalClassificationHop",
]
