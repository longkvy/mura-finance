"""
Hop 3: Quote currency sentiment – Hawkish/Bullish, Dovish/Bearish, or Neutral.

Plain-text output: one token (Positive / Negative / Neutral). No JSON.
"""

from __future__ import annotations

from .base import BaseHop, parse_sentiment_token
from ..context import ReasoningContext


def _build_prompt(title: str, ticker: str, quote: str) -> str:
    """Hop 3 – Classify sentiment for the quote currency."""
    return f"""
Task: Financial Sentiment Analysis.

Input Ticker: {ticker}
Input Headline: {title}

Quote Currency: {quote}

Is the headline Hawkish/Bullish, Dovish/Bearish, or Neutral for {quote}?
If unclear, just return Neutral.

Answer ONLY:
Positive
Negative
Neutral
""".strip()


class QuoteCurrencySentimentHop(BaseHop):
    """Hop 3: Quote currency sentiment – one token, plain text."""

    def __init__(self):
        super().__init__(
            name="quote_currency_sentiment",
            description="Classify sentiment for the quote currency",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        ticker = context.ticker or "EURUSD"
        quote = ticker[3:] if len(ticker) >= 6 else "USD"
        return _build_prompt(context.text, ticker, quote)

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        sentiment = parse_sentiment_token(response)
        return {"quote_sentiment": sentiment, "raw_response": response}

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        context = super().update_context(context, parsed_result, raw_response)
        context.quote_sentiment = parsed_result.get("quote_sentiment")
        return context
