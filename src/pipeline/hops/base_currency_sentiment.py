"""
Hop 2: Base currency sentiment – Hawkish/Bullish, Dovish/Bearish, or Neutral.

Plain-text output: one token (Positive / Negative / Neutral). No JSON.
"""

from __future__ import annotations

from .base import BaseHop, parse_sentiment_token
from ..context import ReasoningContext


def _build_prompt(
    title: str, ticker: str, base: str, hop1: str, extra_context: str = ""
) -> str:
    """Hop 2 – Classify sentiment for the base currency."""
    context_block = (
        f"\nContext (optional, only choose the most relevant):\n{extra_context}"
        if extra_context
        else ""
    )
    return f"""
Task: Financial Sentiment Analysis.

Input Ticker: {ticker}
Input Headline: {title}
Directional signals: {hop1}{context_block}

Base Currency: {base}

Is the headline Hawkish/Bullish, Dovish/Bearish, or Neutral for {base}?
If unclear, just return Neutral.

Answer ONLY:
Positive
Negative
Neutral
""".strip()


class BaseCurrencySentimentHop(BaseHop):
    """Hop 2: Base currency sentiment – one token, plain text."""

    def __init__(self):
        super().__init__(
            name="base_currency_sentiment",
            description="Classify sentiment for the base currency",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        ticker = context.ticker or "EURUSD"
        base = ticker[:3] if len(ticker) >= 3 else ticker
        hop1 = (context.fx_insight or "No directional pressure detected.").strip()
        return _build_prompt(context.text, ticker, base, hop1)

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        sentiment = parse_sentiment_token(response)
        return {"base_sentiment": sentiment, "raw_response": response}

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        context = super().update_context(context, parsed_result, raw_response)
        context.base_sentiment = parsed_result.get("base_sentiment")
        return context
