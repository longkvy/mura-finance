"""
Hop 4: Final classification – combine BC/QC sentiments into one pair-level label.

Rules: BC Positive & QC not Positive → Positive; BC Negative & QC not Negative → Negative; else Neutral.
Plain-text output: one token. No JSON.
"""

from __future__ import annotations

from .base import BaseHop, parse_sentiment_token
from ..context import ReasoningContext


def _build_prompt(ticker: str, hop2: str, hop3: str) -> str:
    """Hop 4 – Combine BC/QC sentiments into a single pair-level prediction."""
    return f"""
Task: Financial Sentiment Analysis.

Ticker: {ticker}

BC Sentiment: {hop2}
QC Sentiment: {hop3}

If BC Positive and QC not Positive → Positive
If BC Negative and QC not Negative → Negative
If both same or both Neutral → Neutral

Return ONLY one token:
Positive
Negative
Neutral
""".strip()


class FinalClassificationHop(BaseHop):
    """Hop 4: Final classification – one token from BC/QC rules, plain text."""

    def __init__(self):
        super().__init__(
            name="final_classification",
            description="Combine base/quote sentiment into pair-level label",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        ticker = context.ticker or "EURUSD"
        hop2 = context.base_sentiment or "Neutral"
        hop3 = context.quote_sentiment or "Neutral"
        return _build_prompt(ticker, hop2, hop3)

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        sentiment = parse_sentiment_token(response)
        return {"sentiment": sentiment, "raw_response": response}

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        context = super().update_context(context, parsed_result, raw_response)
        context.sentiment = parsed_result.get("sentiment")
        return context
