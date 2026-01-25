"""
Hop 4: Implicit Sentiment Inference

Adapt THOR's opinion + polarity hops to infer implicit sentiment.
Classify as Positive, Negative, or Neutral.
"""

from __future__ import annotations


from .base import BaseHop, extract_json_from_response
from ..context import ReasoningContext
from ..prompts._loader import build_from_template


_POSITIVE_WORDS = ["positive", "gains", "rises", "surges", "strong", "bullish", "up"]
_NEGATIVE_WORDS = ["negative", "falls", "drops", "declines", "weak", "bearish", "down"]


def _normalize_sentiment(value: str) -> str:
    """Map sentiment string to canonical Positive/Negative/Neutral."""
    raw = (value or "").strip()
    v = raw.lower()
    if "positive" in v or raw == "1":
        return "Positive"
    if "negative" in v or raw == "-1":
        return "Negative"
    if "neutral" in v or raw == "0":
        return "Neutral"
    return "Neutral"


class SentimentInferenceHop(BaseHop):
    """
    Fourth hop: Infer implicit sentiment.

    Adapts THOR's opinion + polarity hops:
    - Uses context from previous hops (entity, aspect, cues)
    - Infers sentiment even when not explicit
    - Outputs: Positive, Negative, or Neutral

    Prompt templates: `prompts/templates/sentiment_inference.md` and `*_schema.txt`.
    """

    def __init__(self):
        super().__init__(
            name="sentiment_inference",
            description="Infer implicit sentiment (Positive/Negative/Neutral)",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for sentiment inference from templates."""
        prev = context.get_previous_reasoning()
        previous_reasoning = prev if prev else "None"
        return build_from_template(
            "sentiment_inference",
            headline=context.text,
            previous_reasoning=previous_reasoning,
        )

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse sentiment inference response."""
        result = extract_json_from_response(response)
        if result is not None:
            sentiment = _normalize_sentiment(result.get("sentiment", ""))
            return {
                "sentiment": sentiment,
                "sentiment_score": result.get("sentiment_score"),
                "reasoning": result.get("reasoning", ""),
            }
        text_lower = context.text.lower()
        pos = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
        sentiment = (
            "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
        )
        return {
            "sentiment": sentiment,
            "sentiment_score": None,
            "reasoning": "Keyword-based fallback",
        }

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        """Update context with sentiment inference results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.sentiment = parsed_result.get("sentiment")
        context.sentiment_score = parsed_result.get("sentiment_score")
        context.sentiment_reasoning = parsed_result.get("reasoning")
        return context
