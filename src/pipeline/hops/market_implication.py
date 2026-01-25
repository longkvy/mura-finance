"""
Hop 5: Market Implication Inference

Finance-specific: Translate sentiment to market implications.
Output: Bullish, Bearish, or Uncertain.
"""

from __future__ import annotations


from .base import BaseHop, extract_json_from_response
from ..context import ReasoningContext
from ..prompts._loader import build_from_template


def _normalize_implication(value: str, sentiment: str | None) -> str:
    """Map implication string to Bullish/Bearish/Uncertain; fallback to sentiment."""
    v = (value or "").strip().lower()
    if "bullish" in v or "bull" in v:
        return "Bullish"
    if "bearish" in v or "bear" in v:
        return "Bearish"
    if "uncertain" in v or "neutral" in v:
        return "Uncertain"
    if sentiment == "Positive":
        return "Bullish"
    if sentiment == "Negative":
        return "Bearish"
    return "Uncertain"


class MarketImplicationHop(BaseHop):
    """
    Fifth hop: Infer market implications.

    Finance-specific final step:
    - Translates sentiment to market direction
    - Considers entity, aspect, and sentiment together
    - Outputs: Bullish, Bearish, or Uncertain

    Prompt templates: `prompts/templates/market_implication.md` and `*_schema.txt`.
    """

    def __init__(self):
        super().__init__(
            name="market_implication",
            description="Infer market implications (Bullish/Bearish/Uncertain)",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for market implication inference from templates."""
        prev = context.get_previous_reasoning()
        previous_reasoning = prev if prev else "None"
        sentiment = context.sentiment if context.sentiment else "Not yet determined"
        return build_from_template(
            "market_implication",
            headline=context.text,
            previous_reasoning=previous_reasoning,
            sentiment=sentiment,
        )

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse market implication inference response."""
        result = extract_json_from_response(response)
        if result is not None:
            implication = _normalize_implication(
                result.get("market_implication", ""), context.sentiment
            )
            return {
                "market_implication": implication,
                "reasoning": result.get("reasoning", ""),
            }
        implication = _normalize_implication("", context.sentiment)
        return {
            "market_implication": implication,
            "reasoning": "Fallback mapping from sentiment",
        }

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        """Update context with market implication results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.market_implication = parsed_result.get("market_implication")
        context.market_reasoning = parsed_result.get("reasoning")
        return context
