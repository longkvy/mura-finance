"""
Hop 1: FX Insight – directional pressure for both currencies in the pair.

Plain-text output (1–3 bullet points). No JSON.
"""

from __future__ import annotations

from .base import BaseHop
from ..context import ReasoningContext


def _build_prompt(title: str, ticker: str) -> str:
    """Hop 1 – Detect directional pressure for both currencies."""
    return f"""
You are an FX signal analyst.

Pair: {ticker}
Headline: {title}

For BOTH currencies in the pair, determine if the headline implies:

Upward pressure
Downward pressure
No clear pressure

Consider:
• Technical language (breaks, trims, downside, bulls, bears, resistance, floor)
• Policy tone (too fast, cautious, tightening, easing)
• Yield movement
• Risk language (woes, concerns, uncertainty)
• Sustainability language (unsustainable, fading)

For each detected pressure, state:
• Currency affected
• Direction of pressure (upward / downward)

If absolutely no directional pressure is implied, state:
No directional pressure detected.

Do NOT classify sentiment.
Return 1–3 short bullet points.
""".strip()


class FXInsightHop(BaseHop):
    """Hop 1: FX Insight – directional pressure, plain text."""

    def __init__(self):
        super().__init__(
            name="fx_insight",
            description="Detect directional pressure for both currencies",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        ticker = context.ticker or "EURUSD"
        return _build_prompt(context.text, ticker)

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        text = (response or "").strip()
        return {"fx_insight": text, "raw_response": text}

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        context = super().update_context(context, parsed_result, raw_response)
        context.fx_insight = parsed_result.get("fx_insight", "").strip() or None
        return context
