"""
Hop 1: Entity Grounding

Identify the financial entity or ticker mentioned in the text.
Finance-specific adaptation: focuses on tickers, currencies, assets.
"""

from __future__ import annotations

import re

from .base import BaseHop, extract_json_from_response
from ..context import ReasoningContext
from ..prompts._loader import build_from_template


class EntityGroundingHop(BaseHop):
    """
    First hop: Identify financial entities (tickers, currencies, assets).

    Finance-specific: Unlike general entity recognition, we focus on:
    - Currency pairs (EURUSD, GBPUSD)
    - Stock tickers (AAPL, TSLA)
    - Cryptocurrencies (BTC, ETH)
    - Financial instruments

    Prompt templates: `prompts/templates/entity_grounding.md` and `*_schema.txt`.
    """

    def __init__(self):
        super().__init__(
            name="entity_grounding",
            description="Identify financial entities (tickers, currencies, assets)",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for entity identification from templates."""
        ticker_line = (
            f"Provided ticker (metadata): {context.ticker}" if context.ticker else ""
        )
        return build_from_template(
            "entity_grounding",
            headline=context.text,
            ticker_line=ticker_line,
        )

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse entity identification response."""
        result = extract_json_from_response(response)
        if result is not None:
            return {
                "entities": result.get("entities", []),
                "primary_entity": result.get("primary_entity"),
                "confidence": result.get("confidence", "medium"),
                "reasoning": result.get("reasoning", ""),
            }
        # Fallback: pattern matching
        entities = []
        currency_pattern = r"\b([A-Z]{3}/[A-Z]{3}|[A-Z]{6})\b"
        entities.extend(re.findall(currency_pattern, context.text.upper()))
        ticker_pattern = r"\b([A-Z]{3,5})\b"
        ticker_matches = re.findall(ticker_pattern, context.text.upper())
        entities.extend([t for t in ticker_matches if len(t) >= 3])
        return {
            "entities": list(set(entities))[:5],
            "primary_entity": entities[0] if entities else None,
            "confidence": "low",
            "reasoning": "Fallback pattern matching",
        }

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        """Update context with entity grounding results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.entities = parsed_result.get("entities", [])
        context.primary_entity = parsed_result.get("primary_entity")
        if not context.primary_entity and context.ticker:
            context.primary_entity = context.ticker
            if context.ticker not in context.entities:
                context.entities.append(context.ticker)
        return context
