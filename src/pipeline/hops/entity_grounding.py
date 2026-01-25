"""
Hop 1: Entity Grounding

Identify the financial entity or ticker mentioned in the text.
Finance-specific adaptation: focuses on tickers, currencies, assets.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseHop
from ..context import ReasoningContext


class EntityGroundingHop(BaseHop):
    """
    First hop: Identify financial entities (tickers, currencies, assets).

    Finance-specific: Unlike general entity recognition, we focus on:
    - Currency pairs (EURUSD, GBPUSD)
    - Stock tickers (AAPL, TSLA)
    - Cryptocurrencies (BTC, ETH)
    - Financial instruments
    """

    def __init__(self):
        super().__init__(
            name="entity_grounding",
            description="Identify financial entities (tickers, currencies, assets)",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for entity identification."""
        prompt = f"""You are analyzing a financial news headline to identify the financial entity or ticker mentioned.

Headline: "{context.text}"

Task: Identify all financial entities mentioned in this headline. Focus on:
- Currency pairs (e.g., EURUSD, GBPUSD, USDJPY)
- Stock tickers (e.g., AAPL, TSLA, MSFT)
- Cryptocurrencies (e.g., BTC, ETH)
- Financial instruments or indices

If a ticker is already provided in the metadata, verify it matches the text.

Respond in JSON format:
{{
    "entities": ["entity1", "entity2", ...],
    "primary_entity": "main_entity",
    "confidence": "high|medium|low",
    "reasoning": "brief explanation"
}}"""
        return prompt

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse entity identification response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: try to parse the whole response
                result = json.loads(response)

            return {
                "entities": result.get("entities", []),
                "primary_entity": result.get("primary_entity"),
                "confidence": result.get("confidence", "medium"),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback parsing: look for common patterns
            entities = []
            # Common currency pairs
            currency_pattern = r"\b([A-Z]{{3}}/[A-Z]{{3}}|[A-Z]{{6}})\b"
            matches = re.findall(currency_pattern, context.text.upper())
            entities.extend(matches)

            # Common ticker patterns (3-5 uppercase letters)
            ticker_pattern = r"\b([A-Z]{{3,5}})\b"
            ticker_matches = re.findall(ticker_pattern, context.text.upper())
            entities.extend([t for t in ticker_matches if len(t) >= 3])

            return {
                "entities": list(set(entities))[:5],  # Limit to 5 unique
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

        # If ticker was provided in metadata, use it if no entity found
        if not context.primary_entity and context.ticker:
            context.primary_entity = context.ticker
            if context.ticker not in context.entities:
                context.entities.append(context.ticker)

        return context

    def execute(
        self, context: ReasoningContext, llm_client: Any, **kwargs
    ) -> ReasoningContext:
        """Execute entity grounding hop."""
        prompt = self.build_prompt(context)

        # Call LLM
        response = llm_client.generate(prompt, **kwargs)

        # Parse and update context
        parsed = self.parse_response(response, context)
        context = self.update_context(context, parsed, response)

        return context
