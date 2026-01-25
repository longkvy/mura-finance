"""
Hop 2: Financial Aspect Identification

Adapt THOR's aspect hop to financial domain.
Identify key economic drivers: inflation, rates, growth, risk, etc.
"""

from __future__ import annotations


from .base import BaseHop, extract_json_from_response
from ..context import ReasoningContext
from ..prompts._loader import build_from_template


class FinancialAspectHop(BaseHop):
    """
    Second hop: Identify financial aspects/drivers.

    Adapts THOR's aspect identification to finance:
    - Economic indicators (inflation, GDP, unemployment)
    - Monetary policy (interest rates, QE, tapering)
    - Market factors (volatility, liquidity, risk)
    - Corporate factors (earnings, guidance, M&A)

    Prompt templates: `prompts/templates/financial_aspect.md` and `*_schema.txt`.
    """

    FINANCIAL_ASPECTS = [
        "inflation",
        "interest_rates",
        "monetary_policy",
        "economic_growth",
        "unemployment",
        "gdp",
        "volatility",
        "liquidity",
        "risk",
        "earnings",
        "revenue",
        "guidance",
        "mergers_acquisitions",
        "regulatory",
        "geopolitical",
        "supply_chain",
        "commodity_prices",
    ]

    def __init__(self):
        super().__init__(
            name="financial_aspect",
            description="Identify key financial aspects/economic drivers",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for financial aspect identification from templates."""
        prev = context.get_previous_reasoning()
        previous_reasoning = f"Previous analysis: {prev}" if prev else ""
        return build_from_template(
            "financial_aspect",
            headline=context.text,
            previous_reasoning=previous_reasoning,
        )

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse financial aspect identification response."""
        result = extract_json_from_response(response)
        if result is not None:
            return {
                "aspects": result.get("aspects", []),
                "primary_aspect": result.get("primary_aspect"),
                "reasoning": result.get("reasoning", ""),
            }
        text_lower = context.text.lower()
        detected = [
            a
            for a in self.FINANCIAL_ASPECTS
            if a.replace("_", " ") in text_lower or a in text_lower
        ]
        return {
            "aspects": detected[:3],
            "primary_aspect": detected[0] if detected else None,
            "reasoning": "Keyword-based fallback",
        }

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        """Update context with financial aspect results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.financial_aspects = parsed_result.get("aspects", [])
        context.primary_aspect = parsed_result.get("primary_aspect")
        return context
