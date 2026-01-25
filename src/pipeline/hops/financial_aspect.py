"""
Hop 2: Financial Aspect Identification

Adapt THOR's aspect hop to financial domain.
Identify key economic drivers: inflation, rates, growth, risk, etc.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseHop
from ..context import ReasoningContext


class FinancialAspectHop(BaseHop):
    """
    Second hop: Identify financial aspects/drivers.
    
    Adapts THOR's aspect identification to finance:
    - Economic indicators (inflation, GDP, unemployment)
    - Monetary policy (interest rates, QE, tapering)
    - Market factors (volatility, liquidity, risk)
    - Corporate factors (earnings, guidance, M&A)
    """
    
    # Common financial aspects
    FINANCIAL_ASPECTS = [
        "inflation", "interest_rates", "monetary_policy", "economic_growth",
        "unemployment", "gdp", "volatility", "liquidity", "risk",
        "earnings", "revenue", "guidance", "mergers_acquisitions",
        "regulatory", "geopolitical", "supply_chain", "commodity_prices"
    ]
    
    def __init__(self):
        super().__init__(
            name="financial_aspect",
            description="Identify key financial aspects/economic drivers"
        )
    
    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for financial aspect identification."""
        previous_reasoning = context.get_previous_reasoning()
        
        prompt = f"""You are analyzing a financial news headline to identify the key financial aspect or economic driver.

Headline: "{context.text}"
{("Previous analysis: " + previous_reasoning) if previous_reasoning else ""}

Task: Identify the primary financial aspect(s) being discussed. Common aspects include:
- Economic indicators: inflation, GDP, unemployment
- Monetary policy: interest rates, quantitative easing, central bank actions
- Market factors: volatility, liquidity, risk sentiment
- Corporate factors: earnings, revenue, guidance, M&A activity
- Other: regulatory changes, geopolitical events, supply chain issues

Respond in JSON format:
{{
    "aspects": ["aspect1", "aspect2", ...],
    "primary_aspect": "main_aspect",
    "reasoning": "brief explanation of why this aspect is relevant"
}}"""
        return prompt
    
    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse financial aspect identification response."""
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            return {
                "aspects": result.get("aspects", []),
                "primary_aspect": result.get("primary_aspect"),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback: keyword matching
            text_lower = context.text.lower()
            detected_aspects = []
            
            for aspect in self.FINANCIAL_ASPECTS:
                if aspect.replace("_", " ") in text_lower or aspect in text_lower:
                    detected_aspects.append(aspect)
            
            return {
                "aspects": detected_aspects[:3],  # Top 3
                "primary_aspect": detected_aspects[0] if detected_aspects else None,
                "reasoning": "Keyword-based fallback",
            }
    
    def update_context(
        self,
        context: ReasoningContext,
        parsed_result: dict,
        raw_response: str
    ) -> ReasoningContext:
        """Update context with financial aspect results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.financial_aspects = parsed_result.get("aspects", [])
        context.primary_aspect = parsed_result.get("primary_aspect")
        return context
    
    def execute(
        self,
        context: ReasoningContext,
        llm_client: Any,
        **kwargs
    ) -> ReasoningContext:
        """Execute financial aspect identification hop."""
        prompt = self.build_prompt(context)
        response = llm_client.generate(prompt, **kwargs)
        parsed = self.parse_response(response, context)
        context = self.update_context(context, parsed, response)
        return context
