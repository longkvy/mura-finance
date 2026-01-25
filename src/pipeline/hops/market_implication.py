"""
Hop 5: Market Implication Inference

Finance-specific: Translate sentiment to market implications.
Output: Bullish, Bearish, or Uncertain.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseHop
from ..context import ReasoningContext


class MarketImplicationHop(BaseHop):
    """
    Fifth hop: Infer market implications.
    
    Finance-specific final step:
    - Translates sentiment to market direction
    - Considers entity, aspect, and sentiment together
    - Outputs: Bullish, Bearish, or Uncertain
    """
    
    def __init__(self):
        super().__init__(
            name="market_implication",
            description="Infer market implications (Bullish/Bearish/Uncertain)"
        )
    
    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for market implication inference."""
        previous_reasoning = context.get_previous_reasoning()
        
        prompt = f"""You are analyzing a financial news headline to infer its market implications.

Headline: "{context.text}"
Previous analysis: {previous_reasoning if previous_reasoning else "None"}
Sentiment: {context.sentiment if context.sentiment else "Not yet determined"}

Task: Based on the complete analysis (entity, financial aspect, implicit cues, and sentiment), determine the market implication:
- Bullish: Suggests upward price movement or positive outlook
- Bearish: Suggests downward price movement or negative outlook
- Uncertain: Mixed signals, hedging, or unclear implications

Consider:
- The specific financial aspect (e.g., inflation news may have different implications than earnings)
- The strength of the sentiment (strong positive vs. weak positive)
- The presence of hedging or uncertainty cues

Respond in JSON format:
{{
    "market_implication": "Bullish"|"Bearish"|"Uncertain",
    "reasoning": "detailed explanation of how you arrived at this market implication"
}}"""
        return prompt
    
    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse market implication inference response."""
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            implication = result.get("market_implication", "").strip()
            # Normalize implication
            implication_lower = implication.lower()
            if "bullish" in implication_lower or "bull" in implication_lower:
                implication = "Bullish"
            elif "bearish" in implication_lower or "bear" in implication_lower:
                implication = "Bearish"
            elif "uncertain" in implication_lower or "neutral" in implication_lower:
                implication = "Uncertain"
            else:
                # Fallback based on sentiment
                if context.sentiment == "Positive":
                    implication = "Bullish"
                elif context.sentiment == "Negative":
                    implication = "Bearish"
                else:
                    implication = "Uncertain"
            
            return {
                "market_implication": implication,
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback: map sentiment to market implication
            if context.sentiment == "Positive":
                implication = "Bullish"
            elif context.sentiment == "Negative":
                implication = "Bearish"
            else:
                implication = "Uncertain"
            
            return {
                "market_implication": implication,
                "reasoning": "Fallback mapping from sentiment",
            }
    
    def update_context(
        self,
        context: ReasoningContext,
        parsed_result: dict,
        raw_response: str
    ) -> ReasoningContext:
        """Update context with market implication results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.market_implication = parsed_result.get("market_implication")
        context.market_reasoning = parsed_result.get("reasoning")
        return context
    
    def execute(
        self,
        context: ReasoningContext,
        llm_client: Any,
        **kwargs
    ) -> ReasoningContext:
        """Execute market implication inference hop."""
        prompt = self.build_prompt(context)
        response = llm_client.generate(prompt, **kwargs)
        parsed = self.parse_response(response, context)
        context = self.update_context(context, parsed, response)
        return context
