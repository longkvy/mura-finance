"""
Hop 4: Implicit Sentiment Inference

Adapt THOR's opinion + polarity hops to infer implicit sentiment.
Classify as Positive, Negative, or Neutral.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseHop
from ..context import ReasoningContext


class SentimentInferenceHop(BaseHop):
    """
    Fourth hop: Infer implicit sentiment.
    
    Adapts THOR's opinion + polarity hops:
    - Uses context from previous hops (entity, aspect, cues)
    - Infers sentiment even when not explicit
    - Outputs: Positive, Negative, or Neutral
    """
    
    def __init__(self):
        super().__init__(
            name="sentiment_inference",
            description="Infer implicit sentiment (Positive/Negative/Neutral)"
        )
    
    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for sentiment inference."""
        previous_reasoning = context.get_previous_reasoning()
        
        prompt = f"""You are analyzing a financial news headline to infer its implicit sentiment.

Headline: "{context.text}"
Previous analysis: {previous_reasoning if previous_reasoning else "None"}

Task: Based on the identified entity, financial aspect, and implicit cues, infer the sentiment even if it's not explicitly stated. Consider:
- The financial aspect and its implications
- Hedging language may indicate uncertainty or caution
- Euphemisms often mask negative sentiment
- Mixed framing suggests neutral or uncertain sentiment

Classify the sentiment as one of: Positive, Negative, Neutral

Respond in JSON format:
{{
    "sentiment": "Positive"|"Negative"|"Neutral",
    "sentiment_score": -1.0 to 1.0 (optional confidence score),
    "reasoning": "detailed explanation of how you inferred the sentiment from implicit cues"
}}"""
        return prompt
    
    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse sentiment inference response."""
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            sentiment = result.get("sentiment", "").strip()
            # Normalize sentiment
            sentiment_lower = sentiment.lower()
            if "positive" in sentiment_lower or sentiment == "1":
                sentiment = "Positive"
            elif "negative" in sentiment_lower or sentiment == "-1":
                sentiment = "Negative"
            elif "neutral" in sentiment_lower or sentiment == "0":
                sentiment = "Neutral"
            else:
                sentiment = "Neutral"  # Default
            
            return {
                "sentiment": sentiment,
                "sentiment_score": result.get("sentiment_score"),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback: simple keyword-based sentiment
            text_lower = context.text.lower()
            positive_words = ["positive", "gains", "rises", "surges", "strong", "bullish", "up"]
            negative_words = ["negative", "falls", "drops", "declines", "weak", "bearish", "down"]
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "Positive"
            elif neg_count > pos_count:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                "sentiment": sentiment,
                "sentiment_score": None,
                "reasoning": "Keyword-based fallback",
            }
    
    def update_context(
        self,
        context: ReasoningContext,
        parsed_result: dict,
        raw_response: str
    ) -> ReasoningContext:
        """Update context with sentiment inference results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.sentiment = parsed_result.get("sentiment")
        context.sentiment_score = parsed_result.get("sentiment_score")
        context.sentiment_reasoning = parsed_result.get("reasoning")
        return context
    
    def execute(
        self,
        context: ReasoningContext,
        llm_client: Any,
        **kwargs
    ) -> ReasoningContext:
        """Execute sentiment inference hop."""
        prompt = self.build_prompt(context)
        response = llm_client.generate(prompt, **kwargs)
        parsed = self.parse_response(response, context)
        context = self.update_context(context, parsed, response)
        return context
