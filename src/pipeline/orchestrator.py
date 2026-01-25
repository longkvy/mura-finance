"""
Pipeline orchestrator for 5-hop reasoning.

Chains hops together with context passing, similar to THOR's approach.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .context import ReasoningContext
from .llm_client import LLMClient
from .hops import (
    EntityGroundingHop,
    FinancialAspectHop,
    ImplicitCueHop,
    SentimentInferenceHop,
    MarketImplicationHop,
)


class ReasoningPipeline:
    """
    Main orchestrator for 5-hop reasoning pipeline.

    Inspired by THOR's step-by-step CoT approach:
    - Chains hops sequentially
    - Passes context between hops
    - Accumulates reasoning at each step
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize reasoning pipeline.

        Args:
            llm_client: Optional pre-configured LLM client
            api_key: OpenAI API key (if llm_client not provided)
            model: Model name (or set OPENAI_MODEL env var; if llm_client not provided)
        """
        if llm_client is None:
            self.llm_client = LLMClient(api_key=api_key, model=model)
        else:
            self.llm_client = llm_client

        # Initialize hops
        self.hops = [
            EntityGroundingHop(),
            FinancialAspectHop(),
            ImplicitCueHop(),
            SentimentInferenceHop(),
            MarketImplicationHop(),
        ]

    def run(
        self, text: str, ticker: Optional[str] = None, **kwargs
    ) -> ReasoningContext:
        """
        Run the complete 5-hop reasoning pipeline.

        Args:
            text: Financial news headline or text to analyze
            ticker: Optional ticker/entity (if known from metadata)
            **kwargs: Additional parameters to pass to LLM calls

        Returns:
            ReasoningContext with all hop results
        """
        # Initialize context
        context = ReasoningContext(text=text, ticker=ticker)

        # Execute each hop sequentially
        for hop in self.hops:
            try:
                context = hop.execute(context, self.llm_client, **kwargs)
            except Exception as e:
                # Log error but continue with next hop
                context.add_hop_result(
                    hop.name, {"error": str(e)}, raw_response=f"Error: {str(e)}"
                )

        return context

    def get_final_result(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Extract final result from context.

        Args:
            context: Completed reasoning context

        Returns:
            Dictionary with final predictions and reasoning
        """
        return {
            "text": context.text,
            "entity": context.primary_entity,
            "financial_aspect": context.primary_aspect,
            "implicit_cues": context.implicit_cues,
            "sentiment": context.sentiment,
            "sentiment_score": context.sentiment_score,
            "market_implication": context.market_implication,
            "reasoning": {
                "entity_reasoning": context.hop_results.get("entity_grounding", {}).get(
                    "reasoning"
                ),
                "aspect_reasoning": context.hop_results.get("financial_aspect", {}).get(
                    "reasoning"
                ),
                "cue_reasoning": context.hop_results.get("implicit_cue", {}).get(
                    "reasoning"
                ),
                "sentiment_reasoning": context.sentiment_reasoning,
                "market_reasoning": context.market_reasoning,
            },
            "all_hop_results": context.hop_results,
        }

    def get_usage_stats(self) -> Dict[str, int]:
        """Get LLM usage statistics."""
        return self.llm_client.get_usage_stats()
