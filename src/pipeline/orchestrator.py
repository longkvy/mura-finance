"""
Pipeline orchestrator: 4-hop reasoning or single-prompt (zero-shot) mode.

- mode="4hop": FX Insight → Base/Quote sentiment → Final classification.
- mode="single": One LLM call with a simple classification prompt (no hops).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .context import ReasoningContext
from .llm_client import LLMClient
from .hops import (
    FXInsightHop,
    BaseCurrencySentimentHop,
    QuoteCurrencySentimentHop,
    FinalClassificationHop,
    parse_sentiment_token,
)

# Match per-hop max_new_tokens used in `pipeline/src/pipelines.py`
try:  # pragma: no cover - optional dependency on external package layout
    from pipeline.src.config import MAX_NEW_TOKENS_SHORT, MAX_NEW_TOKENS_HOP1
except Exception:  # Fallback defaults if pipeline package is unavailable
    MAX_NEW_TOKENS_SHORT = 16
    MAX_NEW_TOKENS_HOP1 = 40


def _simple_prompt(title: str, ticker: str) -> str:
    """Zero-shot single-step classification prompt (same as pipeline/src/prompts.py prompt_only)."""
    return f"""
Act as an expert at forex trading.
Classify the sentiment for ***{ticker}*** based only on the headline '{title}'
Answer in one token: Positive, Negative, or Neutral
""".strip()


class ReasoningPipeline:
    """
    Reasoning pipeline in two modes:

    - mode="4hop" (default): FX Insight → Base sentiment → Quote sentiment → Final classification.
    - mode="single": One call with a simple prompt; returns context.sentiment only.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mode: str = "4hop",
    ):
        """
        Initialize reasoning pipeline.

        Args:
            llm_client: Optional pre-configured LLM client
            api_key: OpenAI API key (if llm_client not provided)
            model: Model name (or set OPENAI_MODEL env var; if llm_client not provided)
            mode: "4hop" for full 4-hop chain, "single" for one-shot simple prompt only
        """
        if llm_client is None:
            self.llm_client = LLMClient(api_key=api_key, model=model)
        else:
            self.llm_client = llm_client

        self.mode = mode.lower()
        if self.mode not in ("4hop", "single"):
            raise ValueError(f'mode must be "4hop" or "single"; got {mode!r}')

        self.hops = (
            [
                FXInsightHop(),
                BaseCurrencySentimentHop(),
                QuoteCurrencySentimentHop(),
                FinalClassificationHop(),
            ]
            if self.mode == "4hop"
            else []
        )

    def run(
        self, text: str, ticker: Optional[str] = None, **kwargs
    ) -> ReasoningContext:
        """
        Run the pipeline (4-hop or single-prompt depending on mode).

        Args:
            text: Financial news headline or text to analyze
            ticker: FX pair (e.g. EURCHF); for single mode used in the prompt
            **kwargs: Additional parameters to pass to LLM calls

        Returns:
            ReasoningContext; sentiment always set; fx_insight/base_sentiment/quote_sentiment only in 4hop mode
        """
        context = ReasoningContext(text=text, ticker=ticker)

        if self.mode == "single":
            prompt = _simple_prompt(text, ticker or "FX")
            try:
                # Single-prompt generation mirrors `single_prompt_pipeline`:
                # short max tokens and strict first-token label extraction.
                gen_kwargs = dict(kwargs)
                gen_kwargs.setdefault("max_tokens", MAX_NEW_TOKENS_SHORT)
                response = self.llm_client.generate(prompt, **gen_kwargs)
                sentiment = parse_sentiment_token(response)
                context.sentiment = sentiment
                context.add_hop_result(
                    "simple_prompt", {"sentiment": sentiment}, raw_response=response
                )
            except Exception as e:
                context.add_hop_result(
                    "simple_prompt", {"error": str(e)}, raw_response=f"Error: {str(e)}"
                )
            return context

        # 4-hop mode: align generation behaviour with `multihop_pipeline`:
        # - Hop 1 (FX insight): MAX_NEW_TOKENS_HOP1
        # - Hops 2–4: MAX_NEW_TOKENS_SHORT
        for hop in self.hops:
            try:
                hop_kwargs = dict(kwargs)
                if isinstance(hop, FXInsightHop):
                    hop_kwargs.setdefault("max_tokens", MAX_NEW_TOKENS_HOP1)
                else:
                    hop_kwargs.setdefault("max_tokens", MAX_NEW_TOKENS_SHORT)
                context = hop.execute(context, self.llm_client, **hop_kwargs)
            except Exception as e:
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
            Dictionary with final predictions and 4-hop outputs
        """
        return {
            "text": context.text,
            "ticker": context.ticker,
            "fx_insight": context.fx_insight,
            "base_sentiment": context.base_sentiment,
            "quote_sentiment": context.quote_sentiment,
            "sentiment": context.sentiment,
            "all_hop_results": context.hop_results,
        }

    def get_usage_stats(self) -> Dict[str, int]:
        """Get LLM usage statistics."""
        return self.llm_client.get_usage_stats()
