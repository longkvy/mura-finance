"""
Reasoning context for the 4-hop pipeline (FX Insight → Base/Quote sentiment → Final).

Maintains state and intermediate results as we progress through hops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ReasoningContext:
    """
    Context object that passes information between hops.

    4-hop flow: fx_insight → base_sentiment → quote_sentiment → sentiment (final).
    """

    # Input
    text: str
    ticker: Optional[str] = None  # e.g. EURCHF (base=ticker[:3], quote=ticker[3:])

    # Hop 1: FX Insight (directional pressure for both currencies)
    fx_insight: Optional[str] = None  # 1–3 bullet points, plain text

    # Hop 2: Base currency sentiment
    base_sentiment: Optional[str] = None  # "Positive", "Negative", "Neutral"

    # Hop 3: Quote currency sentiment
    quote_sentiment: Optional[str] = None  # "Positive", "Negative", "Neutral"

    # Hop 4: Final pair-level sentiment (combine BC/QC by rules)
    sentiment: Optional[str] = None  # "Positive", "Negative", "Neutral"

    # Metadata
    hop_results: Dict[str, Any] = field(default_factory=dict)
    raw_responses: Dict[str, str] = field(default_factory=dict)

    def add_hop_result(
        self, hop_name: str, result: Any, raw_response: Optional[str] = None
    ):
        """Store result from a hop."""
        self.hop_results[hop_name] = result
        if raw_response:
            self.raw_responses[hop_name] = raw_response

    def get_previous_reasoning(self) -> str:
        """Get accumulated reasoning from previous hops (for compatibility)."""
        parts = []
        if self.fx_insight:
            parts.append(f"FX insight: {self.fx_insight[:200]}...")
        if self.base_sentiment:
            parts.append(f"Base: {self.base_sentiment}")
        if self.quote_sentiment:
            parts.append(f"Quote: {self.quote_sentiment}")
        if self.sentiment:
            parts.append(f"Sentiment: {self.sentiment}")
        return " | ".join(parts) if parts else ""
