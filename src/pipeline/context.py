"""
Reasoning context for 5-hop pipeline.

Maintains state and intermediate results as we progress through hops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ReasoningContext:
    """
    Context object that passes information between hops.

    Similar to THOR's approach of accumulating context at each step.
    """

    # Input
    text: str
    ticker: Optional[str] = None  # May be known from metadata

    # Hop 1: Entity Grounding
    entities: list[str] = field(default_factory=list)
    primary_entity: Optional[str] = None

    # Hop 2: Financial Aspect Identification
    financial_aspects: list[str] = field(default_factory=list)
    primary_aspect: Optional[str] = None

    # Hop 3: Implicit Cue Detection
    implicit_cues: list[str] = field(default_factory=list)
    cue_types: list[str] = field(default_factory=list)  # e.g., "hedging", "euphemism"

    # Hop 4: Implicit Sentiment Inference
    sentiment: Optional[str] = None  # "Positive", "Negative", "Neutral"
    sentiment_score: Optional[float] = None
    sentiment_reasoning: Optional[str] = None

    # Hop 5: Market Implication Inference
    market_implication: Optional[str] = None  # "Bullish", "Bearish", "Uncertain"
    market_reasoning: Optional[str] = None

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
        """Get accumulated reasoning from previous hops for context passing."""
        parts = []
        if self.primary_entity:
            parts.append(f"Entity: {self.primary_entity}")
        if self.primary_aspect:
            parts.append(f"Aspect: {self.primary_aspect}")
        if self.implicit_cues:
            parts.append(f"Cues: {', '.join(self.implicit_cues)}")
        if self.sentiment:
            parts.append(f"Sentiment: {self.sentiment}")
        return " | ".join(parts) if parts else ""
