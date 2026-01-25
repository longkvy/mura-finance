"""
Hop 3: Implicit Cue Detection

Finance-specific: Detect hedging, euphemisms, indirect language.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseHop
from ..context import ReasoningContext


class ImplicitCueHop(BaseHop):
    """
    Third hop: Detect implicit linguistic cues.

    Finance-specific: Focus on:
    - Hedging language ("may", "could", "remains cautious")
    - Euphemisms ("challenging environment", "headwinds")
    - Indirect warnings ("concerns persist")
    - Mixed framing (positive and negative signals)
    """

    # Common hedging patterns
    HEDGING_PATTERNS = [
        r"\b(may|might|could|possibly|potentially|likely|unlikely)\b",
        r"\b(remains?|stays?|keeps?)\s+(cautious|uncertain|volatile)\b",
        r"\b(while|although|despite|however)\b",
    ]

    # Common euphemisms
    EUPHEMISMS = [
        "challenging",
        "headwinds",
        "uncertainty",
        "volatility",
        "concerns",
        "risks",
        "pressures",
        "weakness",
    ]

    def __init__(self):
        super().__init__(
            name="implicit_cue",
            description="Detect implicit linguistic cues (hedging, euphemisms)",
        )

    def build_prompt(self, context: ReasoningContext) -> str:
        """Build prompt for implicit cue detection."""
        previous_reasoning = context.get_previous_reasoning()

        prompt = f"""You are analyzing a financial news headline to detect implicit linguistic cues that indicate hedging, euphemisms, or indirect language.

Headline: "{context.text}"
{("Previous analysis: " + previous_reasoning) if previous_reasoning else ""}

Task: Identify implicit cues that suggest the sentiment is not explicit. Look for:
1. Hedging language: "may", "could", "remains cautious", "while", "however"
2. Euphemisms: "challenging environment", "headwinds", "uncertainty"
3. Indirect warnings: "concerns persist", "risks remain"
4. Mixed framing: both positive and negative signals in the same text

Respond in JSON format:
{{
    "cues": ["cue1", "cue2", ...],
    "cue_types": ["hedging"|"euphemism"|"indirect_warning"|"mixed_framing"],
    "has_implicit_language": true|false,
    "reasoning": "brief explanation"
}}"""
        return prompt

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse implicit cue detection response."""
        try:
            json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            return {
                "cues": result.get("cues", []),
                "cue_types": result.get("cue_types", []),
                "has_implicit_language": result.get("has_implicit_language", False),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback: pattern matching
            text_lower = context.text.lower()
            detected_cues = []
            cue_types = []

            # Check for hedging
            for pattern in self.HEDGING_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_cues.append("hedging_detected")
                    if "hedging" not in cue_types:
                        cue_types.append("hedging")

            # Check for euphemisms
            for euphemism in self.EUPHEMISMS:
                if euphemism in text_lower:
                    detected_cues.append(euphemism)
                    if "euphemism" not in cue_types:
                        cue_types.append("euphemism")

            return {
                "cues": detected_cues[:5],
                "cue_types": cue_types,
                "has_implicit_language": len(detected_cues) > 0,
                "reasoning": "Pattern-based fallback",
            }

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        """Update context with implicit cue results."""
        context = super().update_context(context, parsed_result, raw_response)
        context.implicit_cues = parsed_result.get("cues", [])
        context.cue_types = parsed_result.get("cue_types", [])
        return context

    def execute(
        self, context: ReasoningContext, llm_client: Any, **kwargs
    ) -> ReasoningContext:
        """Execute implicit cue detection hop."""
        prompt = self.build_prompt(context)
        response = llm_client.generate(prompt, **kwargs)
        parsed = self.parse_response(response, context)
        context = self.update_context(context, parsed, response)
        return context
