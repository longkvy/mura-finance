"""
Hop 3: Implicit Cue Detection

Finance-specific: Detect hedging, euphemisms, indirect language.
"""

from __future__ import annotations

import re

from .base import BaseHop, extract_json_from_response
from ..context import ReasoningContext
from ..prompts._loader import build_from_template


class ImplicitCueHop(BaseHop):
    """
    Third hop: Detect implicit linguistic cues.

    Finance-specific: Focus on:
    - Hedging language ("may", "could", "remains cautious")
    - Euphemisms ("challenging environment", "headwinds")
    - Indirect warnings ("concerns persist")
    - Mixed framing (positive and negative signals)

    Prompt templates: `prompts/templates/implicit_cue.md` and `*_schema.txt`.
    """

    HEDGING_PATTERNS = [
        r"\b(may|might|could|possibly|potentially|likely|unlikely)\b",
        r"\b(remains?|stays?|keeps?)\s+(cautious|uncertain|volatile)\b",
        r"\b(while|although|despite|however)\b",
    ]
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
        """Build prompt for implicit cue detection from templates."""
        prev = context.get_previous_reasoning()
        previous_reasoning = f"Previous analysis: {prev}" if prev else ""
        return build_from_template(
            "implicit_cue",
            headline=context.text,
            previous_reasoning=previous_reasoning,
        )

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """Parse implicit cue detection response."""
        result = extract_json_from_response(response)
        if result is not None:
            return {
                "cues": result.get("cues", []),
                "cue_types": result.get("cue_types", []),
                "has_implicit_language": result.get("has_implicit_language", False),
                "reasoning": result.get("reasoning", ""),
            }
        text_lower = context.text.lower()
        detected_cues = []
        cue_types = []
        for pattern in self.HEDGING_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_cues.append("hedging_detected")
                if "hedging" not in cue_types:
                    cue_types.append("hedging")
        for e in self.EUPHEMISMS:
            if e in text_lower:
                detected_cues.append(e)
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
