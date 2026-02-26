"""
Base class for all reasoning hops.

Inspired by THOR's modular step-by-step approach.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..context import ReasoningContext


def parse_sentiment_token(response: str) -> str:
    """
    Extract a single Positive/Negative/Neutral token from plain text (no JSON).

    Matches the behaviour of `pipeline/src/pipelines._safe_label`:
    - Look only at the **first token** of the response.
    - If it is exactly one of Positive/Negative/Neutral (case-insensitive),
      return that label (capitalized).
    - Otherwise, fall back to "Neutral".
    """
    if not response or not isinstance(response, str):
        return "Neutral"
    raw = response.strip()
    if not raw:
        return "Neutral"
    first = raw.split()[0]
    token = first.capitalize()
    if token in {"Positive", "Negative", "Neutral"}:
        return token
    return "Neutral"


class BaseHop(ABC):
    """
    Abstract base class for a reasoning hop.

    Each hop:
    1. Takes context from previous hops
    2. Performs its specific reasoning task
    3. Updates the context with its results

    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(
        self, context: ReasoningContext, llm_client: Any, **kwargs
    ) -> ReasoningContext:
        """
        Execute this hop's reasoning.

        Default: build_prompt -> LLM generate -> parse_response -> update_context.
        Override only if a hop needs custom execution flow.

        Args:
            context: Current reasoning context (accumulated from previous hops)
            llm_client: LLM client for making API calls
            **kwargs: Additional parameters (e.g. passed to generate). LLMClient's
                max_tokens (e.g. LLMClient(max_tokens=256)) applies to all hops
                unless overridden here via kwargs.

        Returns:
            Updated context with this hop's results
        """
        prompt = self.build_prompt(context)
        response = llm_client.generate(prompt, **kwargs)
        parsed = self.parse_response(response, context)
        return self.update_context(context, parsed, response)

    @abstractmethod
    def build_prompt(self, context: ReasoningContext) -> str:
        """
        Build the prompt for this hop.

        Args:
            context: Current reasoning context

        Returns:
            Prompt string to send to LLM
        """
        pass

    def parse_response(self, response: str, context: ReasoningContext) -> dict:
        """
        Parse LLM response and extract structured information.

        Override in subclasses for hop-specific parsing. Use
        extract_json_from_response() for JSON extraction.

        Args:
            response: Raw LLM response
            context: Current context

        Returns:
            Dictionary with parsed results
        """
        return {"raw_response": response}

    def update_context(
        self, context: ReasoningContext, parsed_result: dict, raw_response: str
    ) -> ReasoningContext:
        """
        Update context with this hop's results.

        Args:
            context: Current context
            parsed_result: Parsed results from parse_response
            raw_response: Raw LLM response

        Returns:
            Updated context
        """
        context.add_hop_result(self.name, parsed_result, raw_response)
        return context
