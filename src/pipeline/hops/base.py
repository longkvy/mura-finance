"""
Base class for all reasoning hops.

Inspired by THOR's modular step-by-step approach.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ..context import ReasoningContext


def extract_json_from_response(response: str) -> dict | None:
    """
    Extract a JSON object from an LLM response string.

    Tries json.loads(response) first, then looks for {...} with
    brace-matching to support nested objects. Returns None on failure.

    Used by all hops to avoid duplicating JSON extraction logic.
    """
    if not response or not isinstance(response, str):
        return None
    s = response.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(s[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class BaseHop(ABC):
    """
    Abstract base class for a reasoning hop.

    Each hop:
    1. Takes context from previous hops
    2. Performs its specific reasoning task
    3. Updates the context with its results

    Set max_tokens on the subclass to limit output length (faster on Colab/API).
    """

    max_tokens: int | None = None  # override in subclass to cap output tokens

    def __init__(self, name: str, description: str, max_tokens: int | None = None):
        self.name = name
        self.description = description
        if max_tokens is not None:
            self.max_tokens = max_tokens

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
            **kwargs: Additional parameters (e.g. passed to generate)

        Returns:
            Updated context with this hop's results
        """
        prompt = self.build_prompt(context)
        gen_kwargs = dict(kwargs)
        if getattr(self, "max_tokens", None) is not None:
            gen_kwargs["max_tokens"] = self.max_tokens
        response = llm_client.generate(prompt, **gen_kwargs)
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
