"""
Base class for all reasoning hops.

Inspired by THOR's modular step-by-step approach.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..context import ReasoningContext


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

    @abstractmethod
    def execute(
        self, context: ReasoningContext, llm_client: Any, **kwargs
    ) -> ReasoningContext:
        """
        Execute this hop's reasoning.

        Args:
            context: Current reasoning context (accumulated from previous hops)
            llm_client: LLM client for making API calls
            **kwargs: Additional parameters

        Returns:
            Updated context with this hop's results
        """
        pass

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

        Override in subclasses for hop-specific parsing.

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
