"""
LLM client for making API calls to OpenAI or other providers.

Handles error handling, retries, and token tracking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import os

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None


class LLMClient:
    """
    Client for interacting with LLM APIs.
    
    Supports OpenAI API with error handling and retries.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (or set OPENAI_MODEL env var; default: gpt-3.5-turbo)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Merge kwargs with defaults
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                
                # Track tokens
                usage = response.usage
                if usage:
                    self.total_prompt_tokens += usage.prompt_tokens
                    self.total_completion_tokens += usage.completion_tokens
                self.total_calls += 1
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
        
        raise RuntimeError(f"Unexpected error: {last_error}")
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
