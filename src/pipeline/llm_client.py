"""
Minimal LLM client that talks only to a local Ollama server.

Designed to be simple and fast: one backend (Ollama), one client (LLMClient).
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Load environment variables from a local .env file (if present)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; if it's not installed we just rely on the process env.
    pass

try:
    import requests
except ImportError:
    requests = None  # type: ignore


class BaseLLMBackend(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate a response. Returns the generated text."""
        ...

    def get_usage_stats(self) -> Dict[str, int]:
        """Return token/call usage. Override if the backend tracks usage."""
        return {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage counters. Override if the backend tracks usage."""
        pass


class OllamaBackend(BaseLLMBackend):
    """
    Talk to an Ollama server (e.g. `ollama serve`) via its HTTP API.

    Assumes a running Ollama instance (default: http://localhost:11434).
    Uses the `/api/generate` endpoint with `stream=False`.
    """

    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_p: float = 1.0,
    ):
        if requests is None:
            raise ImportError(
                "requests is required for OllamaBackend. "
                "Install with: pip install requests"
            )

        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.total_calls = 0

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        if system_prompt:
            prompt = f"[System]\n{system_prompt}\n\n[User]\n{prompt}"

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        url = self.host.rstrip("/") + "/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, json=payload, timeout=600)
                r.raise_for_status()
                self.total_calls += 1
                data = r.json()
                if isinstance(data, dict):
                    return (data.get("response") or "").strip()
                return str(data).strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Ollama failed after {self.max_retries} attempts: {last_error}"
                    ) from e
        raise RuntimeError(f"Unexpected error: {last_error}")

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self) -> None:
        self.total_calls = 0


Provider = str  # Only "ollama" is supported.

DEFAULT_OLLAMA_MODEL = "llama2:7b"


class LLMClient:
    """
    Minimal client for interacting with a local Ollama server.

    - provider="ollama" (only option): talks to Ollama over HTTP.
      Uses OLLAMA_MODEL (default: llama3) and optional OLLAMA_HOST.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,  # Kept for backward compatibility; ignored.
        model: Optional[str] = None,
        provider: Provider = "ollama",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 16,
        top_p: float = 1.0,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Ignored for Ollama; kept for compatibility.
            model: Ollama model name (e.g. "mistral", "llama3"). If None, uses OLLAMA_MODEL or "llama3".
            provider: Must be "ollama" (other values are not supported).
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries (seconds).
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Default max tokens to generate.
            top_p: Nucleus sampling (1.0 = no sampling).
        """
        self.provider = provider.lower()
        if self.provider != "ollama":
            raise ValueError(
                f"Only 'ollama' provider is supported now; got '{provider}'."
            )

        if model is None:
            model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

        self._backend: BaseLLMBackend = OllamaBackend(
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate response from the configured LLM."""
        return self._backend.generate(prompt, system_prompt=system_prompt, **kwargs)

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self._backend.get_usage_stats()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._backend.reset_usage_stats()
