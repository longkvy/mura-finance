"""
LLM client supporting multiple backends: Ollama (local) and Flan-T5 (Hugging Face).

- provider="ollama": talks to a local Ollama server (default).
- provider="flan_t5": runs google/flan-t5-xxl (or another Flan-T5 model) via Hugging Face
  Transformers; not available through Ollama. Requires: pip install transformers torch
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


class FlanT5Backend(BaseLLMBackend):
    """
    Run Flan-T5 (e.g. google/flan-t5-xxl) via Hugging Face Transformers.

    Uses AutoModelForSeq2SeqLM + AutoTokenizer (no pipeline task registry),
    so it works across transformers versions including those where
    "text2text-generation" is not a registered task. Model loads lazily
    on first generate(). Requires: pip install transformers torch
    """

    DEFAULT_MODEL = "google/flan-t5-xxl"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_p: float = 1.0,
    ):
        self.model_id = model
        self._device = device  # None => auto (cuda if available)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self.total_calls = 0

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Flan-T5 backend requires: pip install transformers torch. "
                "Optional GPU: pip install accelerate"
            ) from e
        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, **kwargs
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._device_str = device

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        if system_prompt:
            prompt = f"[System]\n{system_prompt}\n\n[User]\n{prompt}"

        self._load_model()
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        do_sample = temperature > 0

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": 1,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        import torch
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        self.total_calls += 1
        decoded = self._tokenizer.decode(
            out[0], skip_special_tokens=True
        ).strip()
        return decoded

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self) -> None:
        self.total_calls = 0


Provider = str  # "ollama" | "flan_t5"

DEFAULT_OLLAMA_MODEL = "llama2:7b"
DEFAULT_FLAN_T5_MODEL = "google/flan-t5-xxl"


class LLMClient:
    """
    Unified LLM client for the 5-hop pipeline.

    - provider="ollama": local Ollama server. Uses OLLAMA_MODEL and OLLAMA_HOST.
    - provider="flan_t5": Flan-T5 via Hugging Face (e.g. google/flan-t5-xxl).
      Not available on Ollama. Requires: pip install transformers torch
    """

    def __init__(
        self,
        api_key: Optional[
            str
        ] = None,  # Kept for compatibility; ignored for ollama/flan_t5.
        model: Optional[str] = None,
        provider: Provider = "ollama",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 16,
        top_p: float = 1.0,
        device: Optional[str] = None,  # For flan_t5: "cuda", "cpu", or None (auto).
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Ignored; kept for compatibility.
            model: For ollama: model name (e.g. "mistral"). For flan_t5: HuggingFace model id
                (e.g. "google/flan-t5-xxl"). If None, uses OLLAMA_MODEL or DEFAULT_FLAN_T5_MODEL.
            provider: "ollama" or "flan_t5".
            max_retries: Retry attempts (Ollama only).
            retry_delay: Delay between retries in seconds (Ollama only).
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Default max tokens to generate (applies to all hops unless overridden).
            top_p: Nucleus sampling (1.0 = no sampling).
            device: For flan_t5 only: "cuda", "cpu", or None (auto: cuda if available).
        """
        self.provider = provider.lower()
        if self.provider not in ("ollama", "flan_t5"):
            raise ValueError(
                f"provider must be 'ollama' or 'flan_t5'; got '{provider}'."
            )

        if model is None:
            model = (
                os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
                if self.provider == "ollama"
                else os.getenv("FLAN_T5_MODEL", DEFAULT_FLAN_T5_MODEL)
            )

        if self.provider == "ollama":
            self._backend: BaseLLMBackend = OllamaBackend(
                model=model,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        else:
            self._backend = FlanT5Backend(
                model=model,
                device=device,
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
