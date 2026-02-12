"""
LLM client for making API calls to OpenAI, Hugging Face, or other providers.

Handles error handling, retries, and token tracking.
Designed to switch between providers (e.g. OpenAI, Flan-T5-XL, Meta Llama 3 8B).
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

# -----------------------------------------------------------------------------
# OpenAI backend
# -----------------------------------------------------------------------------

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class BaseLLMBackend(ABC):
    """Abstract base for LLM backends. All providers implement this interface."""

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


class OpenAIBackend(BaseLLMBackend):
    """
    OpenAI chat completions (gpt-3.5-turbo, gpt-4, etc.).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 16,
        top_p: float = 1.0,
    ):
        if OpenAI is None:
            raise ImportError("openai is required. Install with: pip install openai")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                self.total_calls += 1
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"OpenAI failed after {self.max_retries} attempts: {last_error}"
                    ) from e
        raise RuntimeError(f"Unexpected error: {last_error}")

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def reset_usage_stats(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0


# -----------------------------------------------------------------------------
# Hugging Face Inference API backend (hosted: Flan-T5-XL, Llama 3 8B, etc.)
# -----------------------------------------------------------------------------

try:
    import requests
except ImportError:
    requests = None


class HuggingFaceInferenceBackend(BaseLLMBackend):
    """
    Hugging Face Inference API (hosted).
    Use model IDs such as: google/flan-t5-xl, meta-llama/Meta-Llama-3-8B
    Requires HF token: set HF_TOKEN or pass api_key.
    """

    INFERENCE_URL = "https://api-inference.huggingface.co/models/{model_id}"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        self.model_id = model
        self.token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not self.token:
            raise ValueError(
                "Hugging Face token required. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) or pass api_key."
            )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.total_calls = 0

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        # Many HF models don't support system vs user; prepend system to prompt
        if system_prompt:
            prompt = f"[System]\n{system_prompt}\n\n[User]\n{prompt}"

        url = self.INFERENCE_URL.format(model_id=self.model_id)
        headers = {"Authorization": f"Bearer {self.token}"}
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens)),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "return_full_text": False,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                self.total_calls += 1
                data = r.json()
                # API can return list (e.g. [{"generated_text": "..."}]) or dict
                if isinstance(data, list) and len(data) > 0:
                    return (data[0].get("generated_text") or "").strip()
                if isinstance(data, dict) and "generated_text" in data:
                    return (data["generated_text"] or "").strip()
                return str(data).strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Hugging Face Inference failed after {self.max_retries} attempts: {last_error}"
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


# -----------------------------------------------------------------------------
# Hugging Face Local backend (download once, run offline)
# -----------------------------------------------------------------------------

def _is_seq2seq_model(model_id: str) -> bool:
    """True for T5/Flan-T5 (seq2seq); use model+tokenizer directly to avoid pipeline task."""
    m = model_id.lower()
    return "t5" in m or "flan" in m


try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    hf_pipeline = None
    _TRANSFORMERS_AVAILABLE = False


class HuggingFaceLocalBackend(BaseLLMBackend):
    """
    Run Hugging Face models locally with transformers. No API calls.

    Models are downloaded once from the Hub and cached (default: ~/.cache/huggingface/hub/).
    After the first download you can use them offline.

    - google/flan-t5-xl: loaded with AutoModelForSeq2SeqLM (no pipeline task needed).
    - meta-llama/Meta-Llama-3-8B: loaded with pipeline("text-generation", ...).

    Install: pip install transformers torch
    """

    def __init__(
        self,
        model: str,
        task: Optional[str] = None,
        device: Optional[int | str] = None,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for local HF models. "
                "Install with: pip install transformers torch"
            )
        self.model_id = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.total_calls = 0

        # Device: -1 = CPU, 0 = CUDA, "auto" = use CUDA if available
        if device is None:
            device = os.getenv("HF_DEVICE", "auto")
        if device == "auto":
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1
        self.device = device

        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self._cache_dir = cache_dir
        self._use_seq2seq = _is_seq2seq_model(model)

        if self._use_seq2seq:
            # Flan-T5 / T5: use model + tokenizer directly (avoids pipeline task "text2text-generation")
            model_kwargs: Dict[str, Any] = {}
            if cache_dir:
                model_kwargs["cache_dir"] = cache_dir
            if self.token:
                model_kwargs["token"] = self.token
            self._tokenizer = AutoTokenizer.from_pretrained(model, **model_kwargs)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model, **model_kwargs)
            dev = self.device if isinstance(self.device, int) and self.device >= 0 else -1
            if dev >= 0:
                self._model = self._model.to(dev)
            self._pipe = None
            # Warn when XL runs on CPU (1–3+ min per generate is common)
            if dev < 0 and ("xl" in model.lower() or "large" in model.lower()):
                print("⚠ Flan-T5-XL/ Large on CPU: expect 1–3+ minutes per generate(). Use flan-t5-base for fast tests or set HF_DEVICE=0 for GPU.")
        else:
            # Causal LM (e.g. Llama): use pipeline("text-generation")
            kwargs: Dict[str, Any] = {
                "model": model,
                "device": self.device if isinstance(self.device, int) and self.device >= 0 else -1,
            }
            if cache_dir:
                kwargs["model_kwargs"] = {"cache_dir": cache_dir}
            if self.token:
                kwargs["model_kwargs"] = kwargs.get("model_kwargs", {})
                kwargs["model_kwargs"]["token"] = self.token
            self._pipe = hf_pipeline("text-generation", **kwargs)
            self._model = None  # type: ignore
            self._tokenizer = None  # type: ignore

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        if system_prompt:
            prompt = f"[System]\n{system_prompt}\n\n[User]\n{prompt}"

        max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        do_sample = temperature > 0 or top_p < 1.0

        self.total_calls += 1

        if self._use_seq2seq and self._model is not None and self._tokenizer is not None:
            # Seq2seq (Flan-T5): forward with model.generate
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-7)
                gen_kwargs["top_p"] = top_p
            out_ids = self._model.generate(**inputs, **gen_kwargs)
            text = self._tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # For seq2seq, decode includes input in some setups; strip to generated part only if needed
            if prompt and text.startswith(prompt[:50]):
                text = text[len(prompt):].strip()
            return text.strip()
        else:
            # Causal LM via pipeline
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "return_full_text": False,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-7)
                gen_kwargs["top_p"] = top_p
            out = self._pipe(prompt, **gen_kwargs)
            if isinstance(out, list) and len(out) > 0:
                text = out[0].get("generated_text") or ""
                return text.strip()
            if isinstance(out, dict):
                return (out.get("generated_text") or "").strip()
            return str(out).strip()

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self) -> None:
        self.total_calls = 0


# -----------------------------------------------------------------------------
# llama.cpp backend for local GGUF models (e.g. Mistral 7B Q4)
# -----------------------------------------------------------------------------

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore


class LlamaCppBackend(BaseLLMBackend):
    """
    Local GGUF models via llama.cpp (e.g. Mistral 7B Q4).

    This backend is intended for quantized models such as Mistral‑7B in Q4
    format, loaded from a `.gguf` file on disk.

    Requirements:
        - pip install llama-cpp-python
        - A local GGUF model file (e.g. Mistral-7B-Instruct Q4_*). Point
          `model_path` to this file or set `LLAMA_CPP_MODEL_PATH` /
          `MISTRAL_MODEL_PATH`.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
    ):
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required for local Mistral/LLM GGUF models. "
                "Install with: pip install llama-cpp-python"
            )

        if model_path is None:
            model_path = (
                os.getenv("LLAMA_CPP_MODEL_PATH")
                or os.getenv("MISTRAL_MODEL_PATH")
                or None
            )
        if not model_path:
            raise ValueError(
                "model_path is required for LlamaCppBackend. "
                "Pass it explicitly or set LLAMA_CPP_MODEL_PATH or MISTRAL_MODEL_PATH."
            )

        if n_threads is None:
            # Default: use available CPU cores
            try:
                n_threads = os.cpu_count() or 4
            except Exception:
                n_threads = 4

        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.total_calls = 0

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        if system_prompt:
            prompt = f"[System]\n{system_prompt}\n\n[User]\n{prompt}\n\n[Assistant]\n"

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        self.total_calls += 1

        result = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # llama_cpp returns an OpenAI-like dict with "choices"
        if isinstance(result, dict):
            choices = result.get("choices") or []
            if choices:
                text = choices[0].get("text") or ""
                return text.strip()
        return str(result).strip()

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self) -> None:
        self.total_calls = 0


# -----------------------------------------------------------------------------
# Unified LLMClient (default: OpenAI, optional: huggingface / huggingface_local / llama_cpp)
# -----------------------------------------------------------------------------

Provider = str  # "openai" | "huggingface" | "huggingface_local" | "llama_cpp"

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
# Local/offline HF: "google/flan-t5-xl", "meta-llama/Meta-Llama-3-8B"


class LLMClient:
    """
    Client for interacting with LLM APIs. Supports multiple providers.

    - provider="openai" (default): OpenAI chat (gpt-3.5-turbo, gpt-4). Uses OPENAI_API_KEY, OPENAI_MODEL.
    - provider="huggingface": Hugging Face Inference API (hosted). Uses HF_TOKEN, HF_MODEL.
    - provider="huggingface_local": Run HF models locally (download once, then offline). Uses HF_MODEL.
    - provider="llama_cpp": Local GGUF models via llama.cpp (e.g. Mistral 7B Q4). Uses model path or
      LLAMA_CPP_MODEL_PATH / MISTRAL_MODEL_PATH.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Provider = "openai",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 16,
        top_p: float = 1.0,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Provider API key (OpenAI or HF token depending on provider)
            model: Model name/ID (OpenAI/HF) or local path for llama.cpp (GGUF).
            provider: "openai", "huggingface" (hosted API), "huggingface_local" (download + run offline),
                      or "llama_cpp" (local GGUF via llama.cpp, e.g. Mistral 7B Q4)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Default max tokens to generate (OpenAI); max_new_tokens for HF
            top_p: Nucleus sampling (1.0 = no sampling)
        """
        self.provider = provider.lower()
        if model is None:
            if self.provider == "openai":
                model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
            else:
                model = os.getenv("HF_MODEL", "google/flan-t5-xl")

        if self.provider == "openai":
            self._backend: BaseLLMBackend = OpenAIBackend(
                api_key=api_key,
                model=model,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        elif self.provider == "huggingface":
            self._backend = HuggingFaceInferenceBackend(
                model=model,
                api_key=api_key,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif self.provider == "huggingface_local":
            self._backend = HuggingFaceLocalBackend(
                model=model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                token=api_key,
            )
        elif self.provider == "llama_cpp":
            # For llama.cpp we treat `model` as a local path to a GGUF file (e.g. Mistral 7B Q4),
            # or fall back to LLAMA_CPP_MODEL_PATH / MISTRAL_MODEL_PATH.
            self._backend = LlamaCppBackend(
                model_path=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'openai', 'huggingface', 'huggingface_local', or 'llama_cpp'."
            )

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate response from the configured LLM."""
        return self._backend.generate(prompt, system_prompt=system_prompt, **kwargs)

    def get_usage_stats(self) -> Dict[str, int]:
        """Get token usage statistics (provider-dependent)."""
        return self._backend.get_usage_stats()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._backend.reset_usage_stats()
