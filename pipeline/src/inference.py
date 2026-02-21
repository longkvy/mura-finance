"""
inference.py
------------
Unified text-generation interface supporting multiple backends:

    • "hf"      – HuggingFace local pipeline (FLAN-T5-XXL, default)
    • "openai"  – OpenAI Chat Completions API  (e.g. gpt-3.5-turbo)
    • "gemini"  – Google Gemini API            (e.g. gemini-2.5-flash)

Usage
-----
# Local HuggingFace model (existing behaviour – no change required):
    import src.inference as inference
    inference.set_pipeline(pipe)          # inject loaded HF pipeline
    inference.set_backend("hf")           # default, can be omitted

# OpenAI:
    inference.set_backend("openai", model="gpt-3.5-turbo",  api_key="sk-...")

# Gemini:
    inference.set_backend("gemini", model="gemini-2.5-flash", api_key="AI...")

All three expose the same generate_text(prompt, max_new_tokens) signature,
so pipelines.py / prompts.py / evaluation.py are completely unchanged.
"""

from src.config import MAX_NEW_TOKENS_DEFAULT

# ── shared state ──────────────────────────────────────────────────────────────
_backend: str = "hf"          # "hf" | "openai" | "gemini"
_pipe = None                   # HuggingFace pipeline object
_model_name: str = ""          # model string for API backends
_api_key: str = ""             # API key for API backends


# ── setup helpers ─────────────────────────────────────────────────────────────

def set_pipeline(pipe) -> None:
    """Inject the HuggingFace pipeline (keeps original behaviour)."""
    global _pipe
    _pipe = pipe


def set_backend(backend: str, model: str = "", api_key: str = "") -> None:
    """
    Choose which backend generate_text() will use.

    Parameters
    ----------
    backend  : "hf" | "openai" | "gemini"
    model    : model identifier string (required for openai / gemini)
    api_key  : API key (required for openai / gemini)
    """
    global _backend, _model_name, _api_key
    assert backend in ("hf", "openai", "gemini"), f"Unknown backend: {backend}"
    _backend = backend
    _model_name = model
    _api_key = api_key
    print(f"[inference] Backend set to '{backend}'" + (f"  model={model}" if model else ""))


# ── provider implementations ──────────────────────────────────────────────────

def _generate_hf(prompt: str, max_new_tokens: int) -> str:
    if _pipe is None:
        raise RuntimeError("HF pipeline not set. Call inference.set_pipeline(pipe) first.")
    output = _pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        truncation=True,
        do_sample=False,
        temperature=0.0,
    )
    return output[0]["generated_text"].strip()


def _generate_openai(prompt: str, max_new_tokens: int) -> str:
    """OpenAI Chat Completions (works with gpt-3.5-turbo, gpt-4o, etc.)"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run:  pip install openai")

    client = OpenAI(api_key=_api_key)
    response = client.chat.completions.create(
        model=_model_name or "gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _generate_gemini(prompt: str, max_new_tokens: int) -> str:
    """Google Gemini via google-generativeai SDK."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run:  pip install google-generativeai")

    # Gemini needs more room than FLAN-T5 — enforce a minimum
    max_new_tokens = max(max_new_tokens, 100)

    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel(
        model_name=_model_name or "gemini-2.5-flash",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=0.0,
        ),
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# ── public API ────────────────────────────────────────────────────────────────

def generate_text(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT) -> str:
    """
    Generate text using whichever backend is currently active.

    Parameters
    ----------
    prompt         : str – input prompt
    max_new_tokens : int – maximum tokens to generate

    Returns
    -------
    str – generated text, stripped of whitespace
    """
    if _backend == "hf":
        return _generate_hf(prompt, max_new_tokens)
    elif _backend == "openai":
        return _generate_openai(prompt, max_new_tokens)
    elif _backend == "gemini":
        return _generate_gemini(prompt, max_new_tokens)
    else:
        raise ValueError(f"Unknown backend: {_backend}")
