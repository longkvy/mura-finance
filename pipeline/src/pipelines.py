"""
pipelines.py
------------
Three inference strategies for FX sentiment analysis:

    1. single_prompt_pipeline  – zero-shot, one call per sample
    2. multihop_pipeline       – 4-hop chain-of-thought (no external context)
    3. hybrid_rag_pipeline     – hybrid direct/indirect + optional RAG context

All functions rely on `src.inference.generate_text`, which must be initialised
with a pipeline before calling these functions.
"""

import re
from typing import Optional

from src.config import MAX_NEW_TOKENS_SHORT, MAX_NEW_TOKENS_HOP1
from src.inference import generate_text
from src.prompts import (
    prompt_only,
    direct_few_shot_prompt,
    hop1_fx_insight_prompt,
    hop2_base_currency_prompt,
    hop3_quote_currency_prompt,
    hop4_final_classification_prompt,
)

VALID_LABELS = {"Positive", "Negative", "Neutral"}


def _safe_label(raw: str, fallback: str = "Neutral") -> str:
    # First try the first token (fast path for FLAN-T5)
    first = raw.strip().split()[0] if raw.strip() else ""
    if first in VALID_LABELS:
        return first

    # Search anywhere in the text (for verbose models like Qwen, Gemma)
    match = re.search(r"\b(Positive|Negative|Neutral)\b", raw, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – Single Prompt
# ─────────────────────────────────────────────────────────────────────────────

def single_prompt_pipeline(title: str, ticker: str) -> dict:
    """
    Zero-shot single-step classification.

    Returns
    -------
    dict with keys: final
    """
    prompt = prompt_only(title, ticker)
    raw = generate_text(prompt, MAX_NEW_TOKENS_SHORT)
    return {"final": _safe_label(raw)}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 – Multi-hop (no context)
# ─────────────────────────────────────────────────────────────────────────────

def multihop_pipeline(title: str, ticker: str) -> dict:
    """
    4-hop chain-of-thought pipeline (no external context).

    Hops:
        1. Detect directional FX pressure
        2. Base currency sentiment
        3. Quote currency sentiment
        4. Combine → final pair sentiment

    Returns
    -------
    dict with keys: hop1, hop2_bc, hop3_qc, final
    """
    base = ticker[:3]
    quote = ticker[3:]

    hop1 = generate_text(hop1_fx_insight_prompt(title, ticker), MAX_NEW_TOKENS_HOP1).strip()
    hop2 = _safe_label(generate_text(hop2_base_currency_prompt(title, ticker, base, hop1), MAX_NEW_TOKENS_SHORT))
    hop3 = _safe_label(generate_text(hop3_quote_currency_prompt(title, ticker, quote), MAX_NEW_TOKENS_SHORT))
    final = _safe_label(generate_text(hop4_final_classification_prompt(ticker, hop2, hop3), MAX_NEW_TOKENS_SHORT))

    return {"hop1": hop1, "hop2_bc": hop2, "hop3_qc": hop3, "final": final}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 – Hybrid (direct + indirect) with optional RAG context
# ─────────────────────────────────────────────────────────────────────────────

def _run_indirect_hops(
    title: str,
    ticker: str,
    in_context: str = "",
    ex_context_1: str = "",
    ex_context_2: str = "",
    ex_context_3: str = "",
) -> dict:
    """Internal helper: run the 4-hop indirect path."""
    base = ticker[:3]
    quote = ticker[3:]

    # Truncate contexts to keep the prompt within model limits
    extra = "\n".join(filter(None, [
        in_context[:150]
    ]))

    hop1 = generate_text(hop1_fx_insight_prompt(title, ticker), MAX_NEW_TOKENS_HOP1).strip()
    hop2 = _safe_label(generate_text(
        hop2_base_currency_prompt(title, ticker, base, hop1, extra_context=extra),
        MAX_NEW_TOKENS_SHORT,
    ))
    hop3 = _safe_label(generate_text(hop3_quote_currency_prompt(title, ticker, quote), MAX_NEW_TOKENS_SHORT))
    final = _safe_label(generate_text(hop4_final_classification_prompt(ticker, hop2, hop3), MAX_NEW_TOKENS_SHORT))

    return {"mode": "indirect", "hop1": hop1, "hop2_bc": hop2, "hop3_qc": hop3, "final": final}


def hybrid_rag_pipeline(
    title: str,
    ticker: str,
    in_context: str = "",
    ex_context_1: str = "",
    ex_context_2: str = "",
    ex_context_3: str = "",
) -> dict:
    """
    Hybrid pipeline:
      - DIRECT headlines (contain the ticker symbol) → few-shot single-step.
      - INDIRECT headlines → 4-hop chain with optional RAG context.

    Parameters
    ----------
    title, ticker : str
        Headline text and currency pair symbol (e.g. "EURUSD").
    in_context : str
        In-domain context string (from dataset `text` column).
    ex_context_1/2/3 : str
        External context strings retrieved via RAG.

    Returns
    -------
    dict with keys: mode, hop1, hop2_bc, hop3_qc, final
    """
    if re.search(rf"\b{ticker}\b", title):
        raw = generate_text(direct_few_shot_prompt(title, ticker), MAX_NEW_TOKENS_SHORT).strip()
        prediction = _safe_label(raw)
        return {"mode": "direct", "hop1": None, "hop2_bc": None, "hop3_qc": None, "final": prediction}

    return _run_indirect_hops(title, ticker, in_context, ex_context_1, ex_context_2, ex_context_3)
