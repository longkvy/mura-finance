"""
model_loader.py
---------------
Loads any HuggingFace model and returns a text-generation pipeline.
Automatically detects whether the model is seq2seq (T5-family) or
causal LM (Gemma, Qwen, Mistral, etc.) and uses the correct class.

Usage:
    from src.model_loader import load_model
    pipe = load_model()                                  # default FLAN-T5-XXL
    pipe = load_model(model_id="google/gemma-3-27b-it")  # causal LM
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)

from src.config import MODEL_ID, BATCH_SIZE

# Model architectures that use seq2seq (encoder-decoder)
SEQ2SEQ_TYPES = {"t5", "mt5", "bart", "pegasus", "mbart", "longt5"}


def _is_seq2seq(model_id: str) -> bool:
    """Detect model type from config to choose the right AutoModel class."""
    config = AutoConfig.from_pretrained(model_id)
    return config.model_type.lower() in SEQ2SEQ_TYPES


def load_model(model_id: str = MODEL_ID, batch_size: int = BATCH_SIZE):
    """
    Load tokenizer and model, return a text-generation pipeline.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    batch_size : int
        Batch size for the pipeline.

    Returns
    -------
    pipe : transformers.Pipeline
        Ready-to-use text-generation pipeline.
    """
    print(f"[model_loader] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    seq2seq = _is_seq2seq(model_id)
    model_cls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
    task = "text2text-generation" if seq2seq else "text-generation"

    print(f"[model_loader] Loading model: {model_id}  (dtype=float16, device_map=auto, task={task})")
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        batch_size=batch_size,
    )

    print("[model_loader] Model ready.")
    return pipe