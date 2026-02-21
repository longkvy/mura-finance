"""
model_loader.py
---------------
Loads the FLAN-T5-XXL model and returns a HuggingFace text-generation pipeline.

Usage:
    from src.model_loader import load_model
    pipe = load_model()
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from src.config import MODEL_ID, BATCH_SIZE


def load_model(model_id: str = MODEL_ID, batch_size: int = BATCH_SIZE):
    """
    Load the tokenizer and model, return a text2text-generation pipeline.

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

    print(f"[model_loader] Loading model: {model_id}  (float16, device_map=auto)")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        batch_size=batch_size,
    )

    print("[model_loader] Model ready.")
    return pipe
