from dataclasses import dataclass


@dataclass
class RAGConfig:
    # Retrieval window
    mode: str = "same_day"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3

    # Safety limits
    max_chars_per_doc: int = 1500
    max_external_docs: int = 200  # hard cap after filtering

    # Debug
    verbose: bool = False
