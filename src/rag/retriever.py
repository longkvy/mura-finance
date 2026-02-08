from typing import List, Dict
from .config import RAGConfig
from .vector_store import ChromaVectorStore


class VectorRAGRetriever:
    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.vs = vector_store

    def retrieve(self, query_embedding, date):
        return self.vs.search(
            query_embedding,
            date=date,
            top_k=self.config.top_k,
        )

