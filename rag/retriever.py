from .config import RAGConfig


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
