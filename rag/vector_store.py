import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ChromaVectorStore:
    COLLECTION_NAME = "external_corpus"

    def __init__(self, model_name: str):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.model = SentenceTransformer(model_name, device="cuda")
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
        )

    def build_global_index(
        self,
        texts: list[str],
        metadatas: list[dict],
        batch_size: int = 512,
    ):
        assert len(texts) == len(metadatas)

        # Prevent duplicate indexing
        if self.collection.count() > 0:
            print("⚠️ Collection already populated. Skipping build.")
            return

        for i in tqdm(range(0, len(texts), batch_size), desc="Adding to Chroma"):
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = [str(j) for j in range(i, i + len(batch_texts))]

            embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
            ).tolist()

            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids,
            )

    def search(self, query_embedding, date, top_k):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"date": date},
        )
        return results["documents"][0]
