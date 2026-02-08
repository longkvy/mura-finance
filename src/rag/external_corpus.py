import pyarrow.parquet as pq
from typing import List, Dict, Tuple


class ExternalCorpus:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path

    def filter_by_time(self, published_at):
        day = published_at.strftime("%Y-%m-%d")

        start = f"{day}T00:00:00Z"
        end = f"{day}T23:59:59Z"

        table = pq.read_table(
            self.parquet_path,
            filters=[
                ("date", ">=", start),
                ("date", "<=", end),
            ],
            columns=["text", "date"],
        )

        return table.to_pylist()

    def get_all_texts_and_metadata(self) -> Tuple[List[str], List[Dict]]:
        """
        Load ALL external docs ONCE for global indexing
        """
        table = pq.read_table(
            self.parquet_path,
            columns=["text", "date"],
        )

        rows = table.to_pylist()

        texts = []
        metadatas = []

        for row in rows:
            text = row.get("text")
            date = row.get("date")

            if isinstance(text, str) and isinstance(date, str):
                texts.append(text)
                metadatas.append({
                    "date": date[:10]  # YYYY-MM-DD
                })

        return texts, metadatas
