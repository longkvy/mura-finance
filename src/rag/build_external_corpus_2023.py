###
# Run once to collect 2023 data and save to financial_news_2023.parquet
###

from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

DATASET = "Brianferrell787/financial-news-multisource"
OUT_PATH = "financial_news_2023.parquet"


def is_2023(date_str: str) -> bool:
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.year == 2023
    except Exception:
        return False


def main():
    print("Streaming dataset (no full download)...")

    ds = load_dataset(
        DATASET,
        split="train",
        streaming=True,
    )

    rows = []
    kept = 0

    for item in ds:
        date = item.get("date")

        if not date:
            continue

        if is_2023(date):
            rows.append({"date": date, "text": item.get("text")})
            kept += 1

            print(kept)

    print(f"Collected {len(rows)} rows from 2023")

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT_PATH)

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
