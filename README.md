# MURA-Finance: Adaptive Multi-Hop Reasoning with Augmented Context for Implicit Financial Sentiment Analysis

**Master Capstone Project · UA MSIS**

---

## Project Overview

This project evaluates multi-hop chain-of-thought prompting strategies for classifying **FX headline sentiment** (Positive / Neutral / Negative), benchmarked across multiple LLMs against a FinBERT baseline.

Four prompting strategies are implemented:

| # | Strategy | Description |
|---|----------|-------------|
| 0 | **Single Prompt** | Zero-shot single-step classification |
| 1 | **Multi-hop** | 4-hop chain-of-thought (no external context) |
| 2 | **Hybrid** | Direct-vs-indirect routing, no context |
| 3 | **Hybrid RAG** | Hybrid routing + distilled RAG context |

Each strategy is benchmarked across two models:

| Model | Backend | Notes |
|-------|---------|-------|
| FLAN-T5-XXL | HuggingFace (local) | Dev + Test, all strategies |
| GPT-3.5-Turbo | OpenAI API | Test only, all strategies |

---

## Folder Structure

```
repo/
│
├── pipeline.ipynb              ← Main notebook: FLAN-T5 full run (dev + test)
├── test_api_models.ipynb       ← API model evaluation (test set only)
│
├── src/
│   ├── config.py               ← All paths, column names, model settings
│   ├── model_loader.py         ← Loads FLAN-T5-XXL and returns a HF pipeline
│   ├── inference.py            ← generate_text() — unified interface for all backends
│   ├── prompts.py              ← All prompt templates (easily editable)
│   ├── pipelines.py            ← The four strategy implementations
│   └── evaluation.py          ← Metrics, classification report, confusion matrix
│
├── rag/
│   ├── __init__.py
│   ├── build_external_corpus_2023.py  ← Script to build external news parquet
│   ├── config.py               ← RAG settings (top_k, embedding model, etc.)
│   ├── external_corpus.py      ← Loads and filters external news parquet
│   ├── local_corpus.py         ← Loads in-domain context from dataset
│   ├── rag_run.ipynb           ← Notebook to run RAG retrieval and enrich dataset
│   ├── retriever.py            ← VectorRAGRetriever — wraps vector store search
│   ├── time_utils.py           ← Date parsing utilities
│   └── vector_store.py         ← ChromaDB indexing and semantic search
│
├── data/
│   └── finance/
│       ├── dev.csv                                    ← Development set (input)
│       ├── test.csv                                   ← Test set (input)
│       ├── sentiment_annotated_with_texts.csv         ← Raw annotated data
│       ├── sentiment_annotated_with_texts_context.csv ← Annotated data + RAG context
│       ├── split_dev_test.py                          ← Script to split data into dev/test
│       ├── all_predictions_dev.csv                    ← All strategy predictions on dev set (FLAN-T5)
│       ├── result_dev_*.csv                           ← FLAN-T5 per-strategy dev results (4 files)
│       ├── result_test_*.csv                          ← FLAN-T5 per-strategy test results (4 files)
│       ├── result_test_gpt35_*.csv                    ← GPT-3.5 per-strategy test results (4 files)
│       ├── comparison_flan_xxl.csv                    ← FLAN-T5 dev + test summary
│       └── comparison_all_models.csv                  ← Final cross-model summary (FLAN-T5 + GPT-3.5)
│
├── docs/                        ← Project reports and reference documents
├── .gitignore
├── allfiles.txt
└── requirements.txt
```

---

## Quickstart

### FLAN-T5-XXL — Google Colab (GPU required)

1. Upload the repo folder to **Google Drive**.
2. Open `pipeline.ipynb` in Colab (set runtime to **A100 GPU**).
3. Set the working directory path in **Section 0**.
4. Run all cells top-to-bottom (`Runtime → Run all`).

> **Note:** Sections 0–8 run on the **dev set only** for prompt development. Section 9 runs the **test set once** after strategies are finalized.

### API Models — OpenAI

1. Open `test_api_models.ipynb` in Colab (no GPU needed).
2. Fill in your API key in **Section 1**:

```python
OPENAI_API_KEY = "sk-..."
```

3. Run all cells. Results are saved to `data/finance/` and a cross-model comparison table is printed at the end.

> **Note:** Automatic retry with exponential backoff handles rate limit errors. A checkpoint is saved every 50 rows in case of interruption.

### RAG Context Retrieval (preprocessing — run once)

1. Open `rag/rag_run.ipynb` in Colab (GPU recommended for embedding).
2. Run all cells to build the ChromaDB index and retrieve external context for each headline.
3. Output is saved to `data/finance/sentiment_annotated_with_texts_context.csv`.

---

## Reloading After Code Changes

Python caches imported modules. After editing any file in `src/`, run this cell before re-running the pipeline:

```python
import importlib, src.prompts, src.inference, src.pipelines

importlib.reload(src.prompts)
importlib.reload(src.inference)
importlib.reload(src.pipelines)

src.inference.set_pipeline(pipe)  # re-inject after reload (HF only)

from src.pipelines import single_prompt_pipeline, multihop_pipeline, hybrid_rag_pipeline
print("src/ reloaded.")
```

---

## Pipeline Design

### Hybrid Routing
Both the Hybrid and Hybrid RAG pipelines use a **direct vs. indirect routing** step before running any hops. If the headline explicitly mentions the ticker symbol (e.g. `EURUSD`), it is classified as **direct** and handled with a single few-shot prompt. If not, it is classified as **indirect** and goes through the full 4-hop chain.

```
Headline contains ticker?
    ├── YES (direct)  → few-shot single-step classification
    └── NO (indirect) → 4-hop chain-of-thought
```

### Multi-hop Chain
```
Hop 1 → detect FX directional pressure for base currency
Hop 2 → base currency sentiment (+ truncated RAG context injected here)
Hop 3 → quote currency sentiment
Hop 4 → combine BC/QC → final prediction
```

### Hybrid RAG
Extends the indirect path by injecting truncated external context into **Hop 2** where it is most useful for disambiguating implicit signals. Falls back to non-RAG hybrid behaviour if no context is available.

---

## Output Files

| File | Description |
|------|-------------|
| `result_dev_*.csv` | FLAN-T5 per-strategy dev results |
| `result_test_*.csv` | FLAN-T5 per-strategy test results |
| `result_test_gpt35_*.csv` | GPT-3.5 test results per strategy |
| `comparison_all_models.csv` | Accuracy + Macro F1 for all models × strategies |

---

## Configuration

All tunable settings are in `src/config.py`:

```python
MODEL_ID   = "google/flan-t5-xxl"
BATCH_SIZE = 8
DEV_PATH   = "data/finance/dev.csv"
TEST_PATH  = "data/finance/test.csv"
```

RAG settings are in `rag/config.py`:

```python
embedding_model  = "sentence-transformers/all-MiniLM-L6-v2"
top_k            = 3       # number of external articles retrieved per headline
max_chars_per_doc = 1500   # character limit per retrieved document
```

---

## Data Format

Each input CSV must contain:

| Column | Description |
|--------|-------------|
| `title` | Headline text |
| `ticker` | FX pair symbol (e.g. `EURUSD`) |
| `true_sentiment` | Ground-truth label (`Positive` / `Neutral` / `Negative`) |
| `finbert_sentiment` | FinBERT baseline prediction |
| `text` | In-domain context (original article body) |
| `external_context_1/2/3` | External RAG context snippets (from RAG retrieval) |

---

## Dependencies

```bash
pip install -r requirements.txt
```

API backends require:

```bash
pip install openai
```
