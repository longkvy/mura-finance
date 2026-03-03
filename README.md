# FX Sentiment Analysis — Multi-hop Reasoning for FX Headlines

**Master Capstone Project · UA MSIS**

---

## Project Overview

This project evaluates multi-hop chain-of-thought prompting strategies for classifying **FX headline sentiment** (Positive / Neutral / Negative), benchmarked across multiple LLMs against a FinBERT baseline.

Three prompting strategies are implemented:

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Multi-hop** | 4-hop chain-of-thought (no external context) |
| 2 | **Hybrid** | Direct-vs-indirect routing, no context |
| 3 | **Hybrid RAG** | Hybrid routing + distilled RAG context |

Each strategy is benchmarked across three models:

| Model | Backend | Notes |
|-------|---------|-------|
| FLAN-T5-XXL | HuggingFace (local) | Dev + Test, all strategies |
| GPT-3.5-Turbo | OpenAI API | Test only, multi-hop + hybrid + hybrid RAG |

---

## Folder Structure

```
repo/
│
└── pipeline/
    │
    ├── pipeline.ipynb          ← Main notebook: FLAN-T5 full run (dev + test)
    ├── test_api_models.ipynb   ← API model evaluation (test set only)
    │
    ├── src/
    │   ├── config.py           ← All paths, column names, model settings
    │   ├── model_loader.py     ← Loads FLAN-T5-XXL and returns a HF pipeline
    │   ├── inference.py        ← generate_text() — unified interface for all backends
    │   ├── prompts.py          ← All prompt templates (easily editable)
    │   ├── pipelines.py        ← The three strategy implementations
    │   └── evaluation.py       ← Metrics, classification report, confusion matrix
    │
    ├── data/
    │   └── finance/
    │       ├── dev.csv                        ← Development set (input)
    │       ├── test.csv                       ← Test set (input)
    │       ├── all_predictions.csv            ← Merged per-row predictions (all strategies, FLAN-T5)
    │       ├── result_dev_*.csv               ← FLAN-T5 dev results per strategy
    │       ├── result_test_*.csv              ← FLAN-T5 test results per strategy
    │       ├── result_test_gpt35_*.csv        ← GPT-3.5 test results per strategy
    │       └── comparison_all_models.csv      ← Final cross-model summary table
    │
    └── requirements.txt
```

---

## Quickstart

### FLAN-T5-XXL — Google Colab (GPU required)

1. Upload the `pipeline/` folder to **Google Drive**.
2. Open `pipeline.ipynb` in Colab (set runtime to **GPU**).
3. Set the working directory path in **Section 0**.
4. Run all cells top-to-bottom (`Runtime → Run all`).

### API Models — OpenAI or Gemini

1. Open `test_api_models.ipynb` in Colab (no GPU needed).
2. Fill in your API keys in **Section 1**:

```python
OPENAI_API_KEY = "sk-..."
```

3. Run all cells. Results are saved to `data/finance/` and a cross-model comparison table is printed at the end.

> **Note:** Free-tier Gemini is limited to 15 RPM. A `time.sleep(4)` is added between rows automatically. For OpenAI, automatic retry with exponential backoff handles rate limit errors.
> 

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

### Hybrid RAG
The Hybrid RAG pipeline extends the indirect path by feeding truncated context directly into **Hop 2** (base currency sentiment), where external signals are most useful for disambiguating indirect headlines.

```
Hop 1 → detect FX directional pressure
Hop 2 → base currency sentiment (+ truncated RAG context injected here)
Hop 3 → quote currency sentiment
Hop 4 → combine BC/QC → final prediction
```

The `text` (in-domain article body) and `external_context_1/2/3` columns are truncated and concatenated before being passed to Hop 2. If no context is available, it falls back to the non-RAG hybrid behaviour.

---

## Output Files

| File | Description |
|------|-------------|
| `all_predictions.csv` | One row per headline with all strategy predictions, hop outputs, and context columns (FLAN-T5, dev set) |
| `result_dev_*.csv` / `result_test_*.csv` | FLAN-T5 per-strategy results |
| `result_test_gpt35_*.csv` | GPT-3.5 test results per strategy |
| `comparison_all_models.csv` | Accuracy + Macro F1 for all models × strategies, ordered FLAN-T5 → GPT-3.5 → Gemini |

---

## Configuration

All tunable settings are in `src/config.py`:

```python
MODEL_ID   = "google/flan-t5-xxl"   # HuggingFace model (local backend)
BATCH_SIZE = 8
DEV_PATH   = "data/finance/dev.csv"
TEST_PATH  = "data/finance/test.csv"
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
| `external_context_1/2/3` | External RAG context snippets |

---

## Dependencies

```bash
pip install -r requirements.txt
```

API backends require their respective packages:

```bash
pip install openai              # for OpenAI
```
