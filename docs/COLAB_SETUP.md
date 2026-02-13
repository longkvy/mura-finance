# Running the 5-hop pipeline with Ollama on Google Colab

## Quick start

1. **Get your project into Colab**
   - **Option A:** Push the repo to GitHub, then in Colab run:
     ```bash
     !git clone https://github.com/YOUR_USER/uoa-group1-c6.git /content/uoa-group1-c6
     %cd /content/uoa-group1-c6
     ```
   - **Option B:** Zip the project on your machine, then in Colab: **File → Upload** the zip, unzip it, and open `notebooks/06_colab_ollama_setup.ipynb`.

2. **Open the Colab notebook**
   - Open `notebooks/06_colab_ollama_setup.ipynb` in Colab (upload it or open from the cloned repo).

3. **Run all cells in order**
   - Cell 2 installs Ollama (Linux) and starts the server.
   - Cell 3 pulls a small model (e.g. `gemma3:1b`). Use a small model to stay within Colab’s ~12GB RAM.
   - Cells 4–7 install deps, create the pipeline, run on a sample, and optionally evaluate.

## What the notebook does

| Step | Action |
|------|--------|
| 1 | (Optional) Clone repo from GitHub |
| 2 | Install Ollama with the official script and start `ollama serve` in the background |
| 3 | `ollama pull gemma3:1b` (or another small model) |
| 4 | `pip install -r requirements.txt` in project root |
| 5 | Set `sys.path`, `OLLAMA_HOST` / `OLLAMA_MODEL`, and create `LLMClient` + `ReasoningPipeline` |
| 6 | Load `sentiment_predictions_allday_articles.csv`, sample rows, run 5-hop pipeline, add `hop_sentiment` and `hop_sentiment_score` |
| 7 | (Optional) Compare to `gpt_sentiment_p6` with `compute_classification_metrics` |

## Requirements in Colab

- **Data:** `sentiment_predictions_allday_articles.csv` must be in the project root (it’s in the repo).
- **RAM:** Use a small model (e.g. `gemma3:1b`, `llama3.2:1b`, `phi3:mini`). Larger models may OOM on free Colab.
- **Env:** Defaults are `OLLAMA_HOST=http://localhost:11434` and `OLLAMA_MODEL=gemma3:1b`. You can set these in the notebook or in a `.env` file in the project root.

## Using a remote Ollama server instead

If Ollama runs on another machine (e.g. your laptop or a cloud VM):

1. Ensure that machine serves Ollama (e.g. `OLLAMA_HOST=0.0.0.0 ollama serve` or use a tunnel like ngrok).
2. In Colab, set the host before creating the client:
   ```python
   import os
   os.environ["OLLAMA_HOST"] = "http://YOUR_SERVER:11434"
   ```
3. Skip the “Install Ollama” and “Pull model” cells and run from the “Install Python dependencies” cell onward.

## Troubleshooting

- **“Connection refused” to Ollama:** Wait a few seconds after starting `ollama serve`, or run the “start server” cell again.
- **Out of memory:** Use a smaller model (`gemma3:1b`, `llama3.2:1b`) or reduce the sample size in the run cell.
- **ModuleNotFoundError for `src`:** Ensure you ran the cell that sets `ROOT` and `sys.path.insert(0, str(ROOT))` and that you’re in the project root (e.g. `/content/uoa-group1-c6`).
