# Prompt templates (raw text / markdown)

**Edit these files** to change what the LLM sees. No Python required.

- **`*.md`** — Main prompt text (markdown-friendly). Edit freely; keep placeholders as-is.
- **`*_schema.txt`** — Expected JSON output shape. Change only if you also update the hop’s `parse_response` logic.

## Placeholders (do not remove)

| Placeholder | Used in | Meaning |
|-------------|---------|---------|
| `{headline}` | All | The financial news headline |
| `{ticker_line}` | entity_grounding | Optional line e.g. "Provided ticker (metadata): EURUSD", or empty |
| `{previous_reasoning}` | financial_aspect, implicit_cue, sentiment_inference, market_implication | Prior hops’ analysis, or "None" |
| `{sentiment}` | market_implication | Sentiment from Hop 4, or "Not yet determined" |
| `{json_schema}` | All | Injected from the `*_schema.txt` file; don’t add this yourself |

## Hop ↔ files

| Hop | Prompt | Schema |
|-----|--------|--------|
| 1. Entity Grounding | `entity_grounding.md` | `entity_grounding_schema.txt` |
| 2. Financial Aspect | `financial_aspect.md` | `financial_aspect_schema.txt` |
| 3. Implicit Cue | `implicit_cue.md` | `implicit_cue_schema.txt` |
| 4. Sentiment Inference | `sentiment_inference.md` | `sentiment_inference_schema.txt` |
| 5. Market Implication | `market_implication.md` | `market_implication_schema.txt` |

## After editing

Run the pipeline or `notebooks/03_5hop_reasoning_test.ipynb` to verify behaviour.
