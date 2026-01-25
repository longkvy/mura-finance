# Prompt templates — finance expert review

The **LLM prompt text** for each hop lives in **plain markdown (`.md`) and text (`.txt`)** under `templates/`. Edit those files to change what the model sees; no Python needed.

## Where to edit

| Hop | Prompt (edit this) | JSON schema (edit only if you know the implications) |
|-----|--------------------|------------------------------------------------------|
| 1. Entity Grounding | `templates/entity_grounding.md` | `templates/entity_grounding_schema.txt` |
| 2. Financial Aspect | `templates/financial_aspect.md` | `templates/financial_aspect_schema.txt` |
| 3. Implicit Cue | `templates/implicit_cue.md` | `templates/implicit_cue_schema.txt` |
| 4. Sentiment Inference | `templates/sentiment_inference.md` | `templates/sentiment_inference_schema.txt` |
| 5. Market Implication | `templates/market_implication.md` | `templates/market_implication_schema.txt` |

See **`templates/README.md`** for placeholder names (`{headline}`, `{previous_reasoning}`, etc.) and rules.

## Python side

`_loader.py` loads templates and fills placeholders. The **hops** (in `src/pipeline/hops/`) call it when building prompts. Edit loader or hop logic only if you add new placeholders.

## After editing

Run the pipeline or `notebooks/03_5hop_reasoning_test.ipynb` to verify behaviour.

## Questions for finance experts

- Are the entity types (currency pairs, tickers, crypto, instruments) complete for your use case?
- Should we add or remove any aspect categories (e.g. inflation, rates, earnings)?
- Is the implicit-cue wording (hedging, euphemisms, etc.) well aligned with your domain?
- Any domain-specific terms or examples to add to the prompts?
