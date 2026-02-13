You are a FX trader analyzing a news headline for a specific FX pair.

Headline: "{headline}"
Previous analysis: {previous_reasoning}
Sentiment (from Hop 4): {sentiment}

This is **Hop 5 – Market Implication** of a 5‑hop FX reasoning pipeline.

Your task in this hop:
- Based on the full reasoning so far (entities, financial aspect, implicit cues, and sentiment),
  infer the **market implication for the FX pair**.
- Interpret the sentiment in terms of **price direction for the base currency vs the quote currency**:
  - Bullish: base currency expected to strengthen against the quote.
  - Bearish: base currency expected to weaken against the quote.
  - Uncertain: no clear directional bias (mixed or purely descriptive).
- Clearly explain how the earlier hops lead to this directional view (or lack of it).

Classify the `market_implication` as one of:
- Bullish
- Bearish
- Uncertain

Respond in JSON format:
{json_schema}
