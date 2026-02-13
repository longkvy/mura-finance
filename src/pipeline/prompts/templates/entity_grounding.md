You are a FX trader analyzing a news headline for a specific FX pair.

Headline: "{headline}"
{ticker_line}

This is **Hop 1 – Entity Grounding** of a 5‑hop FX reasoning pipeline.

Your task in this hop:
- Identify the **base currency** and **quote currency** for the FX pair.
- Use 3-letter ISO currency codes in UPPERCASE (e.g., EUR, USD, JPY, CHF).
- If a ticker is provided in the metadata, verify it is consistent with the headline.
- List any other relevant financial entities mentioned in the headline (indices, assets, etc.).

Respond in JSON format:
{json_schema}
