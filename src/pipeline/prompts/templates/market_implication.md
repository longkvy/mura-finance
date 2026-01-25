You are analyzing a financial news headline to infer its market implications.

Headline: "{headline}"
Previous analysis: {previous_reasoning}
Sentiment: {sentiment}

Task: Based on the complete analysis (entity, financial aspect, implicit cues, and sentiment), determine the market implication:
- Bullish: Suggests upward price movement or positive outlook
- Bearish: Suggests downward price movement or negative outlook
- Uncertain: Mixed signals, hedging, or unclear implications

Consider:
- The specific financial aspect (e.g., inflation news may have different implications than earnings)
- The strength of the sentiment (strong positive vs. weak positive)
- The presence of hedging or uncertainty cues

Respond in JSON format:
{json_schema}
