You are analyzing a financial news headline to infer its implicit sentiment.

Headline: "{headline}"
Previous analysis: {previous_reasoning}

Task: Based on the identified entity, financial aspect, and implicit cues, infer the sentiment even if it's not explicitly stated. Consider:
- The financial aspect and its implications
- Hedging language may indicate uncertainty or caution
- Euphemisms often mask negative sentiment
- Mixed framing suggests neutral or uncertain sentiment

Classify the sentiment as one of: Positive, Negative, Neutral

Respond in JSON format:
{json_schema}
