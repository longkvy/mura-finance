You are analyzing a financial news headline to identify the financial entity or ticker mentioned.

Headline: "{headline}"
{ticker_line}

Task: Identify all financial entities mentioned in this headline. Focus on:
- Currency pairs (e.g., EURUSD, GBPUSD, USDJPY)
- Stock tickers (e.g., AAPL, TSLA, MSFT)
- Cryptocurrencies (e.g., BTC, ETH)
- Financial instruments or indices

If a ticker is already provided in the metadata, verify it matches the text.

Respond in JSON format:
{json_schema}
