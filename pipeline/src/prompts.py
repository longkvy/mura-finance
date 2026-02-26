"""
prompts.py
----------
All prompt templates used across the pipeline strategies.

Each function returns a formatted string ready to be passed to generate_text().
Keeping prompts here makes A/B testing and refinement straightforward.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – Single Prompt
# ─────────────────────────────────────────────────────────────────────────────


def prompt_only(title: str, ticker: str) -> str:
    """Zero-shot single-step classification prompt."""
    return f"""
Act as an expert at forex trading.
Classify the sentiment for ***{ticker}*** based only on the headline '{title}'
Answer in one token: Positive, Negative, or Neutral
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Shared multi-hop building blocks
# ─────────────────────────────────────────────────────────────────────────────


def direct_few_shot_prompt(title: str, ticker: str) -> str:
    """Few-shot prompt for headlines that directly mention the ticker."""
    return f"""
You are classifying FX headline sentiment.

Here are labeled examples from this dataset:

Example 1:
Headline: EURUSD trims gains near 10650
Label: Negative

Example 2:
Headline: GBPUSD recovers a few pips from daily low
Label: Neutral

Example 3:
Headline: EURUSD breaks above 11000 decisively
Label: Positive

Now classify:

Pair: {ticker}
Headline: {title}

Return ONLY:
Positive
Negative
Neutral
""".strip()


def hop1_fx_insight_prompt(title: str, ticker: str) -> str:
    """Hop 1 – Detect directional pressure for both currencies."""
    return f"""
You are an FX signal analyst.

Pair: {ticker}
Headline: {title}

For BOTH currencies in the pair, determine if the headline implies:

Upward pressure
Downward pressure
No clear pressure

Consider:
• Technical language (breaks, trims, downside, bulls, bears, resistance, floor)
• Policy tone (too fast, cautious, tightening, easing)
• Yield movement
• Risk language (woes, concerns, uncertainty)
• Sustainability language (unsustainable, fading)

For each detected pressure, state:
• Currency affected
• Direction of pressure (upward / downward)

If absolutely no directional pressure is implied, state:
No directional pressure detected.

Do NOT classify sentiment.
Return 1–3 short bullet points.
""".strip()


def hop2_base_currency_prompt(
    title: str, ticker: str, base: str, hop1: str, extra_context: str = ""
) -> str:
    """Hop 2 – Classify sentiment for the base currency."""
    context_block = (
        f"\nContext (optional, only choose the most relevant):\n{extra_context}"
        if extra_context
        else ""
    )
    return f"""
Task: Financial Sentiment Analysis.

Input Ticker: {ticker}
Input Headline: {title}
Directional signals: {hop1}{context_block}

Base Currency: {base}

Is the headline Hawkish/Bullish, Dovish/Bearish, or Neutral for {base}?
If unclear, just return Neutral.

Answer ONLY:
Positive
Negative
Neutral
""".strip()


def hop3_quote_currency_prompt(title: str, ticker: str, quote: str) -> str:
    """Hop 3 – Classify sentiment for the quote currency."""
    return f"""
Task: Financial Sentiment Analysis.

Input Ticker: {ticker}
Input Headline: {title}

Quote Currency: {quote}

Is the headline Hawkish/Bullish, Dovish/Bearish, or Neutral for {quote}?
If unclear, just return Neutral.

Answer ONLY:
Positive
Negative
Neutral
""".strip()


def hop4_final_classification_prompt(ticker: str, hop2: str, hop3: str) -> str:
    """Hop 4 – Combine BC/QC sentiments into a single pair-level prediction."""
    return f"""
Task: Financial Sentiment Analysis.

Ticker: {ticker}

BC Sentiment: {hop2}
QC Sentiment: {hop3}

If BC Positive and QC not Positive → Positive
If BC Negative and QC not Negative → Negative
If both same or both Neutral → Neutral

Return ONLY one token:
Positive
Negative
Neutral
""".strip()
