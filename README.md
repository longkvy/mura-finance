# MURA-Finance
**Multi-Hop Reasoning with Augmented Context for Implicit Financial Sentiment Analysis**

MURA-Finance is a research project that adapts **multi-hop Chain-of-Thought (CoT) reasoning** to the financial domain, where sentiment is often **implicit, hedged, and context-dependent**.

The project extends the THOR (Three-hop Reasoning) framework to finance by introducing a **5-hop reasoning pipeline** and an optional **Retrieval-Augmented Generation (RAG)** layer to infer:
- implicit financial sentiment, and
- market implications (bullish / bearish / uncertain)
from short financial news headlines.

This repository contains the code, data processing scripts, and experiments for a **capstone research project**.

---

## Core Idea

Financial language rarely expresses sentiment explicitly.
Instead, meaning is conveyed through:
- hedging (“may”, “remains cautious”),
- euphemisms (“challenging environment”),
- indirect warnings (“headwinds persist”).

MURA-Finance models this behavior explicitly using structured reasoning steps rather than single-step classification.

---

## 5-Hop Reasoning Framework

1. **Entity Grounding**
   Identify the financial entity or ticker (e.g., EURUSD, BTC, AAPL).

2. **Financial Aspect Identification**
   Determine the key economic driver (e.g., inflation, rates, growth, risk).

3. **Implicit Cue Detection**
   Detect indirect linguistic signals such as hedging, euphemisms, or mixed framing.

4. **Implicit Sentiment Inference**
   Classify implicit sentiment: Positive / Negative / Neutral.

5. **Market Implication Inference**
   Translate sentiment into Bullish / Bearish / Uncertain signals.

An optional **RAG layer** augments short headlines with relevant contextual information.

---

## Datasets

- **Forex Financial News Headline Dataset**
  https://arxiv.org/abs/2308.07935

- **Financial PhraseBank**
  https://huggingface.co/datasets/takala/financial_phrasebank

- **SemEval-2017 Task 5 (FiQA)**
  https://github.com/tocab/SemEval2017Task5

- **Financial News Multisource (for RAG)**
  https://huggingface.co/datasets/Brianferrell787/financial-news-multisource

---

## Evaluation

We compare:
- FinBERT
- Single-shot LLM classification
- 5-hop reasoning (no RAG)
- 5-hop reasoning + RAG

Metrics:
- Accuracy
- Precision / Recall
- Macro F1-score

---

## Team

- Long Nguyen
- Quynh Nguyen
- Johnny

---

## References

- Fei et al., 2023 – Reasoning Implicit Sentiment with Chain-of-Thought Prompting
  https://arxiv.org/abs/2305.11255

- Fatouros et al., 2023 – Transforming Sentiment Analysis in the Financial Domain with ChatGPT
  https://arxiv.org/abs/2308.07935

- Kangtong et al., 2024 – Fine-Tuning Gemma-7B for Financial News Sentiment
  https://arxiv.org/abs/2406.13626
