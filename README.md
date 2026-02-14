# MURA-Finance
**Multi-Hop Reasoning with Augmented Context for Implicit Financial Sentiment Analysis**

MURA-Finance is a research project that adapts **multi-hop Chain-of-Thought (CoT) reasoning** to the financial domain, where sentiment is often **implicit, hedged, and context-dependent**.

The primary objective of this project is to introduce a multi-hop domain-specific pipeline designed to navigate the linguistic complexities of financial markets. In addition, we incorporate Retrieval-Augmented Generation (RAG) to supply relevant contextual information, enabling more informed reasoning and translating inferred sentiment into actionable market signals.

This repository contains the code, data processing scripts, and experiments for a **capstone research project**.

---

## Core Pipeline

<img width="1519" height="801" alt="Screenshot 2026-02-14 at 17 48 32" src="https://github.com/user-attachments/assets/2367b453-2191-41f6-9500-6b50c0273a9b" />


---

## 5-Hop Reasoning Framework

<img width="643" height="738" alt="Screenshot 2026-02-14 at 17 50 05" src="https://github.com/user-attachments/assets/21016496-0282-4761-abc2-cb43e2838dbe" />


An optional **RAG layer** augments short headlines with relevant contextual information.

---

## Datasets

- **Forex Financial News Headline Dataset**
  https://arxiv.org/abs/2308.07935

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
