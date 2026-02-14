# MURA-Finance
**Multi-Hop Reasoning with Augmented Context for Implicit Financial Sentiment Analysis**

MURA-Finance is a research project that adapts **multi-hop Chain-of-Thought (CoT) reasoning** to the financial domain, where sentiment is often **implicit, hedged, and context-dependent**.

The primary objective of this project is to introduce a multi-hop domain-specific pipeline designed to navigate the linguistic complexities of financial markets. In addition, we incorporate Retrieval-Augmented Generation (RAG) to supply relevant contextual information, enabling more informed reasoning and translating inferred sentiment into actionable market signals.

This repository contains the code, data processing scripts, and experiments for a **capstone research project**.

---

## Core Pipeline

<img width="1519" height="801" alt="Screenshot 2026-02-14 at 17 48 32" src="https://github.com/user-attachments/assets/2367b453-2191-41f6-9500-6b50c0273a9b" />


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

- [1] Y. Hao, J. Wang, W. Hong, and D. Zhang, "Reasoning Implicit Sentiment with Chain-of-Thought Prompting," arXiv preprint arXiv:2305.11255, 2023. [Online]. Available: https://arxiv.org/abs/2305.11255
- [2] G. Fatouros et al., "Transforming Sentiment Analysis in the Financial Domain with ChatGPT," arXiv preprint arXiv:2308.07935, 2023. [Online]. Available: https://arxiv.org/abs/2308.07935
- [3] B. Ferrell, "Financial News Multisource Dataset," HuggingFace Datasets, 2024. [Online]. Available: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource
