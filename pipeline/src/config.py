"""
config.py
---------
Centralized configuration for the FX Sentiment Analysis pipeline.
Adjust paths and model settings here before running the pipeline.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "google/flan-t5-xxl"
BATCH_SIZE = 8

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = "data/finance"
DEV_PATH = f"{DATA_DIR}/dev.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"

# Output CSVs
RESULT_SINGLEPROMPT_DEV  = f"{DATA_DIR}/result_dev_singleprompt.csv"
RESULT_MULTIHOP_DEV      = f"{DATA_DIR}/result_dev_multihop.csv"
RESULT_HYBRID_DEV        = f"{DATA_DIR}/result_dev_hybrid.csv"
RESULT_HYBRID_RAG_DEV    = f"{DATA_DIR}/result_dev_hybrid_RAG.csv"

RESULT_SINGLEPROMPT_TEST = f"{DATA_DIR}/result_test_singleprompt.csv"
RESULT_MULTIHOP_TEST     = f"{DATA_DIR}/result_test_multihop.csv"
RESULT_HYBRID_TEST       = f"{DATA_DIR}/result_test_hybrid.csv"
RESULT_HYBRID_RAG_TEST   = f"{DATA_DIR}/result_test_hybrid_RAG.csv"

# API model test results (test set only)
RESULT_GPT35_TEST         = f"{DATA_DIR}/result_test_gpt35.csv"
RESULT_GEMINI_FLASH_TEST  = f"{DATA_DIR}/result_test_gemini_flash.csv"

# ── Column names ──────────────────────────────────────────────────────────────
TEXT_COL        = "title"
TICKER_COL      = "ticker"
LABEL_COL       = "true_sentiment"
FINBERT_COL     = "finbert_sentiment"
IN_CONTEXT_COL  = "text"
EX_CONTEXT_COLS = ["external_context_1", "external_context_2", "external_context_3"]

LABELS = ["Positive", "Neutral", "Negative"]

# ── Inference settings ────────────────────────────────────────────────────────
MAX_NEW_TOKENS_DEFAULT = 64
MAX_NEW_TOKENS_SHORT   = 8
MAX_NEW_TOKENS_HOP1    = 40
