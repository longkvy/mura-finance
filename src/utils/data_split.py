import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ======================
# CONFIG
# ======================
INPUT_FILE = "../../sentiment_annotated_with_texts_add_context.csv"       
OUTPUT_DIR = "../../data"
DEV_FILE = "dev.csv"
TEST_FILE = "test.csv"

TEST_RATIO = 0.2
RANDOM_STATE = 42

DATE_COL = "published_at"
LABEL_COL = "true_sentiment"

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(INPUT_FILE)

# Parse date
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Ensure we only use 2023 data (optional but recommended)
df = df[df[DATE_COL].dt.year == 2023].reset_index(drop=True)

# ======================
# CREATE TIME BUCKETS
# ======================
# Monthly buckets to preserve temporal diversity
df["month"] = df[DATE_COL].dt.to_period("M")

# ======================
# STRATIFIED TIME-AWARE SPLIT
# ======================
dev_parts = []
test_parts = []

for (month, sentiment), group in df.groupby(["month", LABEL_COL]):
    if len(group) < 5:
        # Too small to split safely → keep in dev
        dev_parts.append(group)
        continue

    dev, test = train_test_split(
        group,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    dev_parts.append(dev)
    test_parts.append(test)

dev_df = pd.concat(dev_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

# Drop helper column
dev_df = dev_df.drop(columns=["month"])
test_df = test_df.drop(columns=["month"])

# ======================
# SANITY CHECKS
# ======================
print("==== DEV SET ====")
print(dev_df[LABEL_COL].value_counts())
print("\n==== TEST SET ====")
print(test_df[LABEL_COL].value_counts())

print("\nDate ranges:")
print("DEV :", dev_df[DATE_COL].min(), "→", dev_df[DATE_COL].max())
print("TEST:", test_df[DATE_COL].min(), "→", test_df[DATE_COL].max())

# ======================
# SAVE FILES
# ======================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

dev_path = Path(OUTPUT_DIR) / DEV_FILE
test_path = Path(OUTPUT_DIR) / TEST_FILE

dev_df.to_csv(dev_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nSaved files:")
print(f" - {dev_path}")
print(f" - {test_path}")