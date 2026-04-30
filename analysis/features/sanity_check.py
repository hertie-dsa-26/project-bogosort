"""
Quick sanity check for build_features.py.
Run with: uv run python analysis/features/sanity_check.py
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from analysis.features.build_features import DenseFeatureTransformer, TfidfTransformer

DATA_PATH = "data/raw/jigsaw-dataset/train.csv"

# ---------------------------------------------------------------------------
# Load a small sample
# ---------------------------------------------------------------------------
df     = pd.read_csv(DATA_PATH)
sample = df.sample(1000, random_state=42)
train, test = train_test_split(sample, test_size=0.2, random_state=42)

print(f"Train rows: {len(train)}  |  Test rows: {len(test)}")

# ---------------------------------------------------------------------------
# DenseFeatureTransformer
# ---------------------------------------------------------------------------
print("\n--- DenseFeatureTransformer ---")
t0      = time.time()
dense   = DenseFeatureTransformer()
result  = dense.fit_transform(train)
elapsed = time.time() - t0

print(f"Output shape : {result.shape}")
print(f"Time (800 rows): {elapsed:.1f}s  →  ~{elapsed * 75:.0f}s estimated for 60K rows")
print(f"New columns  : {[c for c in result.columns if c not in train.columns]}")

# Spot-check a known comment
test_cases = [
    ("YOU are TOTALLY WORTHLESS you idiot!!!", "should score high on second_person, uppercase, profanity"),
    ("Thanks for the edit, looks great.",       "should score near zero on everything"),
    ("kys you f**king retard stfu",             "should score high on slang and profanity"),
]
print("\n--- Spot checks ---")
for text, description in test_cases:
    row    = pd.DataFrame({"comment_text": [text]})
    output = dense.transform(row).iloc[0]
    print(f"\n'{text[:60]}'")
    print(f"  ({description})")
    print(f"  vader_compound       : {output['vader_compound']:.3f}")
    print(f"  second_person_count  : {output['second_person_count']}")
    print(f"  uppercase_ratio      : {output['uppercase_ratio']:.2f}")
    print(f"  profanity_count      : {output['profanity_count']}")
    print(f"  slang_count          : {output['slang_count']}")

# ---------------------------------------------------------------------------
# TfidfTransformer
# ---------------------------------------------------------------------------
print("\n--- TfidfTransformer ---")
tfidf   = TfidfTransformer()
X_train = tfidf.fit_transform(train)
X_test  = tfidf.transform(test)        # transform only — no refit

print(f"Train matrix shape : {X_train.shape}")
print(f"Test matrix shape  : {X_test.shape}")
print(f"Same n_cols        : {X_train.shape[1] == X_test.shape[1]}  (must be True)")
