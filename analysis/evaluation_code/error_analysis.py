import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer


MODEL_PATH  = "analysis/models/all_outputs/lasso_log_reg/lasso_log_reg_tuned.pkl"
OUTPUT_DIR  = "analysis/models/all_outputs/lasso_log_reg"
TOP_N       = 20
N_SHOW      = 20
SAMPLE_SIZE = 2000   # rows used for error patterns — keeps it fast

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────────

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model        = bundle["model"]
scaler_dense = bundle["scaler_dense"]
scaler_bert  = bundle["scaler_bert"]
threshold    = bundle["threshold"]
model.decision_threshold = threshold


# ── Load data + reconstruct features ─────────────────────────────────────────

dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
X_train_raw, _, y_train_full, _ = dp.get_data()
y_train_full = y_train_full.values.ravel()

dense_transformer = DenseFeatureTransformer()
X_train_dense     = scaler_dense.transform(dense_transformer.transform(X_train_raw))
X_train_tfidf     = sp.load_npz("data/processed/tfidf_train.npz")
with open("data/processed/bert_train.pkl", "rb") as f:
    X_train_bert = scaler_bert.transform(pickle.load(f))

X_train_proc = hstack([
    sp.csr_matrix(X_train_dense),
    X_train_tfidf,
    sp.csr_matrix(X_train_bert),
]).tocsr()


# ── Reconstruct last CV fold val split ────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_, last_val_idx = list(cv.split(X_train_proc, y_train_full))[-1]

X_val     = X_train_proc[last_val_idx]
y_val     = y_train_full[last_val_idx]
X_val_raw = X_train_raw.iloc[last_val_idx].reset_index(drop=True)

if hasattr(X_val_raw, "columns"):
    X_val_raw = X_val_raw.iloc[:, 0]

y_proba = model.predict_proba(X_val)[:, 1]
y_pred  = model.predict(X_val)


# ── 1. FP / FN samples ───────────────────────────────────────────────────────

def inspect_errors():
    fp_mask = (y_pred == 1) & (y_val == 0)
    fn_mask = (y_pred == 0) & (y_val == 1)

    fp_df = pd.DataFrame({
        "text":  X_val_raw[fp_mask].values,
        "proba": y_proba[fp_mask],
    }).sort_values("proba", ascending=False).head(N_SHOW)

    fn_df = pd.DataFrame({
        "text":  X_val_raw[fn_mask].values,
        "proba": y_proba[fn_mask],
    }).sort_values("proba", ascending=True).head(N_SHOW)

    print(f"\n── Top {N_SHOW} False Positives (non-toxic flagged as toxic, highest confidence) ──")
    for _, row in fp_df.iterrows():
        print(f"  [{row['proba']:.3f}] {row['text']}")

    print(f"\n── Top {N_SHOW} False Negatives (toxic missed, lowest confidence) ──")
    for _, row in fn_df.iterrows():
        print(f"  [{row['proba']:.3f}] {row['text']}")

    fp_df.to_csv(os.path.join(OUTPUT_DIR, "false_positives.csv"), index=False)
    fn_df.to_csv(os.path.join(OUTPUT_DIR, "false_negatives.csv"), index=False)
    print(f"\nSaved false_positives.csv and false_negatives.csv to {OUTPUT_DIR}/")


# ── 2. Error patterns by feature value ───────────────────────────────────────

def error_patterns_by_feature():
    rng        = np.random.default_rng(42)
    row_idx    = rng.choice(X_val.shape[0], size=min(SAMPLE_SIZE, X_val.shape[0]), replace=False)
    top_idx    = np.argsort(np.abs(model.coef_))[::-1][:TOP_N]

    X_dense      = np.asarray(X_val[row_idx][:, top_idx].todense())
    y_pred_sub   = y_pred[row_idx]
    y_val_sub    = y_val[row_idx]

    fp_mask      = (y_pred_sub == 1) & (y_val_sub == 0)
    fn_mask      = (y_pred_sub == 0) & (y_val_sub == 1)
    correct_mask = y_pred_sub == y_val_sub

    feature_names = [f"feature_{i}" for i in top_idx]

    means = {
        "correct":        X_dense[correct_mask].mean(axis=0),
        "false_positive": X_dense[fp_mask].mean(axis=0),
        "false_negative": X_dense[fn_mask].mean(axis=0),
    }
    df = pd.DataFrame(means, index=feature_names)
    df["fp_vs_correct"] = df["false_positive"] - df["correct"]
    df["fn_vs_correct"] = df["false_negative"] - df["correct"]

    print("\n── Error Patterns by Feature ──")
    print(df[["correct", "false_positive", "false_negative",
              "fp_vs_correct", "fn_vs_correct"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, TOP_N * 0.35)))
    for ax, col, title, colour in [
        (axes[0], "fp_vs_correct", "False Positive − Correct\n(features over-triggering)", "tomato"),
        (axes[1], "fn_vs_correct", "False Negative − Correct\n(features under-triggering)", "steelblue"),
    ]:
        vals = df[col].sort_values()
        ax.barh(vals.index, vals.values, color=colour, alpha=0.8)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Mean feature value difference")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Error Patterns by Feature Value", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "error_patterns_by_feature.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    df.to_csv(os.path.join(OUTPUT_DIR, "error_patterns_by_feature.csv"))


# ── 3. Confidence distribution ────────────────────────────────────────────────

def plot_confidence_distribution():
    fp_mask = (y_pred == 1) & (y_val == 0)
    fn_mask = (y_pred == 0) & (y_val == 1)
    tp_mask = (y_pred == 1) & (y_val == 1)
    tn_mask = (y_pred == 0) & (y_val == 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 30)

    axes[0].hist(y_proba[fp_mask], bins=bins, color="tomato",    alpha=0.7, label=f"FP (n={fp_mask.sum()})")
    axes[0].hist(y_proba[tn_mask], bins=bins, color="steelblue", alpha=0.7, label=f"TN (n={tn_mask.sum()})")
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.2f}")
    axes[0].set_title("Non-toxic samples — confidence distribution")
    axes[0].set_xlabel("Predicted probability (toxic)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(y_proba[fn_mask], bins=bins, color="tomato",    alpha=0.7, label=f"FN (n={fn_mask.sum()})")
    axes[1].hist(y_proba[tp_mask], bins=bins, color="steelblue", alpha=0.7, label=f"TP (n={tp_mask.sum()})")
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.2f}")
    axes[1].set_title("Toxic samples — confidence distribution")
    axes[1].set_xlabel("Predicted probability (toxic)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "error_confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ── Run ───────────────────────────────────────────────────────────────────────

print("── 1. FP / FN samples ──")
inspect_errors()

print("\n── 2. Error patterns by feature ──")
error_patterns_by_feature()

print("\n── 3. Confidence distribution ──")
plot_confidence_distribution()

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
