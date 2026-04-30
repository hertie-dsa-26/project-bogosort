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
# import shap
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer


MODEL_PATH   = "analysis/models/all_outputs/lasso_log_reg/lasso_log_reg_tuned.pkl"
OUTPUT_DIR   = "analysis/models/all_outputs/lasso_log_reg"
TOP_N        = 20
N_REPEATS    = 30
SHAP_N       = 1000
RANDOM_STATE = 42

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

X_val  = X_train_proc[last_val_idx]
y_val  = y_train_full[last_val_idx]

n_features    = X_val.shape[1]
feature_names = [f"feature_{i}" for i in range(n_features)]   # << replace with real names if available


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_auc(X):
    return roc_auc_score(y_val, model.predict_proba(X)[:, 1])

def predict_proba_dense(X):
    return model.predict_proba(sp.csr_matrix(X))


# ── 1. Coefficient importance ─────────────────────────────────────────────────

def run_coef_importance():
    df = pd.DataFrame({
        "feature":  feature_names,
        "coef":     model.coef_,
        "abs_coef": np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    top     = df.head(TOP_N).copy()
    colours = ["steelblue" if v >= 0 else "tomato" for v in top["coef"]]

    fig, ax = plt.subplots(figsize=(10, max(6, TOP_N * 0.35)))
    ax.barh(top["feature"][::-1], top["coef"][::-1], color=colours[::-1], alpha=0.85)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient value  (blue = positive, red = negative)")
    ax.set_title(f"Top {TOP_N} Features — Coefficient Importance")
    ax.grid(axis="x", alpha=0.3)
    n_zero = (df["coef"] == 0).sum()
    ax.text(0.98, 0.02, f"Zeroed by L1: {n_zero}/{len(df)}",
            transform=ax.transAxes, ha="right", fontsize=9, color="grey")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "coef_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    return df


# ── 2. Permutation importance ─────────────────────────────────────────────────

def run_permutation_importance():
    rng      = np.random.default_rng(RANDOM_STATE)
    base_auc = get_auc(X_val)
    n_feat   = X_val.shape[1]

    means, stds = [], []
    for j in range(n_feat):
        scores = []
        for _ in range(N_REPEATS):
            X_perm       = X_val.tolil()
            col          = np.asarray(X_perm[:, j].todense()).ravel()
            X_perm[:, j] = rng.permutation(col).reshape(-1, 1)
            scores.append(base_auc - get_auc(X_perm.tocsr()))
        means.append(np.mean(scores))
        stds.append(np.std(scores))

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": means,
        "std":        stds,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top = df.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, TOP_N * 0.35)))
    ax.barh(top["feature"], top["importance"], xerr=top["std"],
            color="steelblue", alpha=0.8, capsize=3)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean decrease in ROC-AUC when feature is shuffled")
    ax.set_title(f"Top {TOP_N} Features — Permutation Importance (last CV fold val set)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "perm_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}  |  Baseline ROC-AUC: {base_auc:.4f}")
    return df


# ── 3. SHAP ───────────────────────────────────────────────────────────────────

def run_shap():
    rng      = np.random.default_rng(RANDOM_STATE)
    idx      = rng.choice(X_val.shape[0], size=min(SHAP_N, X_val.shape[0]), replace=False)
    X_sample = np.asarray(X_val[idx].todense())

    background  = shap.kmeans(X_sample, 10)
    explainer   = shap.KernelExplainer(predict_proba_dense, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      max_display=TOP_N, show=False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    return df


# ── Run ───────────────────────────────────────────────────────────────────────

print("── 1. Coefficient importance ──")
coef_df = run_coef_importance()
print(coef_df.head(TOP_N).to_string(index=False))
coef_df.to_csv(os.path.join(OUTPUT_DIR, "coef_importance.csv"), index=False)

print("\n── 2. Permutation importance ──")
perm_df = run_permutation_importance()
print(perm_df.head(TOP_N).to_string(index=False))
perm_df.to_csv(os.path.join(OUTPUT_DIR, "perm_importance.csv"), index=False)

print(f"\n── 3. SHAP (KernelExplainer, {SHAP_N} samples) ──")
shap_df = run_shap()
print(shap_df.head(TOP_N).to_string(index=False))
shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
