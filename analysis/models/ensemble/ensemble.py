import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer
from analysis.evaluation_code.evaluator import evaluate_classification


# ── Config ────────────────────────────────────────────────────────────────────
# All bundles must follow the standard format:
#   {"model": <fitted classifier>, "scaler_dense": ..., "scaler_bert": ..., "threshold": float}
# The classifier must implement predict_proba(X) -> (n, 2) array.
# RidgeClassifier does NOT support predict_proba — use LogisticRegression(penalty="l2")
# or wrap it with sklearn.calibration.CalibratedClassifierCV before saving.

ARTIFACT_PATHS = {
    "logistic_regression": "analysis/models/all_outputs/lasso_log_reg/lasso_log_reg_tuned.pkl",
    "random_forest":       "analysis/models/all_outputs/random_forest/random_forest_tuned.pkl",
    "svm":                 "analysis/models/all_outputs/svm/svm_tuned.pkl",
    "ridge":               "analysis/models/all_outputs/ridge_log_reg/ridge_log_reg_tuned.pkl",
}

WEIGHTS    = None   # uniform; replace with {"logistic_regression": 2, "random_forest": 1, ...}
OUTPUT_DIR = "analysis/models/all_outputs/ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N       = 20
N_SHOW      = 20
SAMPLE_SIZE = 2000


# ── Ensemble class ────────────────────────────────────────────────────────────

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble over fitted sklearn-compatible classifiers.

    estimators : list of (name, fitted_model) tuples
    weights    : list of floats (same order as estimators), or None for uniform
    """

    def __init__(self, estimators, weights=None, decision_threshold=0.5):
        self.estimators = estimators
        self.weights = weights
        self.decision_threshold = decision_threshold
        self.classes_ = np.array([0, 1])

    def _resolved_weights(self):
        if self.weights is None:
            return [1.0] * len(self.estimators)
        if isinstance(self.weights, dict):
            return [self.weights.get(name, 1.0) for name, _ in self.estimators]
        return list(self.weights)

    def predict_proba(self, X):
        ws = self._resolved_weights()
        total_w = sum(ws)
        avg = np.zeros((X.shape[0], 2))
        for (_, model), w in zip(self.estimators, ws):
            avg += w * model.predict_proba(X)
        return avg / total_w

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(int)

    def score(self, X, y):
        return (self.predict(X) == np.asarray(y)).mean()

    @property
    def coef_(self):
        """Weighted average of linear coefficients / tree importances.
        Used by error_patterns_by_feature() to select top features."""
        ws = self._resolved_weights()
        coefs, used_ws = [], []
        for (_, model), w in zip(self.estimators, ws):
            if hasattr(model, "coef_"):
                coefs.append(w * np.asarray(model.coef_).ravel())
                used_ws.append(w)
            elif hasattr(model, "feature_importances_"):
                coefs.append(w * model.feature_importances_)
                used_ws.append(w)
        if not coefs:
            return None
        return np.sum(coefs, axis=0) / sum(used_ws)


# ── Bundle loading ────────────────────────────────────────────────────────────

def load_bundles():
    """Load all available model bundles.

    Returns
    -------
    estimators   : list of (name, model) tuples ready for EnsembleClassifier
    scaler_dense : MinMaxScaler fitted on dense features (from first bundle)
    scaler_bert  : MinMaxScaler fitted on BERT embeddings  (from first bundle)
    avg_threshold: mean of individual model thresholds (starting point for tuning)
    """
    estimators   = []
    scaler_dense = None
    scaler_bert  = None
    threshold_sum = 0.0

    for name, path in ARTIFACT_PATHS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: no artifact at {path}")
            continue

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        if not isinstance(bundle, dict) or "model" not in bundle:
            print(f"[skip] {name}: unexpected bundle format (expected dict with 'model' key)")
            continue

        if "scaler_dense" not in bundle or "scaler_bert" not in bundle:
            print(f"[skip] {name}: bundle missing scaler_dense/scaler_bert — "
                  "retrain this model on the full feature pipeline before adding to the ensemble")
            continue

        model = bundle["model"]
        if not (hasattr(model, "predict_proba") and hasattr(model, "predict")):
            print(f"[skip] {name}: model does not implement predict_proba")
            continue

        estimators.append((name, model))
        threshold_sum += bundle.get("threshold", 0.5)

        if scaler_dense is None:
            scaler_dense = bundle["scaler_dense"]
            scaler_bert  = bundle["scaler_bert"]

        print(f"[loaded] {name}  (threshold={bundle.get('threshold', 0.5):.4f})")

    if not estimators:
        raise RuntimeError(
            "No compatible model artifacts found. "
            "Train individual models first and save them with the standard bundle format."
        )

    avg_threshold = threshold_sum / len(estimators)
    return estimators, scaler_dense, scaler_bert, avg_threshold


# ── Feature reconstruction ────────────────────────────────────────────────────

def build_features(X_raw, scaler_dense, scaler_bert, split):
    """Reconstruct the full feature matrix (dense + TF-IDF + BERT) for a split."""
    dense_t = DenseFeatureTransformer()
    X_dense = scaler_dense.transform(dense_t.transform(X_raw))
    X_tfidf = sp.load_npz(f"data/processed/tfidf_{split}.npz")
    with open(f"data/processed/bert_{split}.pkl", "rb") as f:
        X_bert = scaler_bert.transform(pickle.load(f))
    return hstack([
        sp.csr_matrix(X_dense),
        X_tfidf,
        sp.csr_matrix(X_bert),
    ]).tocsr()


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(ensemble, X_val, y_val):
    """Choose the decision threshold that maximises F1 on the validation fold."""
    y_proba = ensemble.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


# ── Error analysis ────────────────────────────────────────────────────────────

def inspect_errors(X_val_raw, y_val, y_pred, y_proba):
    fp_mask = (y_pred == 1) & (y_val == 0)
    fn_mask = (y_pred == 0) & (y_val == 1)

    fp_df = (
        pd.DataFrame({"text": X_val_raw[fp_mask].values, "proba": y_proba[fp_mask]})
        .sort_values("proba", ascending=False)
        .head(N_SHOW)
    )
    fn_df = (
        pd.DataFrame({"text": X_val_raw[fn_mask].values, "proba": y_proba[fn_mask]})
        .sort_values("proba", ascending=True)
        .head(N_SHOW)
    )

    print(f"\n── Top {N_SHOW} False Positives (non-toxic flagged as toxic) ──")
    for _, row in fp_df.iterrows():
        print(f"  [{row['proba']:.3f}] {row['text']}")

    print(f"\n── Top {N_SHOW} False Negatives (toxic missed) ──")
    for _, row in fn_df.iterrows():
        print(f"  [{row['proba']:.3f}] {row['text']}")

    fp_df.to_csv(os.path.join(OUTPUT_DIR, "false_positives.csv"), index=False)
    fn_df.to_csv(os.path.join(OUTPUT_DIR, "false_negatives.csv"), index=False)
    print(f"Saved false_positives.csv and false_negatives.csv to {OUTPUT_DIR}/")


def error_patterns_by_feature(ensemble, X_val, y_val, y_pred):
    coef = ensemble.coef_
    if coef is None:
        print("[skip] error_patterns_by_feature: no coef_/feature_importances_ on any member")
        return

    rng     = np.random.default_rng(42)
    row_idx = rng.choice(X_val.shape[0], size=min(SAMPLE_SIZE, X_val.shape[0]), replace=False)
    top_idx = np.argsort(np.abs(coef))[::-1][:TOP_N]

    X_dense    = np.asarray(X_val[row_idx][:, top_idx].todense())
    y_pred_sub = y_pred[row_idx]
    y_val_sub  = y_val[row_idx]

    fp_mask      = (y_pred_sub == 1) & (y_val_sub == 0)
    fn_mask      = (y_pred_sub == 0) & (y_val_sub == 1)
    correct_mask = y_pred_sub == y_val_sub

    feature_names = [f"feature_{i}" for i in top_idx]
    means = {
        "correct":        X_dense[correct_mask].mean(axis=0),
        "false_positive": X_dense[fp_mask].mean(axis=0) if fp_mask.any() else np.zeros(TOP_N),
        "false_negative": X_dense[fn_mask].mean(axis=0) if fn_mask.any() else np.zeros(TOP_N),
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

    plt.suptitle("Ensemble — Error Patterns by Feature Value", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "error_patterns_by_feature.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    df.to_csv(os.path.join(OUTPUT_DIR, "error_patterns_by_feature.csv"))


def plot_confidence_distribution(y_val, y_pred, y_proba, threshold):
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load individual model bundles
    print("Loading model bundles...")
    estimators, scaler_dense, scaler_bert, avg_threshold = load_bundles()
    print(f"\n{len(estimators)} model(s) loaded: {[n for n, _ in estimators]}")

    # 2. Reconstruct feature matrices
    print("\nReconstructing feature matrices...")
    dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
    X_train_raw, X_test_raw, y_train_full, y_test = dp.get_data()
    y_train_full = y_train_full.values.ravel()
    y_test       = y_test.values.ravel()

    X_train_proc = build_features(X_train_raw, scaler_dense, scaler_bert, split="train")
    X_test_proc  = build_features(X_test_raw,  scaler_dense, scaler_bert, split="test")

    # 3. Build ensemble with initial threshold (will be tuned below)
    ensemble = EnsembleClassifier(estimators, weights=WEIGHTS, decision_threshold=avg_threshold)

    # 4. Tune threshold on last CV fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _, last_val_idx = list(cv.split(X_train_proc, y_train_full))[-1]

    X_val = X_train_proc[last_val_idx]
    y_val = y_train_full[last_val_idx]

    X_val_raw_df = X_train_raw.iloc[last_val_idx].reset_index(drop=True)
    X_val_raw    = X_val_raw_df.iloc[:, 0] if hasattr(X_val_raw_df, "columns") else X_val_raw_df

    threshold = tune_threshold(ensemble, X_val, y_val)
    ensemble.decision_threshold = threshold
    print(f"\nTuned threshold: {threshold:.4f}  (avg of individual thresholds: {avg_threshold:.4f})")

    # 5. Evaluate on held-out test set
    print("\n" + "="*60)
    y_pred_test  = ensemble.predict(X_test_proc)
    y_proba_test = ensemble.predict_proba(X_test_proc)[:, 1]

    evaluate_classification(
        y_test, y_pred_test, y_proba_test,
        name="Ensemble",
        plot_curves=True,
        save_path=os.path.join(OUTPUT_DIR, "ensemble_evaluation.png"),
    )

    # 6. Error analysis on validation fold
    y_proba_val = ensemble.predict_proba(X_val)[:, 1]
    y_pred_val  = ensemble.predict(X_val)

    print("\n── 1. FP / FN samples ──")
    inspect_errors(X_val_raw, y_val, y_pred_val, y_proba_val)

    print("\n── 2. Error patterns by feature ──")
    error_patterns_by_feature(ensemble, X_val, y_val, y_pred_val)

    print("\n── 3. Confidence distribution ──")
    plot_confidence_distribution(y_val, y_pred_val, y_proba_val, threshold)

    # 7. Save ensemble bundle (same format as individual models)
    bundle = {
        "model":        ensemble,
        "scaler_dense": scaler_dense,
        "scaler_bert":  scaler_bert,
        "threshold":    threshold,
    }
    out_path = os.path.join(OUTPUT_DIR, "ensemble.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nEnsemble bundle saved to {out_path}")
    print(f"All outputs saved to {OUTPUT_DIR}/")
