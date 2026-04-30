import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler

from analysis.models.lasso_log_reg.core_logistic_regression_lasso import LassoLogisticRegression
from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer


# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
OUTPUT_DIR    = "analysis/models/all_outputs/lasso_log_reg"


# ── Load data ─────────────────────────────────────────────────────────────────

dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
X_train, X_test, y_train, y_test = dp.get_data()

y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")
print(f"Toxic rate (train): {y_train.mean():.1%}")


# ── Load cached features and merge ────────────────────────────────────────────

print("Generating dense features...")
dense_transformer = DenseFeatureTransformer()
X_train_dense = dense_transformer.transform(X_train)
X_test_dense  = dense_transformer.transform(X_test)

scaler_dense = MinMaxScaler()
X_train_dense_scaled = scaler_dense.fit_transform(X_train_dense)
X_test_dense_scaled  = scaler_dense.transform(X_test_dense)

print("Loading cached TF-IDF...")
X_train_tfidf = sp.load_npz(os.path.join(PROCESSED_DIR, "tfidf_train.npz"))
X_test_tfidf  = sp.load_npz(os.path.join(PROCESSED_DIR, "tfidf_test.npz"))

print("Loading cached BERT embeddings...")
with open(os.path.join(PROCESSED_DIR, "bert_train.pkl"), "rb") as f:
    X_train_bert = pickle.load(f)
with open(os.path.join(PROCESSED_DIR, "bert_test.pkl"), "rb") as f:
    X_test_bert = pickle.load(f)

scaler_bert = MinMaxScaler()
X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert)
X_test_bert_scaled  = scaler_bert.transform(X_test_bert)

print("Merging features...")
X_train_proc = hstack([
    sp.csr_matrix(X_train_dense_scaled),
    X_train_tfidf,
    sp.csr_matrix(X_train_bert_scaled),
]).tocsr()

X_test_proc = hstack([
    sp.csr_matrix(X_test_dense_scaled),
    X_test_tfidf,
    sp.csr_matrix(X_test_bert_scaled),
]).tocsr()

print(f"Feature matrix shape: {X_train_proc.shape}")


# ── Grid search ───────────────────────────────────────────────────────────────

alpha_values         = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
learning_rate_values = [0.001, 0.01, 0.05, 0.1]

# PR-AUC is used as the primary metric rather than ROC-AUC because the dataset
# is heavily imbalanced (~9:1 non-toxic:toxic). ROC-AUC can be misleadingly
# high in imbalanced settings as it accounts for true negatives, which are
# abundant. PR-AUC focuses only on the positive (toxic) class and is a more
# honest measure of how well the model identifies toxic comments.
cv             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sample_weights = compute_sample_weight("balanced", y_train)

records = []
for alpha in alpha_values:
    for lr in learning_rate_values:
        fold_pr_aucs  = []
        fold_roc_aucs = []
        fold_f1s      = []

        for train_idx, val_idx in cv.split(X_train_proc, y_train):
            X_fold_train = X_train_proc[train_idx]
            X_fold_val   = X_train_proc[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val   = y_train[val_idx]
            w_fold       = sample_weights[train_idx]

            lasso = LassoLogisticRegression(
                alpha=alpha,
                learning_rate=lr,
                max_iter=1000,
            )
            lasso.fit(X_fold_train, y_fold_train, sample_weight=w_fold)

            y_proba = lasso.predict_proba(X_fold_val)[:, 1]
            y_pred  = lasso.predict(X_fold_val)

            fold_pr_aucs.append(average_precision_score(y_fold_val, y_proba))
            fold_roc_aucs.append(roc_auc_score(y_fold_val, y_proba))
            fold_f1s.append(f1_score(y_fold_val, y_pred, average="macro", zero_division=0))

        records.append({
            "alpha":             alpha,
            "learning_rate":     lr,
            "val_pr_auc_mean":   np.mean(fold_pr_aucs),
            "val_pr_auc_std":    np.std(fold_pr_aucs),
            "val_roc_auc_mean":  np.mean(fold_roc_aucs),
            "val_macro_f1_mean": np.mean(fold_f1s),
        })
        print(f"alpha={alpha:<6}  lr={lr:<6}  PR-AUC: {np.mean(fold_pr_aucs):.4f} +/- {np.std(fold_pr_aucs):.4f}")

results_df = pd.DataFrame(records).sort_values("val_pr_auc_mean", ascending=False)

print()
print("Best CV PR-AUC:", round(results_df.iloc[0]["val_pr_auc_mean"], 4))
print("Best alpha:", results_df.iloc[0]["alpha"])
print("Best learning_rate:", results_df.iloc[0]["learning_rate"])
print()
print(results_df[["alpha", "learning_rate", "val_pr_auc_mean", "val_pr_auc_std", "val_roc_auc_mean", "val_macro_f1_mean"]].to_string(index=False))


# ── Refit final model on full training set ────────────────────────────────────

best_alpha = results_df.iloc[0]["alpha"]
best_lr    = results_df.iloc[0]["learning_rate"]

final_lasso = LassoLogisticRegression(
    alpha=best_alpha,
    learning_rate=best_lr,
    max_iter=1000,
)
final_lasso.fit(X_train_proc, y_train, sample_weight=sample_weights)


# ── Threshold tuning ──────────────────────────────────────────────────────────

# threshold is tuned on the last CV fold's validation set rather than the
# full training set, to avoid the model seeing its own training data when
# picking the threshold.
last_train_idx, last_val_idx = list(cv.split(X_train_proc, y_train))[-1]

lasso_threshold = LassoLogisticRegression(
    alpha=best_alpha,
    learning_rate=best_lr,
    max_iter=1000,
)
lasso_threshold.fit(
    X_train_proc[last_train_idx],
    y_train[last_train_idx],
    sample_weight=sample_weights[last_train_idx],
)

y_proba_val = lasso_threshold.predict_proba(X_train_proc[last_val_idx])[:, 1]
y_val       = y_train[last_val_idx]

precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
f1_scores      = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
best_threshold = float(thresholds[np.argmax(f1_scores)])

print(f"\nOptimal classification threshold: {best_threshold:.4f}  (default 0.5)")


# ── Test set evaluation ───────────────────────────────────────────────────────

final_lasso.decision_threshold = best_threshold

y_proba_test = final_lasso.predict_proba(X_test_proc)[:, 1]
y_pred_test  = final_lasso.predict(X_test_proc)

pr_auc   = average_precision_score(y_test, y_proba_test)
roc_auc  = roc_auc_score(y_test, y_proba_test)
macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

print()
print("Test set performance:")
print(f"  PR-AUC:   {pr_auc:.4f}")
print(f"  ROC-AUC:  {roc_auc:.4f}")
print(f"  Macro-F1: {macro_f1:.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

results_df.to_csv(os.path.join(OUTPUT_DIR, "lasso_tuning_results.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "lasso_log_reg_tuned.pkl"), "wb") as f:
    pickle.dump({
        "model":        final_lasso,
        "scaler_dense": scaler_dense,
        "scaler_bert":  scaler_bert,
        "threshold":    best_threshold,
    }, f)

print(f"\nSaved lasso_tuning_results.csv and lasso_log_reg_tuned.pkl to {OUTPUT_DIR}/")