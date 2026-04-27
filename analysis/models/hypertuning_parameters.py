import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis.models.data_pipeline import DataPipeline
from analysis.features.build_features import FeatureBuilder, FeaturePreprocessor


# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR      = "./data/processed"
ARTIFACTS_DIR      = "./analysis/models/artifacts"
X_TRAIN_PROC_PATH  = os.path.join(PROCESSED_DIR, "X_train_proc.npz")
X_TEST_PROC_PATH   = os.path.join(PROCESSED_DIR, "X_test_proc.npz")
PREPROCESSOR_PATH  = os.path.join(PROCESSED_DIR, "preprocessor.pkl")
Y_TRAIN_PATH       = os.path.join(PROCESSED_DIR, "y_train.npy")
Y_TEST_PATH        = os.path.join(PROCESSED_DIR, "y_test.npy")


# ── Load data ─────────────────────────────────────────────────────────────────

dp = DataPipeline("./data/processed/test_train_data.pkl", label_columns=["toxic"])
X_train, X_test, y_train, y_test = dp.get_data()

y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")
print(f"Toxic rate (train): {y_train.mean():.1%}")


# ── Features + preprocessing (cached) ────────────────────────────────────────

if (os.path.exists(X_TRAIN_PROC_PATH) and
    os.path.exists(X_TEST_PROC_PATH)  and
    os.path.exists(PREPROCESSOR_PATH)):

    print("Loading cached processed matrices...")
    X_train_proc = sp.load_npz(X_TRAIN_PROC_PATH).tocsr()
    X_test_proc  = sp.load_npz(X_TEST_PROC_PATH).tocsr()
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

else:
    fb = FeatureBuilder()

    if os.path.exists(fb.tfidf_path):
        fb.load()
    else:
        print("Fitting TF-IDF vectorizers...")
        fb.fit(X_train)

    print("Transforming train features...")
    X_train_feat = fb.transform(X_train, split="train")
    print("Transforming test features...")
    X_test_feat  = fb.transform(X_test,  split="test")

    preprocessor = FeaturePreprocessor()
    print("Preprocessing features...")
    X_train_proc = preprocessor.fit_transform(X_train_feat).tocsr()
    X_test_proc  = preprocessor.transform(X_test_feat).tocsr()

    # cache to disk so future runs skip this step
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    sp.save_npz(X_TRAIN_PROC_PATH, X_train_proc)
    sp.save_npz(X_TEST_PROC_PATH,  X_test_proc)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    print("Processed matrices cached to disk.")


# ── Grid search ───────────────────────────────────────────────────────────────

# C is the inverse of regularisation strength — smaller C = more regularisation.
# We tune over a log-scale range to find the right sparsity for the L1 penalty.
C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

# PR-AUC is used as the primary metric rather than ROC-AUC because the dataset
# is heavily imbalanced (~9:1 non-toxic:toxic). ROC-AUC can be misleadingly
# high in imbalanced settings as it accounts for true negatives, which are
# abundant. PR-AUC focuses only on the positive (toxic) class and is a more
# honest measure of how well the model identifies toxic comments.
cv             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sample_weights = compute_sample_weight("balanced", y_train)

records = []
for C in C_values:
    fold_pr_aucs  = []
    fold_roc_aucs = []
    fold_f1s      = []

    for train_idx, val_idx in cv.split(X_train_proc, y_train):
        X_fold_train = X_train_proc[train_idx]
        X_fold_val   = X_train_proc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val   = y_train[val_idx]
        w_fold       = sample_weights[train_idx]

        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_fold_train, y_fold_train, sample_weight=w_fold)

        y_proba = model.predict_proba(X_fold_val)[:, 1]
        y_pred  = model.predict(X_fold_val)

        fold_pr_aucs.append(average_precision_score(y_fold_val, y_proba))
        fold_roc_aucs.append(roc_auc_score(y_fold_val, y_proba))
        fold_f1s.append(f1_score(y_fold_val, y_pred, average="macro", zero_division=0))

    records.append({
        "C":                 C,
        "val_pr_auc_mean":   np.mean(fold_pr_aucs),
        "val_pr_auc_std":    np.std(fold_pr_aucs),
        "val_roc_auc_mean":  np.mean(fold_roc_aucs),
        "val_macro_f1_mean": np.mean(fold_f1s),
    })
    print(f"C={C:<6}  PR-AUC: {np.mean(fold_pr_aucs):.4f} +/- {np.std(fold_pr_aucs):.4f}")

results_df = pd.DataFrame(records).sort_values("val_pr_auc_mean", ascending=False)

print()
print("Best CV PR-AUC:", round(results_df.iloc[0]["val_pr_auc_mean"], 4))
print("Best C:", results_df.iloc[0]["C"])
print()
print(results_df[["C", "val_pr_auc_mean", "val_pr_auc_std", "val_roc_auc_mean", "val_macro_f1_mean"]].to_string(index=False))


# ── Threshold tuning ──────────────────────────────────────────────────────────

# The default threshold of 0.5 assumes balanced classes. With a 9:1 imbalance
# the optimal decision boundary is lower — we sweep the precision-recall curve
# and pick the threshold that maximises F1.
best_C = results_df.iloc[0]["C"]

final_model = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=best_C,
    max_iter=1000,
    random_state=42,
)
final_model.fit(X_train_proc, y_train, sample_weight=sample_weights)

y_proba_train  = final_model.predict_proba(X_train_proc)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_train, y_proba_train)
f1_scores      = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
best_threshold = float(thresholds[np.argmax(f1_scores)])

print(f"\nOptimal classification threshold: {best_threshold:.4f}  (default 0.5)")


# ── Test set evaluation ───────────────────────────────────────────────────────

y_proba_test = final_model.predict_proba(X_test_proc)[:, 1]
y_pred_test  = (y_proba_test >= best_threshold).astype(int)

print()
print("Test set performance:")
print(f"  PR-AUC:   {average_precision_score(y_test, y_proba_test):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"  Macro-F1: {f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

results_df.to_csv(os.path.join(ARTIFACTS_DIR, "tuning_results.csv"), index=False)

with open(os.path.join(ARTIFACTS_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump({
        "model":        final_model,
        "preprocessor": preprocessor,
        "threshold":    best_threshold,
    }, f)

print("\nSaved tuning_results.csv and best_model.pkl to artifacts/")