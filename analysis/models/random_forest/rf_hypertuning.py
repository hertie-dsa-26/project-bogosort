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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer


OUTPUT_DIR = "analysis/models/all_outputs/random_forest"


def run():

    # ── Load data ─────────────────────────────────────────────────────────────

    dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
    X_train, X_test, y_train, y_test = dp.get_data()

    y_train = y_train.values.ravel()
    y_test  = y_test.values.ravel()

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    print(f"Toxic rate (train): {y_train.mean():.1%}")

    # ── Dense features only ───────────────────────────────────────────────────

    print("Generating dense features...")
    dense_transformer = DenseFeatureTransformer()
    X_train_dense = dense_transformer.transform(X_train)
    X_test_dense  = dense_transformer.transform(X_test)

    print(f"Feature matrix shape: {X_train_dense.shape}")

    # ── Grid search ───────────────────────────────────────────────────────────

    n_estimators_values      = [50, 100, 200]
    max_depth_values         = [5, 10, 20, None]
    min_samples_split_values = [2, 5, 10]

    # PR-AUC is used as the primary metric rather than ROC-AUC because the dataset
    # is heavily imbalanced (~9:1 non-toxic:toxic). ROC-AUC can be misleadingly
    # high in imbalanced settings as it accounts for true negatives, which are
    # abundant. PR-AUC focuses only on the positive (toxic) class and is a more
    # honest measure of how well the model identifies toxic comments.
    cv             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sample_weights = compute_sample_weight("balanced", y_train)

    records = []
    for n_est in n_estimators_values:
        for max_d in max_depth_values:
            for min_split in min_samples_split_values:
                fold_pr_aucs  = []
                fold_roc_aucs = []
                fold_f1s      = []

                for train_idx, val_idx in cv.split(X_train_dense, y_train):
                    X_fold_train = X_train_dense.iloc[train_idx]
                    X_fold_val   = X_train_dense.iloc[val_idx]
                    y_fold_train = y_train[train_idx]
                    y_fold_val   = y_train[val_idx]
                    w_fold       = sample_weights[train_idx]

                    rf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    )
                    rf.fit(X_fold_train, y_fold_train, sample_weight=w_fold)

                    y_proba = rf.predict_proba(X_fold_val)[:, 1]
                    y_pred  = rf.predict(X_fold_val)

                    fold_pr_aucs.append(average_precision_score(y_fold_val, y_proba))
                    fold_roc_aucs.append(roc_auc_score(y_fold_val, y_proba))
                    fold_f1s.append(f1_score(y_fold_val, y_pred, average="macro", zero_division=0))

                records.append({
                    "n_estimators":      n_est,
                    "max_depth":         max_d,
                    "min_samples_split": min_split,
                    "val_pr_auc_mean":   np.mean(fold_pr_aucs),
                    "val_pr_auc_std":    np.std(fold_pr_aucs),
                    "val_roc_auc_mean":  np.mean(fold_roc_aucs),
                    "val_macro_f1_mean": np.mean(fold_f1s),
                })
                print(f"n_est={n_est:<4} max_depth={str(max_d):<6} min_split={min_split:<4} PR-AUC: {np.mean(fold_pr_aucs):.4f} +/- {np.std(fold_pr_aucs):.4f}")

    results_df = pd.DataFrame(records).sort_values("val_pr_auc_mean", ascending=False)

    print()
    print("Best CV PR-AUC:", round(results_df.iloc[0]["val_pr_auc_mean"], 4))
    print("Best n_estimators:", results_df.iloc[0]["n_estimators"])
    print("Best max_depth:", results_df.iloc[0]["max_depth"])
    print("Best min_samples_split:", results_df.iloc[0]["min_samples_split"])
    print()
    print(results_df[["n_estimators", "max_depth", "min_samples_split",
                       "val_pr_auc_mean", "val_pr_auc_std", "val_roc_auc_mean"]].to_string(index=False))

    # ── Refit final model on full training set ────────────────────────────────

    best_n_est = int(results_df.iloc[0]["n_estimators"])
    best_depth = results_df.iloc[0]["max_depth"]
    best_depth = None if pd.isna(best_depth) else int(best_depth)
    best_split = int(results_df.iloc[0]["min_samples_split"])

    final_rf = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        min_samples_split=best_split,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_rf.fit(X_train_dense, y_train, sample_weight=sample_weights)

    # ── Threshold tuning ──────────────────────────────────────────────────────

    # threshold is tuned on the last CV fold's validation set rather than the
    # full training set, to avoid the model seeing its own training data when
    # picking the threshold.
    last_train_idx, last_val_idx = list(cv.split(X_train_dense, y_train))[-1]

    rf_threshold = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        min_samples_split=best_split,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_threshold.fit(
        X_train_dense.iloc[last_train_idx],
        y_train[last_train_idx],
        sample_weight=sample_weights[last_train_idx],
    )

    y_proba_val = rf_threshold.predict_proba(X_train_dense.iloc[last_val_idx])[:, 1]
    y_val       = y_train[last_val_idx]

    precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
    f1_scores      = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1_scores)])

    print(f"\nOptimal classification threshold: {best_threshold:.4f}  (default 0.5)")

    # ── Test set evaluation ───────────────────────────────────────────────────

    y_proba_test = final_rf.predict_proba(X_test_dense)[:, 1]
    y_pred_test  = (y_proba_test >= best_threshold).astype(int)

    pr_auc   = average_precision_score(y_test, y_proba_test)
    roc_auc  = roc_auc_score(y_test, y_proba_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print()
    print("Test set performance:")
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_tuning_results.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "random_forest_tuned.pkl"), "wb") as f:
        pickle.dump({
            "model":             final_rf,
            "dense_transformer": dense_transformer,
            "threshold":         best_threshold,
        }, f)

    print(f"\nSaved random_forest_tuning_results.csv and random_forest_tuned.pkl to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()