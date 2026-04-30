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
from sklearn.metrics import average_precision_score, f1_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer
from analysis.evaluation_code.evaluator import evaluate_classification


OUTPUT_DIR = "analysis/models/all_outputs/random_forest"


def run_baseline():
    """Trains and evaluates a baseline Random Forest on dense features."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

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

    print(f"Dense feature matrix shape: {X_train_dense.shape}")

    # ── Train model ───────────────────────────────────────────────────────────

    sample_weights = compute_sample_weight("balanced", y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("Fitting Random Forest...")
    model.fit(X_train_dense, y_train, sample_weight=sample_weights)

    # ── Evaluation ────────────────────────────────────────────────────────────

    y_pred  = model.predict(X_test_dense)
    y_proba = model.predict_proba(X_test_dense)[:, 1]

    pr_auc   = average_precision_score(y_test, y_proba)
    roc_auc  = roc_auc_score(y_test, y_proba)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print()
    print("Test set performance:")
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["non-toxic", "toxic"], zero_division=0))

    evaluate_classification(
        y_test, y_pred, y_proba,
        name="random_forest_baseline",
        plot_curves=True,
        save_path=os.path.join(OUTPUT_DIR, "random_forest_baseline_evaluation.png"),
    )

    # ── Feature importance ────────────────────────────────────────────────────

    feat_names  = X_train_dense.columns.tolist()
    importances = model.feature_importances_
    top10_idx   = np.argsort(importances)[::-1][:10]

    print("Top 10 features by importance:")
    print(f"{'#':<4} {'feature':<35} {'importance':>10}")
    print(f"{'-'*4} {'-'*35} {'-'*10}")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"{rank:<4} {feat_names[idx]:<35} {importances[idx]:>10.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────

    pd.DataFrame({"true": y_test, "pred": y_pred}).to_csv(
        os.path.join(OUTPUT_DIR, "predictions", "rf_baseline_predictions.csv"),
        index=False,
    )

    with open(os.path.join(OUTPUT_DIR, "random_forest_baseline.pkl"), "wb") as f:
        pickle.dump({
            "model":             model,
            "dense_transformer": dense_transformer,
        }, f)

    print(f"\nSaved rf_baseline_predictions.csv and random_forest_baseline.pkl to {OUTPUT_DIR}/")


def run_tuned():
    """Loads the tuned Random Forest and runs full evaluation."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load tuned model
    with open(os.path.join(OUTPUT_DIR, "random_forest_tuned.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model             = bundle["model"]
    dense_transformer = bundle["dense_transformer"]
    threshold         = bundle["threshold"]

    # load data
    dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
    _, X_test_raw, _, y_test = dp.get_data()
    y_test = y_test.values.ravel()

    # features
    X_test_dense = dense_transformer.transform(X_test_raw)

    # predict
    y_proba = model.predict_proba(X_test_dense)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # evaluate
    evaluate_classification(
        y_test, y_pred, y_proba,
        name="random_forest_tuned",
        plot_curves=True,
        save_path=os.path.join(OUTPUT_DIR, "random_forest_tuned_evaluation.png"),
    )

    print(f"\nDecision threshold used: {threshold:.4f}")


if __name__ == "__main__":
    # pass "tuned" as argument to run tuned eval, otherwise runs baseline
    if len(sys.argv) > 1 and sys.argv[1] == "tuned":
        run_tuned()
    else:
        run_baseline()
