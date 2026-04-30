import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis.models.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer


def run():

    # ── Load tuned model ──────────────────────────────────────────────────────

    with open("./analysis/models/artifacts/random_forest_tuned.pkl", "rb") as f:
        bundle = pickle.load(f)

    tuned_model       = bundle["model"]
    dense_transformer = bundle["dense_transformer"]
    tuned_threshold   = bundle["threshold"]

    # ── Load data ─────────────────────────────────────────────────────────────

    dp = DataPipeline("./data/processed/test_train_data.pkl", label_columns=["toxic"])
    X_train, X_test, y_train, y_test = dp.get_data()

    y_train = y_train.values.ravel()
    y_test  = y_test.values.ravel()

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    # ── Dense features ────────────────────────────────────────────────────────

    print("Generating dense features...")
    X_train_dense = dense_transformer.transform(X_train)
    X_test_dense  = dense_transformer.transform(X_test)

    feat_names  = X_train_dense.columns.tolist()
    importances = tuned_model.feature_importances_

    # ── Feature selection ─────────────────────────────────────────────────────

    ranked_idx = np.argsort(importances)[::-1]

    print("\nAll features ranked by importance:")
    print(f"{'#':<4} {'feature':<35} {'importance':>10}")
    print(f"{'-'*4} {'-'*35} {'-'*10}")
    for rank, idx in enumerate(ranked_idx, 1):
        print(f"{rank:<4} {feat_names[idx]:<35} {importances[idx]:>10.4f}")

    top_5_features  = [feat_names[i] for i in ranked_idx[:5]]
    top_10_features = [feat_names[i] for i in ranked_idx[:10]]

    print(f"\nTop 5 features:  {top_5_features}")
    print(f"Top 10 features: {top_10_features}")

    # ── Retrain + evaluate on each subset ─────────────────────────────────────

    sample_weights = compute_sample_weight("balanced", y_train)
    cv             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for label, selected_features in [("top_5", top_5_features), ("top_10", top_10_features)]:
        print(f"\n── Training on {label} features ─────────────────────────────────")

        X_train_sel = X_train_dense[selected_features]
        X_test_sel  = X_test_dense[selected_features]

        model = RandomForestClassifier(
            n_estimators=tuned_model.n_estimators,
            max_depth=tuned_model.max_depth,
            min_samples_split=tuned_model.min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_sel, y_train, sample_weight=sample_weights)

        # threshold tuning on last fold
        last_train_idx, last_val_idx = list(cv.split(X_train_sel, y_train))[-1]

        rf_thresh = RandomForestClassifier(
            n_estimators=tuned_model.n_estimators,
            max_depth=tuned_model.max_depth,
            min_samples_split=tuned_model.min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_thresh.fit(
            X_train_sel.iloc[last_train_idx],
            y_train[last_train_idx],
            sample_weight=sample_weights[last_train_idx],
        )

        y_proba_val = rf_thresh.predict_proba(X_train_sel.iloc[last_val_idx])[:, 1]
        y_val       = y_train[last_val_idx]

        precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        threshold = float(thresholds[np.argmax(f1_scores)])

        y_proba_test = model.predict_proba(X_test_sel)[:, 1]
        y_pred_test  = (y_proba_test >= threshold).astype(int)

        pr_auc   = average_precision_score(y_test, y_proba_test)
        roc_auc  = roc_auc_score(y_test, y_proba_test)
        macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

        print(f"  PR-AUC:   {pr_auc:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Threshold: {threshold:.4f}")

        # save predictions
        os.makedirs(f"./analysis/models/model_outputs/random_forest/predictions", exist_ok=True)
        pd.DataFrame({"true": y_test, "pred": y_pred_test}).to_csv(
            f"./analysis/models/model_outputs/random_forest/predictions/rf_{label}_predictions.csv",
            index=False,
        )

        results[label] = {
            "pr_auc":    pr_auc,
            "roc_auc":   roc_auc,
            "macro_f1":  macro_f1,
            "threshold": threshold,
            "features":  selected_features,
            "model":     model,
        }

    # ── Comparison ────────────────────────────────────────────────────────────

    y_proba_full = tuned_model.predict_proba(X_test_dense)[:, 1]
    y_pred_full  = (y_proba_full >= tuned_threshold).astype(int)

    full_pr_auc  = average_precision_score(y_test, y_proba_full)
    full_roc_auc = roc_auc_score(y_test, y_proba_full)
    full_f1      = f1_score(y_test, y_pred_full, average="macro", zero_division=0)

    print("\n── Performance comparison ───────────────────────────────────────────")
    print(f"{'model':<25} {'PR-AUC':>8} {'ROC-AUC':>9} {'Macro-F1':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*9} {'-'*10}")
    print(f"{'all features (32)':<25} {full_pr_auc:>8.4f} {full_roc_auc:>9.4f} {full_f1:>10.4f}")
    for label, res in results.items():
        n = len(res["features"])
        print(f"{label+' ('+str(n)+')':<25} {res['pr_auc']:>8.4f} {res['roc_auc']:>9.4f} {res['macro_f1']:>10.4f}")

    # ── Save best feature-selected model ──────────────────────────────────────

    best_label = max(results, key=lambda k: results[k]["pr_auc"])
    best       = results[best_label]

    os.makedirs("./analysis/models/artifacts", exist_ok=True)

    with open("./analysis/models/artifacts/random_forest_selected.pkl", "wb") as f:
        pickle.dump({
            "model":             best["model"],
            "dense_transformer": dense_transformer,
            "threshold":         best["threshold"],
            "selected_features": best["features"],
        }, f)

    print(f"\nBest feature-selected model: {best_label} ({len(best['features'])} features)")
    print("Saved random_forest_selected.pkl to artifacts/")


if __name__ == "__main__":
    run()