import os
import sys
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    auc,
    confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt


def evaluate_classification(y_true, y_pred, y_score=None, name="Model", plot_curves=True,
    save_path=None):
    print(f"\n===== {name} Evaluation =====")

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\nClassification Report:\n")
    clf_report = classification_report(y_true, y_pred)
    print(clf_report)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"  TP: {tp}  FP: {fp}")
    print(f"  FN: {fn}  TN: {tn}")

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": clf_report,
        "confusion_matrix": cm,
    }

    # ----- ROC / PR / AUC -----
    if y_score is not None:
        y_score = np.asarray(y_score).ravel()

        # ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        # PR
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        pr_auc_trapz = auc(pr_recall, pr_precision)         # area under PR curve via trapezoid
        ap_score = average_precision_score(y_true, y_score) # Average Precision (recommended PR summary)

        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc_trapz:.4f}")
        print(f"Avg Prec.: {ap_score:.4f}")

        metrics.update({
            "roc_auc": roc_auc,
            "pr_auc": pr_auc_trapz,
            "average_precision": ap_score,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
            "pr_curve": {
                "precision": pr_precision.tolist(),
                "recall": pr_recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
        })

        if plot_curves:
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))

            # ROC plot
            axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
            axes[0].plot([0, 1], [0, 1], linestyle="--")
            axes[0].set_title(f"{name} - ROC Curve")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].legend(loc="lower right")
            axes[0].grid(alpha=0.3)

            # PR plot
            axes[1].plot(pr_recall, pr_precision, label=f"PR AUC = {pr_auc_trapz:.4f}\nAP = {ap_score:.4f}")
            axes[1].set_title(f"{name} - Precision-Recall Curve")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].legend(loc="lower left")
            axes[1].grid(alpha=0.3)

            # Confusion matrix plot
            axes[2].imshow(cm, interpolation="nearest", cmap="Blues")
            axes[2].set_title(f"{name} - Confusion Matrix")
            labels = ["Non-toxic", "Toxic"]
            axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(labels)
            axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(labels)
            axes[2].set_xlabel("Predicted")
            axes[2].set_ylabel("True")
            for i in range(2):
                for j in range(2):
                    axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                                 fontsize=14)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved to: {save_path}")

            plt.show()

    else:
        print("y_score not provided -> skipping ROC/PR/AUC computation.")

    return metrics


# ── Runner ────────────────────────────────────────────────────────────────────
# Run directly: python analysis/models/evaluator.py

if __name__ == "__main__":
    import pickle
    import scipy.sparse as sp
    from scipy.sparse import hstack

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    from analysis.pipeline_and_dispatch.data_pipeline import DataPipeline
    from analysis.features.build_features import DenseFeatureTransformer

    OUTPUT_DIR = "analysis/models/all_outputs/lasso_log_reg"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    with open(os.path.join(OUTPUT_DIR, "lasso_log_reg_tuned.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    scaler_dense = bundle["scaler_dense"]
    scaler_bert  = bundle["scaler_bert"]
    threshold    = bundle["threshold"]
    model.decision_threshold = threshold

    # Load data + reconstruct test features
    dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
    _, X_test_raw, _, y_test = dp.get_data()
    y_test = y_test.values.ravel()

    dense_transformer = DenseFeatureTransformer()
    X_test_dense      = scaler_dense.transform(dense_transformer.transform(X_test_raw))
    X_test_tfidf      = sp.load_npz("data/processed/tfidf_test.npz")
    with open("data/processed/bert_test.pkl", "rb") as f:
        X_test_bert = scaler_bert.transform(pickle.load(f))

    X_test = hstack([
        sp.csr_matrix(X_test_dense),
        X_test_tfidf,
        sp.csr_matrix(X_test_bert),
    ]).tocsr()

    # Evaluate
    y_pred  = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    evaluate_classification(
        y_test, y_pred, y_score,
        name="LassoLogisticRegression",
        plot_curves=True,
        save_path=os.path.join(OUTPUT_DIR, "lasso_log_reg_tuned_evaluation.png"),
    )

    print(f"\nDecision threshold used: {threshold:.4f}")
