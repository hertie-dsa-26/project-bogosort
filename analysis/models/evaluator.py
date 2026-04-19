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
    ConfusionMatrixDisplay,
    confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, y_score=None,name="Model", plot_curves=True,
    save_path="./analysis/models/model_outputs/{name}_evaluation.png"):
    print(f"\n===== {name} Evaluation =====")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\nClassification Report:\n")
    clf_report = classification_report(y_true, y_pred)
    print(clf_report)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": clf_report
    }

    # ----- ROC / PR / AUC -----
    if y_score is not None:
        y_score = np.asarray(y_score).ravel()

        # ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        # PR
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        pr_auc_trapz = auc(pr_recall, pr_precision)              # area under PR curve via trapezoid
        ap_score = average_precision_score(y_true, y_score)      # Average Precision (recommended PR summary)

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
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved ROC/PR curves to: {save_path}")

            plt.show()

    else:
        print("y_score not provided -> skipping ROC/PR/AUC computation.")

    return metrics