"""Linear SVM model + training entrypoint."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

# Linear SVM model
class ManualLinearSVM(BaseEstimator, ClassifierMixin):
    """Linear SVM (hinge loss) + L2."""

    def __init__(self, c=1.0, learning_rate=0.01, max_iter=2000, tol=1e-5, fit_intercept=True):
        self.c = c
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _validate_binary_target(self, y):
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("SVM supports only binary classification")
        if not np.all(np.isin(classes, [0, 1])):
            raise ValueError("Target labels must be in scope of{0, 1}")

# Fit model
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._validate_binary_target(y)

        _, n_features = X.shape
        y_signed = np.where(y == 1, 1.0, -1.0)

        self.classes_ = np.array([0, 1])
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            margin = y_signed * (X @ self.coef_ + self.intercept_)
            active = margin < 1.0

            grad_w = self.coef_.copy()
            if np.any(active):
                grad_w -= self.c * np.mean((y_signed[active, None] * X[active]), axis=0)

            if self.fit_intercept and np.any(active):
                grad_b = -self.c * np.mean(y_signed[active])
            else:
                grad_b = 0.0

            new_coef = self.coef_ - self.learning_rate * grad_w
            new_intercept = self.intercept_ - self.learning_rate * grad_b

            delta_w = np.max(np.abs(new_coef - self.coef_))
            delta_b = abs(new_intercept - self.intercept_)
            self.coef_ = new_coef
            self.intercept_ = new_intercept

            if max(delta_w, delta_b) < self.tol:
                break

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


# Best Threshold search on validation scores
def find_best_threshold(y_true, scores, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(-1.0, 1.0, 81)

    best_threshold = 0.0
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        curr_f1 = f1_score(y_true, y_pred, zero_division=0)
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_threshold = float(thr)

    return best_threshold, float(best_f1)

# Load training matrix and drop columns
def load_training_matrix(
    csv_path,
    target_col="toxic",
    drop_cols=None,
):
    if drop_cols is None:
        drop_cols = [
            "id",
            "comment_text",
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feature_cols = [col for col in df.columns if col not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns left after applying drop_cols.")

    X = df[feature_cols].fillna(0.0).astype(float).to_numpy()
    y = df[target_col].astype(int).to_numpy()

    return X, y, feature_cols

# Run stratified Crossvalidation
def run_stratified_kfold(
    X,
    y,
    n_splits=5,
    c=1.0,
    learning_rate=0.01,
    max_iter=2000,
    tol=1e-5,
    random_state=42,
    use_fixed_threshold=True,
    fixed_threshold=-0.595,
):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = ManualLinearSVM(
            c=c,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=True,
        )
        model.fit(X_train, y_train)

        val_scores = model.decision_function(X_val)
        if use_fixed_threshold:
            # Use threshold from prior tuning for stable final runs.
            best_thr = float(fixed_threshold)
        else:
            # Recompute best threshold on each validation fold.
            best_thr, _ = find_best_threshold(y_true=y_val, scores=val_scores)
        y_pred = (val_scores >= best_thr).astype(int)

        metrics = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "threshold": best_thr,
        }
        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)

# Hyperparameter tuning
def tune_hyperparameters(
    X,
    y,
    param_grid=None,
    n_splits=5,
    random_state=42,
    use_fixed_threshold=True,
    fixed_threshold=-0.595,
):
    if param_grid is None:
        param_grid = {
            "c": [0.1, 1.0, 3.0],
            "learning_rate": [0.001, 0.01],
            "max_iter": [2000, 4000],
            "tol": [1e-5],
        }

    search_rows = []
    for params in ParameterGrid(param_grid):
        fold_df = run_stratified_kfold(
            X,
            y,
            n_splits=n_splits,
            c=params["c"],
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            tol=params["tol"],
            random_state=random_state,
            use_fixed_threshold=use_fixed_threshold,
            fixed_threshold=fixed_threshold,
        )
        mean_metrics = fold_df[["accuracy", "precision", "recall", "f1"]].mean()
        search_rows.append({
            "c": params["c"],
            "learning_rate": params["learning_rate"],
            "max_iter": params["max_iter"],
            "tol": params["tol"],
            "mean_accuracy": float(mean_metrics["accuracy"]),
            "mean_precision": float(mean_metrics["precision"]),
            "mean_recall": float(mean_metrics["recall"]),
            "mean_f1": float(mean_metrics["f1"]),
            "mean_threshold": float(fold_df["threshold"].mean()),
        })

    search_df = pd.DataFrame(search_rows).sort_values(
        by=["mean_f1", "mean_recall", "mean_precision"],
        ascending=False,
    ).reset_index(drop=True)
    best_params = search_df.iloc[0].to_dict()
    return search_df, best_params


def main():
    #path to training data
    default_data_path = Path(__file__).resolve().parents[2] / "01_data" / "01_processed" / "train_set_with_features.csv"

    #Load features and target
    X, y, feature_cols = load_training_matrix(default_data_path, target_col="toxic")
    print(f"Loaded matrix: X={X.shape}, y={y.shape}, features={len(feature_cols)}")

    # Toggle between fixed threshold and per-fold threshold search.
    use_fixed_threshold = True
    # Fixed threshold chosen from prior tuning.
    final_threshold = -0.595
    results = run_stratified_kfold(
        X,
        y,
        n_splits=5,
        c=1.0,
        learning_rate=0.01,
        max_iter=2000,
        tol=1e-5,
        random_state=42,
        use_fixed_threshold=use_fixed_threshold,
        fixed_threshold=final_threshold,
    )

    print("\nFold metrics (fixed threshold):")
    print(results.to_string(index=False))

    print("\nMean metrics (fixed threshold):")
    print(results[["accuracy", "precision", "recall", "f1"]].mean().to_string())
    print(f"Fixed threshold used in all folds: {final_threshold:.4f}")

    # Hyperparameter tuning with 5-fold CV
    tuning_results, best = tune_hyperparameters(
        X,
        y,
        n_splits=5,
        random_state=42,
        use_fixed_threshold=use_fixed_threshold,
        fixed_threshold=final_threshold,
    )

    print("\nHyperparameter tuning (top 5 by mean_f1):")
    print(tuning_results.head(5).to_string(index=False))

    print("\nBest config.:")
    print(pd.Series(best).to_string())


if __name__ == "__main__":
    main()
