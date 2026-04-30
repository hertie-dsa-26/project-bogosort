"""Tests for manual linear SVM model module."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_analysis", "02_models"))

from core_vector_machine import (  # noqa: E402
    ManualLinearSVM,
    load_training_matrix,
    run_stratified_kfold,
    tune_hyperparameters,
)


def _make_linearly_separable():
    x_neg = np.array([[-3, -1], [-2, -2], [-2, -1], [-1, -2], [-3, -2]])
    x_pos = np.array([[1, 2], [2, 1], [2, 2], [3, 1], [3, 2]])
    X = np.vstack([x_neg, x_pos]).astype(float)
    y = np.array([0] * len(x_neg) + [1] * len(x_pos))
    return X, y


class TestManualLinearSVMUnit:
    def test_fit_sets_parameters(self):
        X, y = _make_linearly_separable()
        model = ManualLinearSVM(c=1.0, learning_rate=0.01, max_iter=5000, tol=1e-6)
        model.fit(X, y)
        assert model.coef_.shape[0] == X.shape[1]
        assert isinstance(model.intercept_, float)

    def test_predict_returns_binary_labels(self):
        X, y = _make_linearly_separable()
        model = ManualLinearSVM(max_iter=3000).fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_rejects_non_binary_target(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError):
            ManualLinearSVM().fit(X, y)


class TestVectorMachineIntegration:
    def test_run_stratified_kfold_returns_expected_columns(self):
        X, y = _make_linearly_separable()
        fold_df = run_stratified_kfold(X, y, n_splits=5, max_iter=3000)
        assert set(["fold", "accuracy", "precision", "recall", "f1"]).issubset(fold_df.columns)
        assert len(fold_df) == 5

    def test_tune_hyperparameters_returns_ranked_results(self):
        X, y = _make_linearly_separable()
        grid = {
            "c": [0.1, 1.0],
            "learning_rate": [0.001, 0.01],
            "max_iter": [1000],
            "tol": [1e-5],
        }
        tuning_df, best = tune_hyperparameters(X, y, param_grid=grid, n_splits=5, random_state=0)
        assert not tuning_df.empty
        assert "mean_f1" in tuning_df.columns
        assert best["mean_f1"] == tuning_df["mean_f1"].max()

    def test_load_training_matrix_uses_drop_cols_and_target(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "comment_text": ["a", "b", "c"],
            "toxic": [0, 1, 0],
            "severe_toxic": [0, 0, 0],
            "feat_a": [0.1, 0.2, 0.3],
            "feat_b": [1, 2, 3],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            path = tmp.name
        try:
            df.to_csv(path, index=False)
            X, y, feature_cols = load_training_matrix(path, target_col="toxic")
            assert X.shape == (3, 2)
            assert y.tolist() == [0, 1, 0]
            assert feature_cols == ["feat_a", "feat_b"]
        finally:
            os.remove(path)
