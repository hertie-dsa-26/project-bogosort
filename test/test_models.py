import numpy as np
import pytest

from analysis.models.core_logistic_regression_lasso import LassoLogisticRegression
from analysis.models.evaluator import evaluate_classification
from analysis.models.run_model import model_run


@pytest.fixture
def model():
    return LassoLogisticRegression()


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 4))
    y = (X[:, 0] > 0).astype(int)
    m = LassoLogisticRegression(alpha=0.01, max_iter=500)
    m.fit(X, y)
    return m, X


class TestSigmoid:
    def test_zero_input_returns_half(self, model):
        # zero is the symmetry point — neither class is favoured
        assert model._sigmoid(0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self, model):
        # strong positive signal should mean near-certain class 1
        assert model._sigmoid(100) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_zero(self, model):
        # strong negative signal should mean near-certain class 0
        assert model._sigmoid(-100) == pytest.approx(0.0, abs=1e-6)

    def test_clip_prevents_overflow(self, model):
        # values beyond ±500 are clipped — result should still be a valid float, not nan
        result = model._sigmoid(1e9)
        assert np.isfinite(result)


class TestSoftThreshold:
    def test_value_inside_threshold_becomes_zero(self, model):
        # coefficients smaller than the penalty are killed completely — this is what makes Lasso sparse
        assert model._soft_threshold(0.03, 0.05) == 0.0

    def test_negative_value_inside_threshold_becomes_zero(self, model):
        # sparsity applies symmetrically to negative coefficients
        assert model._soft_threshold(-0.03, 0.05) == 0.0

    def test_value_outside_threshold_shrinks_by_lambda(self, model):
        # surviving coefficients are pulled toward zero by exactly lambda, not left unchanged
        assert model._soft_threshold(0.8, 0.05) == pytest.approx(0.75)

    def test_negative_value_outside_threshold_shrinks_by_lambda(self, model):
        # shrinkage toward zero means a negative coefficient moves in the positive direction
        assert model._soft_threshold(-0.8, 0.05) == pytest.approx(-0.75)

    def test_sign_is_preserved(self, model):
        # the penalty shrinks the weight but must never flip it to the wrong direction
        assert model._soft_threshold(-0.9, 0.05) < 0


@pytest.fixture
def sparse_data():
    # 200 samples, 20 features — only the first two actually predict y
    rng = np.random.default_rng(42)
    n, p = 200, 20
    X = rng.standard_normal((n, p))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


class TestSparsity:
    def test_high_alpha_zeros_most_coefficients(self, sparse_data):
        # strong regularization should drive noise features to exactly zero
        X, y = sparse_data
        model = LassoLogisticRegression(alpha=1.0, max_iter=1000)
        model.fit(X, y)
        assert np.sum(model.coef_ == 0) > len(model.coef_) // 2

    def test_zero_alpha_produces_fewer_zeros_than_high_alpha(self, sparse_data):
        # without a penalty there is no pressure to zero anything out
        X, y = sparse_data
        model_low  = LassoLogisticRegression(alpha=0.0, max_iter=1000)
        model_high = LassoLogisticRegression(alpha=1.0, max_iter=1000)
        model_low.fit(X, y)
        model_high.fit(X, y)
        assert np.sum(model_low.coef_ == 0) < np.sum(model_high.coef_ == 0)


class TestPredictions:
    def test_predict_proba_shape(self, fitted_model):
        # one probability pair per sample — (n, 2) is the sklearn contract
        m, X = fitted_model
        assert m.predict_proba(X).shape == (len(X), 2)

    def test_predict_proba_values_in_range(self, fitted_model):
        # every value must be a valid probability — nothing below 0 or above 1
        m, X = fitted_model
        proba = m.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_rows_sum_to_one(self, fitted_model):
        # the two columns are complements — P(class 0) + P(class 1) must equal 1
        m, X = fitted_model
        row_sums = m.predict_proba(X).sum(axis=1)
        assert row_sums == pytest.approx(np.ones(len(X)))

    def test_predict_returns_binary(self, fitted_model):
        # a binary classifier must only ever output 0 or 1, nothing in between
        m, X = fitted_model
        assert set(m.predict(X)).issubset({0, 1})

    def test_threshold_zero_predicts_all_ones(self, fitted_model):
        # sigmoid output is always strictly > 0, so every sample clears a threshold of 0
        m, X = fitted_model
        assert np.all(m.predict(X, threshold=0.0) == 1)


class TestScore:
    def test_perfect_accuracy_on_separable_data(self):
        # a wide decision boundary leaves no room for misclassification
        rng = np.random.default_rng(7)
        n = 100
        X = rng.standard_normal((n, 4))
        X[:n // 2, 0] += 10
        X[n // 2:, 0] -= 10
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        model = LassoLogisticRegression(alpha=0.01, max_iter=1000)
        model.fit(X, y)
        assert model.score(X, y) == 1.0


class TestIntercept:
    def test_fit_intercept_false_keeps_intercept_at_zero(self, sparse_data):
        # the flag must suppress intercept updates, not just initialise to zero
        X, y = sparse_data
        model = LassoLogisticRegression(fit_intercept=False, max_iter=500)
        model.fit(X, y)
        assert model.intercept_ == 0.0


class TestConvergence:
    def test_converges_before_max_iter_on_separable_data(self, capsys):
        # a large-margin dataset should let gradient descent stabilise long before the iteration cap
        rng = np.random.default_rng(0)
        n = 100
        X = rng.standard_normal((n, 5))
        X[:n // 2, 0] += 10   # class 1 is far positive on feature 0
        X[n // 2:, 0] -= 10   # class 0 is far negative on feature 0
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        model = LassoLogisticRegression(alpha=0.01, max_iter=1000, tol=1e-4)
        model.fit(X, y)

        assert "converged at iteration" in capsys.readouterr().out


class TestEvaluateClassification:
    def test_perfect_predictions_give_all_metrics_one(self):
        # every metric collapses to 1.0 when there are no errors at all
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        result = evaluate_classification(y_true, y_pred, name="test")
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_return_dict_contains_expected_keys(self):
        # downstream code indexes into these keys — missing one would break silently
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = evaluate_classification(y_true, y_pred, name="test")
        assert {"accuracy", "f1", "precision", "recall", "classification_report"}.issubset(result.keys())

    def test_roc_pr_keys_absent_without_y_score(self):
        # curve metrics require probability scores — they must not appear when only hard labels are given
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_classification(y_true, y_pred, name="test")
        assert "roc_auc" not in result
        assert "pr_auc" not in result

    def test_roc_pr_keys_present_with_y_score(self):
        # passing probability scores should unlock the curve metrics in the returned dict
        y_true  = np.array([0, 1, 0, 1])
        y_pred  = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        result = evaluate_classification(y_true, y_pred, y_score, name="test", plot_curves=False)
        assert "roc_auc" in result
        assert "pr_auc" in result

    def test_imbalanced_labels_do_not_crash(self):
        # zero_division=0 should absorb the 0/0 precision when no positives are predicted
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        evaluate_classification(y_true, y_pred, name="test")


class TestModelRegistry:
    def test_invalid_model_name_raises_value_error(self):
        # an unrecognised name should fail loudly rather than import something unexpected
        with pytest.raises(ValueError):
            model_run("svm", data_path="", mode="train", save_predictions=False, save_model=False)
