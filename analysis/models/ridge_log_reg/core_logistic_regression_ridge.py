# ============================================================
# Ridge Logistic Regression for Toxic Comment Classification
# ============================================================
# This script builds a Ridge (L2-regularized) logistic regression
# model from scratch and trains it on the Jigsaw toxic comment
# dataset. It predicts ONE label: "toxic" (binary: 0 or 1).
#
# WHAT IS RIDGE (L2) REGULARIZATION?
# Without regularization, the model might assign crazy-high
# weights to certain features, overfitting to the training data.
# Ridge adds a penalty proportional to the SQUARE of each weight.
# This pushes weights toward zero (keeping them small and stable)
# but never fully zeroes them out — every feature stays in the mix.
#
# IMPORTANT NOTE ON DATA:
# The file train_set_with_features.csv is ALREADY the training
# split (created by the team's 01_train_test_split_and_features
# notebook with random_state=42, stratified on "toxic").
# We do NOT split it again — we train on all of it.
# ============================================================

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib


# ============================================================
# PART 1: Define the Ridge Logistic Regression class
# ============================================================

class RidgeLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with L2 (Ridge) regularization,
    built from scratch using gradient descent.
    """

    def __init__(self, alpha=0.01, learning_rate=0.1, max_iter=1000,
                 tol=1e-4, fit_intercept=True):
        """
        Set up the model's settings (called "hyperparameters").

        Parameters:
        -----------
        alpha : float (default=0.01)
            Regularization strength. Higher alpha = stricter penalty
            on large weights. Think of it as "how tight the leash is."

        learning_rate : float (default=0.1)
            How big of a step we take each iteration when adjusting
            weights. Too big = we overshoot. Too small = too slow.

        max_iter : int (default=1000)
            Maximum number of training iterations.

        tol : float (default=1e-4)
            If weights change less than this between iterations,
            the model has converged (settled) and stops early.

        fit_intercept : bool (default=True)
            Whether to include a bias/intercept term.
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z):
        """
        Converts any number into a probability between 0 and 1.
        sigmoid(0) = 0.5, large positive -> ~1.0, large negative -> ~0.0
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """
        Train the model: find the best weight for each feature.

        The model:
        1. Starts with all weights at zero
        2. Makes predictions with current weights
        3. Checks how wrong the predictions are (the "residual")
        4. Calculates the gradient (which direction to adjust)
        5. Adds the Ridge penalty to the gradient
        6. Takes a step in the improved direction
        7. Repeats until predictions stop improving
        """
        y = np.asarray(y)
        n, p = X.shape
        self.classes_ = np.unique(y)
        self.coef_ = np.zeros(p)
        self.intercept_ = 0.0

        for iteration in range(self.max_iter):

            # Make predictions with current weights
            raw_score = X @ self.coef_ + self.intercept_
            p_hat = self._sigmoid(raw_score)

            # How wrong are we? (residual)
            residual = p_hat - y

            # Gradient: which direction should we adjust each weight?
            grad_coef = (X.T @ residual) / n

            # THE RIDGE (L2) PENALTY:
            # Add (alpha * current_weight) to the gradient.
            # This shrinks weights toward zero but never to zero.
            grad_coef = grad_coef + self.alpha * self.coef_

            # Update weights (move opposite to gradient)
            coef_new = self.coef_ - self.learning_rate * grad_coef

            # Update intercept (no penalty applied to intercept)
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * np.mean(residual)

            # Check convergence
            delta = np.max(np.abs(coef_new - self.coef_))
            self.coef_ = coef_new

            if delta < self.tol:
                print(f"  Model converged at iteration {iteration}")
                break

        return self

    def predict_proba(self, X):
        """
        Output probability scores.
        Returns [prob_NOT_toxic, prob_IS_toxic] for each comment.
        """
        p = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.stack([1 - p, p], axis=1)

    def predict(self, X, threshold=0.5):
        """
        Output final yes/no predictions (0 or 1).
        """
        return (self._sigmoid(X @ self.coef_ + self.intercept_) >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculate accuracy: what fraction of predictions are correct.
        """
        return np.mean(self.predict(X) == y)


# ============================================================
# PART 2: Load the training data
# ============================================================
# train_set_with_features.csv is ALREADY the training split.
# The team created it in 01_train_test_split_and_features.ipynb
# using random_state=42 and stratify=df["toxic"].
# We use it as-is — NO further splitting.
# ============================================================

print("Loading training data...")

data = pd.read_csv("data/processed/train_set_with_features.csv")

print(f"Training set: {data.shape[0]} comments, {data.shape[1]} columns")

# The ONE label we are predicting
target_column = "toxic"

# Columns to exclude from features
columns_to_exclude = ["id", "comment_text", "toxic", "severe_toxic",
                      "obscene", "threat", "insult", "identity_hate"]

# Everything else is a feature
feature_columns = [col for col in data.columns if col not in columns_to_exclude]

print(f"Number of features: {len(feature_columns)}")

# X = feature values (what the model sees)
# y = toxic label (what the model predicts)
X_train = data[feature_columns].values
y_train = data[target_column].values

print(f"Toxic rate in training data: {y_train.mean():.1%}")


# ============================================================
# PART 3: Scale the features
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# ============================================================
# PART 4: Train the Ridge Logistic Regression model
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL — predicting: toxic")
print("=" * 60)

model = RidgeLogisticRegression(
    alpha=0.01,
    learning_rate=0.1,
    max_iter=1000,
    tol=1e-4
)

model.fit(X_train_scaled, y_train)

# Report training performance (how well it fits the data it learned from)
train_preds = model.predict(X_train_scaled)
accuracy = accuracy_score(y_train, train_preds)
f1 = f1_score(y_train, train_preds, zero_division=0)

print(f"  Training Accuracy: {accuracy:.4f}")
print(f"  Training F1 Score: {f1:.4f}")


# ============================================================
# PART 5: Save the trained model for David's stacking
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

output_dir = "analysis/models/all_outputs/ridge_log_reg"
os.makedirs(output_dir, exist_ok=True)

# Save the trained model and scaler as .pkl files
joblib.dump(model, os.path.join(output_dir, "ridge_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "ridge_scaler.pkl"))
print(f"Saved ridge_model.pkl and ridge_scaler.pkl to {output_dir}/")

print("\n" + "=" * 60)
print("DONE! Ridge Logistic Regression complete.")
print("=" * 60)
