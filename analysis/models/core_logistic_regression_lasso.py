import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LassoLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01, learning_rate=0.1, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha              
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z): # This is the sigmoid we will throw our summed value into for a Prob
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip it here to prevent overflow. For example, 500 is basically 1

    def _soft_threshold(self, beta, threshold): # We need soft thresholding since we add the L1 penalty - which has this kink at zero (i.e., undefined gradient, not zero)
        return np.sign(beta) * np.maximum(np.abs(beta) - threshold, 0) # Enforces sparsity (with maximum), and then multiplies by the sign to get the right weight direction back

    def fit(self, X, y):
        y = np.asarray(y) # Fix the outcome as array, in case not already.
        n, p = X.shape # Define number of data points and total features
        self.classes_ = np.unique(y) # extract possible classes
        self.coef_ = np.zeros(p) # Construct all coefficients as zero to begin with. _ at end to indicate how it only exists after training
        self.intercept_ = 0.0

        for iteration in range(self.max_iter): # the actual learning process, goes until max number of iterations is hit, or not converging meaningfully anymore
            p_hat = self._sigmoid(X @ self.coef_ + self.intercept_) # Computes probabilities for all training points using current weights by calculating dot product
            residual = p_hat - y # size is n - one number per training point
            grad_coef = (X.T @ residual) / n # Calculate slope of each coefficient to identify direction for descent. This is a simplified and reduced way of getting at this.

            coef_new = self.coef_ - self.learning_rate * grad_coef # Take a step in the opposite direction of the gradients, whether gradient is pos or neg
            coef_new = self._soft_threshold(coef_new, self.alpha * self.learning_rate) # Run coef_new through lasso soft threshold. Multiply alpha and learning rate together to have regularizing effect be proportional to each other. 

            if self.fit_intercept: # if clause just in case we don't want an intercept, but we basically always do
                self.intercept_ -= self.learning_rate * np.mean(residual) # We subtract the mean residual, since this is equivalent to the derivate, and scale it to the size of the learning rate to avoid going too far or little

            delta = np.max(np.abs(coef_new - self.coef_)) # We measure the max any of the weights changed
            self.coef_ = coef_new # rewrite the coefficients

            if delta < self.tol: # If the delta is smaller than the tolerance, then we stop iterating and consider the model converged
                print(f"Model converged at iteration {iteration}")
                break
        return self

    def predict_proba(self, X): # Function for building the probabilities for each sample, in case we want to showcase this
        p = self._sigmoid(X @ self.coef_ + self.intercept_) # build probabilities through sigmoid for all of X
        return np.stack([1 - p, p], axis=1) # Just stacks each prob function against each other as (n, 2) array

    def predict(self, X, threshold=0.5): # Function for assigning classes, just takes above and feeds it through the threshold
        return (self._sigmoid(X @ self.coef_ + self.intercept_) >= threshold).astype(int)

    def score(self, X, y): # Function for just giving the plain accuracy
        return np.mean(self.predict(X) == y)