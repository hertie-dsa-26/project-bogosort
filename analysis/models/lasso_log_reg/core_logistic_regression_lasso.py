import numpy as np
import warnings 
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning 
from sklearn.utils.validation import check_array 

class LassoLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01, learning_rate=0.1, max_iter=1000, tol=1e-4, fit_intercept=True, decision_threshold=0.5, verbose=False): 
        self.alpha = alpha              
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.decision_threshold = decision_threshold 
        self.verbose = verbose 

    def _sigmoid(self, z): # This is the sigmoid we will throw our summed value into for a Prob
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip it here to prevent overflow. after all, the limit of 500 is basically 1

    def _soft_threshold(self, beta, threshold): # We need soft thresholding since we add the L1 penalty - which has this kink at zero (i.e., undefined gradient, not zero)
        return np.sign(beta) * np.maximum(np.abs(beta) - threshold, 0) # Enforces sparsity (with maximum), and then multiplies by the sign to get the right weight direction back

    def fit(self, X, y, sample_weight=None):
        X = check_array(X, accept_sparse=True) # Checks whether we actually have an array, that the dimension is right, that we don't have Inf's, NaN's, or strings, and converts to array if not.
        y = np.asarray(y) # Fix the outcome as array, in case not already.
        if not set(np.unique(y).tolist()).issubset({0, 1}): # This ensures that y is binary, and can't be negative or something else weird 
            raise ValueError(f"y must contain only 0/1 labels, got {np.unique(y)}")
        n, n_features = X.shape 
        self.classes_ = np.unique(y) # extract possible classes
        if len(self.classes_) != 2: # just a quick check to make sure we actually only have two classes, as otherwise things break.
            raise ValueError(f"Need both 0 and 1 classes in y, got {self.classes_}")
        self.coef_ = np.zeros(n_features) # Construct all coefficients as zero to begin with. _ at end to indicate how it only exists after training
        self.intercept_ = 0.0

        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float) # This builds the sample weight.
        sw_sum = sw.sum() # normalize gradient by total weight (e.g., + sample_weight) so that step sizes are comparable across the different weight schemes. If we had no weight, we would just divide by n
        if sw_sum <= 0: # Make sure that the sample_weight is not zero, since we can't divide by zero. That would just give a bunch of NaNs
            raise ValueError(f"sample_weight must have a positive total, got sw_sum={sw_sum}")

        converged = False # To track why we exited to loop. Could also do a loop, but this is simpler and more efficient.
        iteration = -1 # This helps to see even if the number of iterations is just 0, it'll track this.
        for iteration in range(self.max_iter): # the actual learning process, goes until max number of iterations is hit, or not converging meaningfully anymore
            p_hat = self._sigmoid(X @ self.coef_ + self.intercept_) # Computes probabilities for all training points using current weights by calculating dot product
            residual = p_hat - y # size is n - one number per training point
            weighted_residual = sw * residual # This is the big role of sample_weight, we multiply the residual with the weight to push the gradient more if needed.
            grad_coef = (X.T @ weighted_residual) / sw_sum # # Calculate slope of each coefficient to identify direction for descent. This is a simplified and reduced way of getting at this, since the slope is all we need. We use the weighted residual and the total weight instead of raw residual and n, though.

            coef_new = self.coef_ - self.learning_rate * grad_coef # Take a step in the opposite direction of the gradients, whether gradient is pos or neg
            coef_new = self._soft_threshold(coef_new, self.alpha * self.learning_rate) # Run coef_new through lasso soft threshold. Multiply alpha and learning rate together to have regularizing effect be proportional to each other. 

            intercept_new = self.intercept_ 
            if self.fit_intercept: # if clause just in case we don't want an intercept, but we basically always do
                intercept_new = self.intercept_ - self.learning_rate * (weighted_residual.sum() / sw_sum) # CHANGED: WRITE INTO intercept_new INSTEAD OF MUTATING self.intercept_ DIRECTLY (same math as before — weighted mean of residual scaled by learning rate — just deferred so delta can include it)

            delta = max(
                np.max(np.abs(coef_new - self.coef_)), # original coef-change measurement
                abs(intercept_new - self.intercept_), # intercept change. Normally, this should follow the coefficients quite naturally, but here we just track in case something strange happens, by making it part of the convergence.
            )
            self.coef_ = coef_new # rewrite the coefficients
            self.intercept_ = intercept_new # commit the intercept update after diff has been measured. 

            if delta < self.tol: # If the delta is smaller than the tolerance, then we stop iterating and consider the model converged
                converged = True # Change the convergence so that post-loop knows not to warn 
                if self.verbose: # here we change whether we want to see the individual convergence iteration outputs, which in a CV could be hundreds.
                    print(f"Model converged at iteration {iteration}")
                break

        self.n_iter_ = iteration + 1 # Tracks how many iterations we actually ran. 
        if not converged: # Throw a warning when we hit max_iter without delta going below tol.
            warnings.warn(
                f"LassoLogisticRegression did not converge after {self.max_iter} iterations "
                f"(final delta={delta:.2e}, tol={self.tol}). Consider increasing max_iter or learning_rate.",
                ConvergenceWarning,
            )
        return self

    def predict_proba(self, X): # Function for building the probabilities for each sample, in case we want to showcase this
        p = self._sigmoid(X @ self.coef_ + self.intercept_) # build probabilities through sigmoid for all of X
        return np.stack([1 - p, p], axis=1) # Just stacks each prob function against each other as (n, 2) array

    def predict(self, X):
        return (self._sigmoid(X @ self.coef_ + self.intercept_) >= self.decision_threshold).astype(int) # Function for assigning classes, just takes above and feeds it through the threshold

    def score(self, X, y): # Function for just giving the plain accuracy
        return np.mean(self.predict(X) == y)