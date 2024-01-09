import numpy as np
from scipy.stats import norm
import pandas as pd

class VarCov:
    """Uses the variance / covariance method and the linearized loss operator to calculate the Value at risk and the Expected Shortfall."""

    def __init__(self):
        self.mu_hat = 0
        self.cov_hat = 0

    def fit(self, x, w, alpha=0.99):
        """Fits the model to the (portfolio) data and returns VaR and ES. \n
        X is the log-difference matrix of the form (N x A ) with A=number of assets.
        Alpha: the corresponding confidence level, between 0 and 1. The standard is 0.99.\n
        w: The weights / price vector. With shape (A x 1) E.g. (120, 90, 80).T"""

        num_assets = x.shape[1]
        n = x.shape[0]
        self.mu_hat = np.mean(x, axis=0).reshape((num_assets, 1))
        self.cov_hat = np.cov(x, rowvar=False)

        VaR = -w.T @ self.mu_hat + np.sqrt(w.T @ self.cov_hat @ w) * norm.ppf(alpha)
        ES = -w.T @ self.mu_hat + np.sqrt(w.T @ self.cov_hat @ w) * norm.pdf(norm.ppf(alpha))/(1 - alpha)
        return VaR.item(0), ES.item(0)

