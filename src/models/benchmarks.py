"""Baseline/benchmark models: naive, historical mean, OLS, single-feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

from src.utils.logger import setup_logger

log = setup_logger(__name__)


class NaiveZero:
    """Predict zero return / majority class direction for all observations."""

    def __init__(self):
        self.train_majority = 1  # stocks go up ~53% of days

    def fit(self, X_train, y_train):
        self.train_majority = int((y_train > 0).mean() >= 0.5)
        return self

    def predict_return(self, X):
        return np.zeros(len(X))

    def predict_direction(self, X):
        return np.full(len(X), self.train_majority)


class HistoricalMean:
    """Predict the training-set mean return for all observations."""

    def __init__(self):
        self.mean_return = 0.0

    def fit(self, X_train, y_train):
        self.mean_return = y_train.mean()
        return self

    def predict_return(self, X):
        return np.full(len(X), self.mean_return)

    def predict_direction(self, X):
        return np.full(len(X), int(self.mean_return > 0))


class OLSModel:
    """OLS linear regression baseline."""

    def __init__(self, feature_cols=None):
        self.model = LinearRegression()
        self.feature_cols = feature_cols

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_return(self, X):
        return self.model.predict(X)

    def predict_direction(self, X):
        return (self.predict_return(X) > 0).astype(int)

    def summary(self, feature_names=None):
        """Return coefficient summary."""
        coefs = self.model.coef_
        intercept = self.model.intercept_
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(coefs))]
        return pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
        }).assign(intercept=intercept)


class SingleFeaturePredictor:
    """OLS with a single feature (for Pan-Poteshman, etc. replication)."""

    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.model = LinearRegression()
        self.alpha = 0.0
        self.beta = 0.0
        self.t_stat = 0.0

    def fit(self, X_train, y_train):
        """X_train should be shape (n, 1)."""
        X = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
        self.model.fit(X, y_train)
        self.alpha = self.model.intercept_
        self.beta = self.model.coef_[0]

        # Compute t-stat
        y_pred = self.model.predict(X)
        residuals = y_train - y_pred
        n = len(y_train)
        sse = np.sum(residuals ** 2)
        mse = sse / (n - 2)
        x_var = np.sum((X.ravel() - X.ravel().mean()) ** 2)
        se_beta = np.sqrt(mse / x_var) if x_var > 0 else np.nan
        self.t_stat = self.beta / se_beta if se_beta > 0 else np.nan

        return self

    def predict_return(self, X):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return self.model.predict(X)

    def predict_direction(self, X):
        return (self.predict_return(X) > 0).astype(int)

    def summary(self):
        return {
            "feature": self.feature_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "t_stat": self.t_stat,
        }


def run_benchmarks(
    X_train, y_train_ret, y_train_dir,
    X_test, y_test_ret, y_test_dir,
    feature_names=None,
) -> pd.DataFrame:
    """Run all benchmarks and return results table.

    Parameters
    ----------
    X_train, X_test : ndarray, shape (n, p)
    y_train_ret, y_test_ret : ndarray (continuous returns)
    y_train_dir, y_test_dir : ndarray (binary direction)
    feature_names : list[str], optional

    Returns
    -------
    DataFrame with one row per benchmark, columns for various metrics.
    """
    results = []

    # 1. Naive Zero
    nz = NaiveZero().fit(X_train, y_train_ret)
    results.append(_eval_benchmark("Naive Zero", nz, X_test, y_test_ret, y_test_dir, y_train_ret))

    # 2. Historical Mean
    hm = HistoricalMean().fit(X_train, y_train_ret)
    results.append(_eval_benchmark("Historical Mean", hm, X_test, y_test_ret, y_test_dir, y_train_ret))

    # 3. OLS Full
    ols = OLSModel().fit(X_train, y_train_ret)
    results.append(_eval_benchmark("OLS Full", ols, X_test, y_test_ret, y_test_dir, y_train_ret))

    # 4. Single-feature predictors
    single_features = {
        "Put-Call Volume (feat_01)": 0,
        "IV Change 1d (feat_10)": 9,
        "IV Skew (feat_08)": 7,
    }
    for name, idx in single_features.items():
        if idx < X_train.shape[1]:
            sf = SingleFeaturePredictor(name)
            sf.fit(X_train[:, idx], y_train_ret)
            pred_ret = sf.predict_return(X_test[:, idx])
            pred_dir = (pred_ret > 0).astype(int)
            r2 = 1 - np.sum((y_test_ret - pred_ret) ** 2) / np.sum((y_test_ret - y_train_ret.mean()) ** 2)
            acc = accuracy_score(y_test_dir, pred_dir)
            results.append({
                "model": name,
                "oos_r2": r2,
                "accuracy": acc,
                "beta": sf.beta,
                "t_stat": sf.t_stat,
            })

    return pd.DataFrame(results)


def _eval_benchmark(name, model, X_test, y_test_ret, y_test_dir, y_train_ret):
    """Evaluate a benchmark model."""
    pred_ret = model.predict_return(X_test)
    pred_dir = model.predict_direction(X_test)

    oos_r2 = 1 - np.sum((y_test_ret - pred_ret) ** 2) / np.sum((y_test_ret - y_train_ret.mean()) ** 2)
    acc = accuracy_score(y_test_dir, pred_dir)

    return {
        "model": name,
        "oos_r2": oos_r2,
        "accuracy": acc,
        "beta": np.nan,
        "t_stat": np.nan,
    }
