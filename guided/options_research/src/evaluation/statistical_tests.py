"""Statistical tests: Clark-West, Diebold-Mariano, McNemar, bootstrap CI."""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.utils.logger import setup_logger

log = setup_logger(__name__)


def _newey_west_se(x: np.ndarray, bandwidth: int) -> float:
    """Newey-West HAC standard error for a 1D series."""
    n = len(x)
    x_demeaned = x - x.mean()
    gamma0 = np.dot(x_demeaned, x_demeaned) / n

    weighted_cov = gamma0
    for j in range(1, bandwidth + 1):
        weight = 1 - j / (bandwidth + 1)
        gamma_j = np.dot(x_demeaned[j:], x_demeaned[:-j]) / n
        weighted_cov += 2 * weight * gamma_j

    return np.sqrt(weighted_cov / n)


def clark_west_test(
    y_true: np.ndarray,
    pred_A: np.ndarray,
    pred_C: np.ndarray,
) -> dict:
    """Clark-West (2007) test for nested model comparison.

    Tests H0: Model A (restricted) forecasts as well as Model C (unrestricted).

    Parameters
    ----------
    y_true : actual values
    pred_A : predictions from restricted model (e.g., stock-only)
    pred_C : predictions from unrestricted model (e.g., all features)

    Returns
    -------
    dict with t_stat and p_value.
    """
    e_A = y_true - pred_A
    e_C = y_true - pred_C

    f_t = e_A ** 2 - (e_C ** 2 - (pred_A - pred_C) ** 2)

    n = len(f_t)
    bandwidth = max(1, int(np.floor(n ** (1 / 3))))
    se = _newey_west_se(f_t, bandwidth)

    t_stat = f_t.mean() / se if se > 0 else np.nan
    p_value = 1 - stats.norm.cdf(t_stat)  # one-sided

    log.info("Clark-West: t=%.3f, p=%.4f (bandwidth=%d)", t_stat, p_value, bandwidth)
    return {"t_stat": t_stat, "p_value": p_value}


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> dict:
    """Diebold-Mariano (1995) test for equal predictive accuracy.

    Parameters
    ----------
    horizon : int
        Forecast horizon (used for bandwidth).
    loss : str
        'squared' or 'absolute'.

    Returns
    -------
    dict with dm_stat and p_value.
    """
    e_A = y_true - pred_A
    e_B = y_true - pred_B

    if loss == "squared":
        d_t = e_A ** 2 - e_B ** 2
    else:
        d_t = np.abs(e_A) - np.abs(e_B)

    n = len(d_t)
    bandwidth = max(1, horizon - 1)
    se = _newey_west_se(d_t, bandwidth)

    dm_stat = d_t.mean() / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))  # two-sided

    log.info("Diebold-Mariano: DM=%.3f, p=%.4f", dm_stat, p_value)
    return {"dm_stat": dm_stat, "p_value": p_value}


def mcnemar_test(y_true: np.ndarray, pred_A: np.ndarray, pred_B: np.ndarray) -> dict:
    """McNemar's test comparing classification accuracy of two models.

    Returns chi2 stat and p-value.
    """
    correct_A = (pred_A == y_true)
    correct_B = (pred_B == y_true)

    # 2x2 contingency: (A right & B wrong), (A wrong & B right)
    b = np.sum(correct_A & ~correct_B)  # A right, B wrong
    c = np.sum(~correct_A & correct_B)  # A wrong, B right

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0}

    chi2 = (b - c) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    log.info("McNemar: chi2=%.3f, p=%.4f (b=%d, c=%d)", chi2, p_value, b, c)
    return {"chi2": chi2, "p_value": p_value}


def bootstrap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for SHAP-based feature importance.

    Parameters
    ----------
    shap_values : ndarray, shape (n_samples, n_features)
    feature_names : list[str]
    n_iterations : int
    alpha : float
        Significance level (0.05 => 95% CI).

    Returns
    -------
    dict with 'importance_df' (DataFrame with mean, lower, upper CI per feature)
    and 'rank_ci' (DataFrame with rank CIs).
    """
    import pandas as pd

    rng = np.random.RandomState(seed)
    n_samples, n_features = shap_values.shape

    # Bootstrap mean |SHAP| per feature
    boot_means = np.zeros((n_iterations, n_features))
    for i in range(n_iterations):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_means[i] = np.abs(shap_values[idx]).mean(axis=0)

    # Importance CIs
    lo = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    mean_imp = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": mean_imp,
        "ci_lower": lo,
        "ci_upper": hi,
        "significant": lo > 0,
    }).sort_values("mean_importance", ascending=False)

    # Rank CIs
    boot_ranks = np.zeros((n_iterations, n_features))
    for i in range(n_iterations):
        order = np.argsort(-boot_means[i])
        for rank, feat_idx in enumerate(order):
            boot_ranks[i, feat_idx] = rank + 1

    rank_lo = np.percentile(boot_ranks, 100 * alpha / 2, axis=0).astype(int)
    rank_hi = np.percentile(boot_ranks, 100 * (1 - alpha / 2), axis=0).astype(int)
    rank_median = np.median(boot_ranks, axis=0).astype(int)

    rank_df = pd.DataFrame({
        "feature": feature_names,
        "median_rank": rank_median,
        "rank_ci_lower": rank_lo,
        "rank_ci_upper": rank_hi,
    }).sort_values("median_rank")

    return {"importance_df": importance_df, "rank_df": rank_df}
