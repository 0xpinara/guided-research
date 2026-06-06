"""Conditional evaluation by VIX regime, earnings proximity, GEX sign."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def regime_eval(
    model_name: str,
    y_test_ret: np.ndarray,
    y_test_dir: np.ndarray,
    pred_ret: np.ndarray,
    pred_dir: np.ndarray,
    pred_proba: np.ndarray | None,
    y_train_mean: float,
    regime_labels: np.ndarray,
    regime_name: str,
    transaction_cost_bps: float = 5,
    dates: np.ndarray | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Evaluate a model separately for each regime.

    Parameters
    ----------
    regime_labels : ndarray of str/int, one per test observation.
        E.g., ["high_vix", "low_vix", "normal"] or [0, 1, 2].
    regime_name : str
        Name of this regime dimension (e.g., "vix_regime").

    Returns
    -------
    DataFrame with one row per regime.
    """
    results = []
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        if mask.sum() < 10:
            log.warning("Regime '%s' has only %d samples, skipping", regime, mask.sum())
            continue

        pp = pred_proba[mask] if pred_proba is not None else None
        sub_dates = dates[mask] if dates is not None else None
        metrics = evaluate_model(
            model_name=f"{model_name}|{regime_name}={regime}",
            y_test_ret=y_test_ret[mask],
            y_test_dir=y_test_dir[mask],
            pred_ret=pred_ret[mask],
            pred_dir=pred_dir[mask],
            pred_proba=pp,
            y_train_mean=y_train_mean,
            transaction_cost_bps=transaction_cost_bps,
            dates=sub_dates,
            horizon=horizon,
        )
        metrics["regime_name"] = regime_name
        metrics["regime_value"] = str(regime)
        metrics["n_samples"] = int(mask.sum())
        results.append(metrics)

    return pd.DataFrame(results)


def assign_vix_regimes(
    vix_values: np.ndarray,
    high_threshold: float = 25,
    low_threshold: float = 15,
) -> np.ndarray:
    """Assign VIX regime labels."""
    labels = np.where(
        vix_values >= high_threshold, "high_vix",
        np.where(vix_values <= low_threshold, "low_vix", "normal_vix"),
    )
    return labels


def assign_earnings_regime(
    days_to_earnings: np.ndarray,
    proximity_days: int = 7,
) -> np.ndarray:
    """Assign earnings proximity labels."""
    return np.where(days_to_earnings <= proximity_days, "near_earnings", "far_earnings")


def assign_gex_regime(gex_values: np.ndarray) -> np.ndarray:
    """Assign GEX sign regime."""
    return np.where(gex_values >= 0, "positive_gex", "negative_gex")


def full_regime_analysis(
    model_name: str,
    y_test_ret, y_test_dir, pred_ret, pred_dir, pred_proba,
    y_train_mean: float,
    test_features: pd.DataFrame,
    cfg,
    dates: np.ndarray | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Run regime analysis across all regime dimensions.

    test_features must contain: vix_close (feat_38), days_to_next_earnings
    (feat_43), gex_estimate (feat_21).
    """
    all_results = []

    def _run(labels, name):
        return regime_eval(
            model_name, y_test_ret, y_test_dir, pred_ret, pred_dir,
            pred_proba, y_train_mean, labels, name,
            dates=dates, horizon=horizon,
        )

    if "feat_38" in test_features.columns:
        vix_labels = assign_vix_regimes(
            test_features["feat_38"].values,
            cfg.regimes.vix_high, cfg.regimes.vix_low,
        )
        all_results.append(_run(vix_labels, "vix_regime"))

    if "feat_43" in test_features.columns:
        earn_labels = assign_earnings_regime(
            test_features["feat_43"].values, cfg.regimes.earnings_proximity_days,
        )
        all_results.append(_run(earn_labels, "earnings_regime"))

    if "feat_21" in test_features.columns:
        gex_labels = assign_gex_regime(test_features["feat_21"].values)
        all_results.append(_run(gex_labels, "gex_regime"))

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
