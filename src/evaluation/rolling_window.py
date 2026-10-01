"""Rolling and expanding-window out-of-sample evaluation.

This module provides two complementary schemes used throughout the
finance-ML literature (e.g. Gu, Kelly, Xiu 2020; Bali, Beckmeyer, Moerke,
Weigert 2022):

* ``expanding_window_eval`` — each retrain uses *all* data available so
  far.  Larger training sample, slower to adapt to regime shifts.
* ``rolling_window_eval``   — each retrain uses a fixed-length window
  immediately preceding the test block.  Adapts faster to regimes; uses
  less data per fit.

Both produce per-window metrics AND a pooled prediction vector so the
caller can compute a single out-of-sample R^2 over the whole concatenated
test period (Bali et al. 2022, Eq. 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.logger import setup_logger
from src.evaluation.metrics import evaluate_model, oos_r2

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pooled_metrics(
    pred_ret: np.ndarray,
    y_ret: np.ndarray,
    pred_dir: np.ndarray,
    y_dir: np.ndarray,
    y_train_mean: float,
) -> dict:
    """Pooled R²/IC/accuracy computed on concatenated OOS predictions."""
    pooled = {
        "pooled_r2": float(oos_r2(y_ret, pred_ret, y_train_mean)),
        "pooled_rmse": float(np.sqrt(np.mean((y_ret - pred_ret) ** 2))),
        "pooled_accuracy": float((pred_dir == y_dir).mean()),
    }
    if len(y_ret) > 10:
        ic, _ = spearmanr(y_ret, pred_ret)
        pooled["pooled_ic"] = float(ic)
    else:
        pooled["pooled_ic"] = np.nan
    return pooled


def _run_window(
    window_idx: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    X: np.ndarray,
    y_ret: np.ndarray,
    y_dir: np.ndarray,
    dates: np.ndarray,
    unique_dates: np.ndarray,
    train_start_idx: int,
    train_end_idx: int,
    test_start_idx: int,
    test_end_idx: int,
    model_factory,
    transaction_cost_bps: float,
) -> tuple[dict | None, np.ndarray | None, np.ndarray | None,
           np.ndarray | None, np.ndarray | None, float | None]:
    """Fit one model and return per-window metrics plus raw predictions.

    Predictions are returned so the caller can build the pooled OOS vector
    across all windows.
    """
    X_tr, y_tr_ret, y_tr_dir = X[train_mask], y_ret[train_mask], y_dir[train_mask]
    X_te, y_te_ret, y_te_dir = X[test_mask], y_ret[test_mask], y_dir[test_mask]

    if len(X_te) == 0 or len(X_tr) == 0:
        return None, None, None, None, None, None

    model = model_factory()
    try:
        model.fit(X_tr, y_tr_ret)
        pred_ret = model.predict(X_te)
        pred_dir = (pred_ret > 0).astype(int)

        metrics = evaluate_model(
            model_name=f"window_{window_idx}",
            y_test_ret=y_te_ret,
            y_test_dir=y_te_dir,
            pred_ret=pred_ret,
            pred_dir=pred_dir,
            pred_proba=None,
            y_train_mean=y_tr_ret.mean(),
            transaction_cost_bps=transaction_cost_bps,
            dates=dates[test_mask],
        )
        metrics["window"] = window_idx
        metrics["train_start"] = unique_dates[train_start_idx]
        metrics["train_end"] = unique_dates[train_end_idx - 1]
        metrics["test_start"] = unique_dates[test_start_idx]
        metrics["test_end"] = unique_dates[test_end_idx - 1]
        metrics["n_train"] = int(len(X_tr))
        metrics["n_test"] = int(len(X_te))
        return (
            metrics, pred_ret, y_te_ret, pred_dir, y_te_dir,
            float(y_tr_ret.mean()),
        )
    except Exception as e:
        log.error("Window %d failed: %s", window_idx, e)
        return None, None, None, None, None, None


def expanding_window_eval(
    dates: np.ndarray,
    X: np.ndarray,
    y_ret: np.ndarray,
    y_dir: np.ndarray,
    model_factory,
    step: int = 63,
    min_train: int = 504,
    transaction_cost_bps: float = 5,
) -> tuple[pd.DataFrame, dict]:
    """Expanding-window evaluation with periodic re-training.

    Each retrain uses *all* data up to the test window.  Returns per-window
    metrics as a DataFrame and a ``pooled`` dict containing the $R^2$/IC/
    accuracy over concatenated OOS predictions.
    """
    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)

    if n_dates < min_train + step:
        log.warning("Not enough data for expanding-window evaluation")
        return pd.DataFrame(), {}

    results = []
    all_pred_ret, all_y_ret = [], []
    all_pred_dir, all_y_dir = [], []
    train_means = []
    window_idx = 0

    for train_end in range(min_train, n_dates - step, step):
        test_start = train_end
        test_end = min(train_end + step, n_dates)

        train_dates_set = set(unique_dates[:train_end])
        test_dates_set = set(unique_dates[test_start:test_end])
        train_mask = np.isin(dates, list(train_dates_set))
        test_mask = np.isin(dates, list(test_dates_set))

        metrics, pr, yr, pd_, yd, tm = _run_window(
            window_idx, train_mask, test_mask, X, y_ret, y_dir,
            dates, unique_dates, 0, train_end, test_start, test_end,
            model_factory, transaction_cost_bps,
        )
        if metrics is not None:
            results.append(metrics)
            all_pred_ret.append(pr); all_y_ret.append(yr)
            all_pred_dir.append(pd_); all_y_dir.append(yd)
            train_means.append(tm)
        window_idx += 1

    df = pd.DataFrame(results)
    pooled = {}
    if not df.empty:
        pr = np.concatenate(all_pred_ret); yr = np.concatenate(all_y_ret)
        pd_ = np.concatenate(all_pred_dir); yd = np.concatenate(all_y_dir)
        pooled = _pooled_metrics(pr, yr, pd_, yd, np.mean(train_means))
        log.info(
            "Expanding eval: %d windows, pooled R²=%.4f, pooled IC=%.4f, "
            "pooled acc=%.4f",
            len(df), pooled["pooled_r2"], pooled["pooled_ic"],
            pooled["pooled_accuracy"],
        )
    return df, pooled


# ---------------------------------------------------------------------------
# Rolling window (fixed-length training)
# ---------------------------------------------------------------------------

def rolling_window_eval(
    dates: np.ndarray,
    X: np.ndarray,
    y_ret: np.ndarray,
    y_dir: np.ndarray,
    model_factory,
    train_size: int = 189,
    test_size: int = 63,
    step: int | None = None,
    transaction_cost_bps: float = 5,
) -> tuple[pd.DataFrame, dict]:
    """Rolling-window evaluation with fixed-length training.

    Parameters
    ----------
    train_size : trading days in each training window (default 189 ≈ 9 months).
    test_size  : trading days in each test block (default 63 ≈ 3 months).
    step       : days between successive test blocks (default = ``test_size``,
                 i.e. non-overlapping test blocks that tile the whole sample).

    Each retrain uses the ``train_size`` trading days immediately preceding
    the test block; earlier data is discarded.  This is the
    "9m train / 3m test, slide and repeat" scheme of Bali et al. 2022's
    robustness check.
    """
    if step is None:
        step = test_size

    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)

    if n_dates < train_size + test_size:
        log.warning("Not enough data for rolling-window evaluation")
        return pd.DataFrame(), {}

    results = []
    all_pred_ret, all_y_ret = [], []
    all_pred_dir, all_y_dir = [], []
    train_means = []
    window_idx = 0

    test_start_idx = train_size
    while test_start_idx + test_size <= n_dates:
        train_start_idx = test_start_idx - train_size
        train_end_idx = test_start_idx
        test_end_idx = test_start_idx + test_size

        train_dates_set = set(unique_dates[train_start_idx:train_end_idx])
        test_dates_set = set(unique_dates[test_start_idx:test_end_idx])
        train_mask = np.isin(dates, list(train_dates_set))
        test_mask = np.isin(dates, list(test_dates_set))

        metrics, pr, yr, pd_, yd, tm = _run_window(
            window_idx, train_mask, test_mask, X, y_ret, y_dir,
            dates, unique_dates,
            train_start_idx, train_end_idx, test_start_idx, test_end_idx,
            model_factory, transaction_cost_bps,
        )
        if metrics is not None:
            results.append(metrics)
            all_pred_ret.append(pr); all_y_ret.append(yr)
            all_pred_dir.append(pd_); all_y_dir.append(yd)
            train_means.append(tm)

        window_idx += 1
        test_start_idx += step

    df = pd.DataFrame(results)
    pooled = {}
    if not df.empty:
        pr = np.concatenate(all_pred_ret); yr = np.concatenate(all_y_ret)
        pd_ = np.concatenate(all_pred_dir); yd = np.concatenate(all_y_dir)
        pooled = _pooled_metrics(pr, yr, pd_, yd, np.mean(train_means))
        log.info(
            "Rolling eval (%d-day train / %d-day test): %d windows, "
            "pooled R²=%.4f, pooled IC=%.4f, pooled acc=%.4f",
            train_size, test_size, len(df),
            pooled["pooled_r2"], pooled["pooled_ic"],
            pooled["pooled_accuracy"],
        )
    return df, pooled
