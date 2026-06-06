"""Regression, classification, and economic evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix,
)

from src.utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def oos_r2(y_true, y_pred, y_train_mean: float) -> float:
    """Out-of-sample R-squared using training-set mean as baseline."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def cross_sectional_oos_r2(y_true, y_pred, dates) -> float:
    """Cross-sectional OOS R-squared, de-meaned by date.

    This follows the finance-ML convention used by Han et al. (2021) and
    Bali et al. (2023): remove the daily cross-sectional mean from both
    realised and predicted returns, then compute the reduction in squared
    error relative to predicting no cross-sectional spread.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(np.asarray(dates)),
        "y": np.asarray(y_true, dtype=np.float64),
        "pred": np.asarray(y_pred, dtype=np.float64),
    }).dropna()
    if df.empty:
        return np.nan

    df["y_dm"] = df["y"] - df.groupby("date")["y"].transform("mean")
    df["pred_dm"] = df["pred"] - df.groupby("date")["pred"].transform("mean")

    ss_res = np.sum((df["y_dm"] - df["pred_dm"]) ** 2)
    ss_tot = np.sum(df["y_dm"] ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def rmse(y_true, y_pred) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    return np.mean(np.abs(y_true - y_pred))


def information_coefficient(y_true, y_pred) -> float:
    """Spearman rank correlation (IC)."""
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def ic_ir(y_true, y_pred, window: int = 63) -> float:
    """Information Coefficient Information Ratio over rolling windows."""
    n = len(y_true)
    if n < window:
        return np.nan
    ics = []
    for i in range(0, n - window + 1, window):
        yt = y_true[i: i + window]
        yp = y_pred[i: i + window]
        ic, _ = spearmanr(yt, yp)
        if not np.isnan(ic):
            ics.append(ic)
    if len(ics) < 2:
        return np.nan
    return np.mean(ics) / np.std(ics) if np.std(ics) > 0 else np.nan


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute accuracy, AUC, precision, recall, F1, confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

    if y_proba is not None:
        try:
            result["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            result["auc"] = np.nan

    # Long-only accuracy: accuracy when model predicts "up"
    up_mask = y_pred == 1
    if up_mask.sum() > 0:
        result["long_only_accuracy"] = accuracy_score(y_true[up_mask], y_pred[up_mask])
        result["n_long_predictions"] = int(up_mask.sum())
    else:
        result["long_only_accuracy"] = np.nan
        result["n_long_predictions"] = 0

    return result


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------

def economic_metrics(
    y_pred_dir: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost_bps: float = 5,
    initial_capital: float = 10_000,
    dates: np.ndarray | None = None,
    horizon: int = 1,
) -> dict:
    """Simulate a long-only strategy and compute economic metrics.

    Go long when predicted up, flat otherwise. No shorting.

    When `dates` are supplied, the panel is treated as a cross-section and
    collapsed to an equal-weighted daily portfolio before compounding.
    This is the intended behaviour for multi-ticker panel data — otherwise
    the function would compound tens of thousands of per-row returns in
    a single serial chain and overflow.
    """
    tc = transaction_cost_bps / 10_000  # convert bps to fraction

    y_pred_dir = np.asarray(y_pred_dir)
    actual_returns = np.asarray(actual_returns, dtype=np.float64)

    position = y_pred_dir.astype(float)
    per_row_ret = position * actual_returns  # before costs

    if dates is not None and len(dates) == len(actual_returns):
        dates_arr = pd.to_datetime(np.asarray(dates))
        df = pd.DataFrame({"date": dates_arr, "pos": position, "pnl": per_row_ret})
        grouped = df.groupby("date", sort=True)
        active = grouped["pos"].sum()
        # Equal-weighted across active positions on each date; zero when no
        # stock is held.  Standard long/flat cross-sectional portfolio.
        daily_ret = grouped["pnl"].sum() / active.replace(0, np.nan)
        daily_ret = daily_ret.fillna(0.0)
        # Cost proxy: turnover ~ |change in active count| / avg active count.
        avg_active = max(active.mean(), 1.0)
        turnover = active.diff().abs().fillna(active.iloc[0]) / avg_active
        tc_daily = (turnover.fillna(0.0) * tc).values

        strategy_returns = daily_ret.values - tc_daily
        periods_per_year = 252 / max(horizon, 1)
    else:
        trades = np.abs(np.diff(position, prepend=0))
        strategy_returns = per_row_ret - trades * tc
        periods_per_year = 252 / max(horizon, 1)

    strategy_returns = np.clip(strategy_returns, -0.99, None)
    equity_curve = initial_capital * np.cumprod(1 + strategy_returns)

    if strategy_returns.std() > 0:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan

    active_returns = strategy_returns[strategy_returns != 0]
    if len(active_returns) > 0:
        win_rate = float((active_returns > 0).mean())
        avg_win = active_returns[active_returns > 0].mean() if (active_returns > 0).any() else 0.0
        avg_loss = active_returns[active_returns < 0].mean() if (active_returns < 0).any() else 0.0
        win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else np.inf
    else:
        win_rate = np.nan
        win_loss_ratio = np.nan

    n_trades = int(np.abs(np.diff(position, prepend=0)).sum())

    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "total_return": float((equity_curve[-1] / initial_capital - 1) * 100),
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "n_trades": n_trades,
        "final_equity": float(equity_curve[-1]),
        "n_periods": int(len(strategy_returns)),
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_test_ret: np.ndarray,
    y_test_dir: np.ndarray,
    pred_ret: np.ndarray,
    pred_dir: np.ndarray,
    pred_proba: np.ndarray | None,
    y_train_mean: float,
    transaction_cost_bps: float = 5,
    dates: np.ndarray | None = None,
    horizon: int = 1,
) -> dict:
    """Full evaluation of one model. Returns a flat dict of metrics.

    Pass `dates` to compute a cross-sectional equal-weighted daily portfolio
    (required when the rows are a ticker-day panel rather than a single
    time series).
    """
    result = {"model": model_name}

    result["oos_r2"] = oos_r2(y_test_ret, pred_ret, y_train_mean)
    result["rmse"] = rmse(y_test_ret, pred_ret)
    result["mae"] = mae(y_test_ret, pred_ret)
    result["ic"] = information_coefficient(y_test_ret, pred_ret)
    result["ic_ir"] = ic_ir(y_test_ret, pred_ret)
    if dates is not None:
        result["xs_oos_r2"] = cross_sectional_oos_r2(y_test_ret, pred_ret, dates)

    clf = classification_metrics(y_test_dir, pred_dir, pred_proba)
    for k, v in clf.items():
        if k != "confusion_matrix":
            result[k] = v

    econ = economic_metrics(
        pred_dir, y_test_ret, transaction_cost_bps,
        dates=dates, horizon=horizon,
    )
    result.update(econ)

    return result


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build a model comparison DataFrame from a list of result dicts."""
    df = pd.DataFrame(results)
    col_order = [
        "model", "oos_r2", "rmse", "mae", "ic", "ic_ir",
        "accuracy", "auc", "precision", "recall", "f1",
        "long_only_accuracy", "n_long_predictions",
        "sharpe_ratio", "max_drawdown", "total_return",
        "win_rate", "win_loss_ratio", "n_trades",
    ]
    cols = [c for c in col_order if c in df.columns]
    return df[cols]
