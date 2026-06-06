"""Professional-grade extensions requested for the final research version.

This script focuses on the SVI/TFT research path:

1. Train/evaluate TFT on feature set D for 1-, 3-, and 5-day horizons.
2. Compare against an OLS linear baseline and run Diebold-Mariano tests.
3. Compute pooled IC, regular OOS R², and cross-sectional OOS R².
4. Backtest long/short decile and long/flat hurdle strategies with:
   - 15% and 25% effective-spread assumptions
   - 50 bps annual short-borrow fee for the short leg
5. Produce a SHAP/gradient-saliency-style feature importance chart for the
   3-day TFT SVI model.

Run after Stage 4 has refreshed split buffers:

    python3 run_pipeline.py --stages 4
    python3 scripts/run_professional_extensions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_pipeline import _resolve_feature_ids, _split_arrays
from src.evaluation.metrics import evaluate_model, cross_sectional_oos_r2
from src.evaluation.statistical_tests import diebold_mariano_test
from src.models.tft_model import (
    MultiTickerTimeSeriesDataset,
    run_tft_experiment,
)
from src.trading.backtest import (
    backtest_long_short_deciles,
    backtest_long_flat_hurdle,
)
from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import FEATURES_DIR, SPLITS_DIR, RESULTS_DIR, load_parquet
from src.utils.logger import setup_logger

log = setup_logger("professional_extensions")


def _load_panel():
    panel_norm = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_normalized.parquet")
    panel_raw = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel_norm, panel_raw, splits):
        df["date"] = pd.to_datetime(df["date"])
    panel_norm = panel_norm.merge(splits, on=["ticker", "date"], how="inner")
    panel_raw = panel_raw.merge(splits, on=["ticker", "date"], how="inner")
    panel_norm = panel_norm.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel_raw = panel_raw.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel_norm, panel_raw


def _align_tft_truth(d_tft: dict, horizon: int, lookback: int):
    ds = MultiTickerTimeSeriesDataset(
        d_tft["X_test"], d_tft["y_test_ret"], d_tft["tickers_test"], lookback,
    )
    idx = np.array(ds.global_indices)
    y_test = np.array([seq[1].item() for seq in ds.sequences], dtype=np.float64)
    y_dir = (y_test > 0).astype(int)
    aligned = {
        "y_test": y_test,
        "y_dir": y_dir,
        "dates": d_tft["dates_test"][idx],
        "tickers": d_tft["tickers_test"][idx],
        "test_features": d_tft["test_features"].iloc[idx].reset_index(drop=True),
        "global_indices": idx,
    }
    return aligned


def _ols_aligned_predictions(d_tft: dict, aligned: dict):
    """Fast linear baseline prediction using ridge-stabilised least squares."""
    X = np.asarray(d_tft["X_train"], dtype=np.float64)
    y = np.asarray(d_tft["y_train_ret"], dtype=np.float64)
    X_te = np.asarray(d_tft["X_test"][aligned["global_indices"]], dtype=np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    X_aug = np.column_stack([np.ones(len(X)), X])
    Xte_aug = np.column_stack([np.ones(len(X_te)), X_te])

    # Small ridge penalty avoids singular matrices and is much faster/more
    # predictable than calling a multi-threaded sklearn solver here.
    alpha = 1e-6
    xtx = X_aug.T @ X_aug
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xtx + penalty, X_aug.T @ y)
    return Xte_aug @ coef


def _strategy_runs(
    horizon: int,
    pred_ret: np.ndarray,
    aligned: dict,
) -> list[dict]:
    """Run decile and hurdle strategies at 15% and 25% effective-spread costs."""
    rows = []
    # The strategy trades the *underlying stocks*, not option contracts.
    # The current stock panel does not contain CRSP/TAQ bid/ask quotes, so
    # we use the documented stock fallback in ``effective_spread_cost``:
    # a 5 bps quoted spread, of which the strategy pays 15% or 25%.
    #
    # Do not use feat_18 here: that is an options-chain spread proxy and is
    # appropriate for an option-trading strategy, not a stock long/short book.
    spread = None

    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    for eff in [0.15, 0.25]:
        ls_daily, ls_sum = backtest_long_short_deciles(
            aligned["dates"], aligned["tickers"], aligned["y_test"], pred_ret,
            quoted_spread=spread, top_n=6, bottom_n=6,
            effective_spread_fraction=eff,
            short_fee_bps_annual=50.0, horizon=horizon,
        )
        if not ls_daily.empty:
            name = f"tft_svi_long_short_{horizon}d_eff{int(eff*100)}"
            ls_daily.to_csv(tables / f"{name}.csv", index=False)
            ls_sum.update({"horizon": horizon, "cost_scheme": f"{int(eff*100)}pct_spread"})
            ls_sum["cost_source"] = "equity_fallback_5bps_quoted_spread"
            rows.append(ls_sum)

        lf_daily, lf_sum = backtest_long_flat_hurdle(
            aligned["dates"], aligned["tickers"], aligned["y_test"], pred_ret,
            quoted_spread=spread, hurdle_bps=10.0,
            effective_spread_fraction=eff, horizon=horizon,
        )
        if not lf_daily.empty:
            name = f"tft_svi_long_flat_hurdle_{horizon}d_eff{int(eff*100)}"
            lf_daily.to_csv(tables / f"{name}.csv", index=False)
            lf_sum.update({"horizon": horizon, "cost_scheme": f"{int(eff*100)}pct_spread"})
            lf_sum["cost_source"] = "equity_fallback_5bps_quoted_spread"
            rows.append(lf_sum)
    return rows


def _compute_tft_importance(model, d_tft: dict, feature_names: list[str], lookback: int):
    """Compute SHAP if available; otherwise gradient saliency fallback."""
    train_ds = MultiTickerTimeSeriesDataset(
        d_tft["X_train"], d_tft["y_train_ret"], d_tft["tickers_train"], lookback,
    )
    test_ds = MultiTickerTimeSeriesDataset(
        d_tft["X_test"], d_tft["y_test_ret"], d_tft["tickers_test"], lookback,
    )
    n_bg = min(128, len(train_ds))
    n_eval = min(256, len(test_ds))
    bg_idx = np.linspace(0, len(train_ds) - 1, n_bg).astype(int)
    ev_idx = np.linspace(0, len(test_ds) - 1, n_eval).astype(int)
    background = torch.stack([train_ds[i][0] for i in bg_idx]).float()
    samples = torch.stack([test_ds[i][0] for i in ev_idx]).float()

    class PredOnly(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            pred, _ = self.base(x)
            return pred

    wrapped = PredOnly(model.cpu()).eval()
    method = "gradient_saliency"
    try:
        import shap  # type: ignore

        explainer = shap.GradientExplainer(wrapped, background)
        shap_values = explainer.shap_values(samples)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        importance = np.abs(values).mean(axis=(0, 1))
        method = "shap_gradient"
    except Exception as exc:
        log.warning("SHAP failed (%s); falling back to gradient saliency", exc)
        x = samples.clone().requires_grad_(True)
        pred = wrapped(x)
        pred.sum().backward()
        importance = (x.grad.abs() * x.abs()).mean(dim=(0, 1)).detach().numpy()

    out = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "method": method,
    }).sort_values("importance", ascending=False)

    tables = RESULTS_DIR / "tables"
    figures = RESULTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables / "tft_svi_feature_importance.csv", index=False)

    top = out.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"], color="#1f77b4", alpha=0.85)
    ax.set_title(f"TFT/SVI feature importance ({method}, 3-day horizon)")
    ax.set_xlabel("Mean absolute attribution")
    fig.tight_layout()
    fig.savefig(figures / "tft_svi_shap_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_outputs(horizon_df: pd.DataFrame, strategy_df: pd.DataFrame):
    figures = RESULTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col, title, zero in [
        (axes[0], "oos_r2", "OOS R²", 0.0),
        (axes[1], "xs_oos_r2", "Cross-sectional OOS R²", 0.0),
        (axes[2], "ic", "Information Coefficient", 0.0),
    ]:
        ax.plot(horizon_df["horizon"], horizon_df[col], "-o", color="#1f77b4")
        ax.axhline(zero, color="black", lw=0.7)
        ax.set_xticks(horizon_df["horizon"])
        ax.set_title(title)
        ax.set_xlabel("Prediction horizon (trading days)")
    fig.suptitle("TFT on SVI surface: signal decay across horizons", y=1.03)
    fig.tight_layout()
    fig.savefig(figures / "tft_svi_signal_decay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not strategy_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sub = strategy_df.copy()
        sub["label"] = (
            sub["strategy"].str.replace("_", " ")
            + " | " + sub["horizon"].astype(str) + "d | "
            + sub["cost_scheme"].astype(str)
        )
        ax.bar(sub["label"], sub["sharpe"], color="#2ca02c", alpha=0.85)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel("Annualized Sharpe")
        ax.set_title("TFT/SVI strategy performance after frictions")
        ax.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        fig.savefig(figures / "tft_svi_strategy_costs.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    cfg = load_config()
    feat_defs = load_feature_defs()
    panel_norm, panel_raw = _load_panel()
    feat_ids = _resolve_feature_ids(feat_defs, "D", panel_norm.columns.tolist())
    lookback = vars(cfg.models.tft).get("lookback_window", 20)
    params = vars(cfg.models.tft).copy()
    # Keep the extension feasible on a laptop. Stage-5 already trained the
    # full 100-epoch setting; this focused robustness run uses early stopping
    # aggressively and is still out-of-sample.
    params["max_epochs"] = min(params.get("max_epochs", 100), 50)
    params["patience"] = min(params.get("patience", 15), 8)

    horizon_rows = []
    strategy_rows = []
    dm_rows = []
    model_for_importance = None
    d_for_importance = None

    for horizon in [1, 3, 5]:
        ret_col = f"ret_{horizon}d"
        dir_col = f"dir_{horizon}d"
        p_norm = panel_norm.dropna(subset=[ret_col, dir_col]).copy()

        d_tft = _split_arrays(p_norm, feat_ids, ret_col, dir_col)
        log.info("Training TFT/SVI horizon %dd on %d features", horizon, len(feat_ids))
        result = run_tft_experiment(
            d_tft["X_train"], d_tft["y_train_ret"],
            d_tft["X_val"], d_tft["y_val_ret"],
            d_tft["X_test"], d_tft["y_test_ret"],
            params, "D_professional", horizon,
            tickers_train=d_tft["tickers_train"],
            tickers_val=d_tft["tickers_val"],
            tickers_test=d_tft["tickers_test"],
        )
        log.info("Finished TFT/SVI horizon %dd training and prediction", horizon)
        aligned = _align_tft_truth(d_tft, horizon, lookback)
        pred_ret = result["predictions"]
        pred_dir = (pred_ret > 0).astype(int)
        metrics = evaluate_model(
            f"TFT_D_{horizon}d", aligned["y_test"], aligned["y_dir"],
            pred_ret, pred_dir, None,
            d_tft["y_train_ret"].mean(), cfg.evaluation.transaction_cost_bps,
            dates=aligned["dates"], horizon=horizon,
        )
        metrics["horizon"] = horizon
        metrics["xs_oos_r2"] = cross_sectional_oos_r2(
            aligned["y_test"], pred_ret, aligned["dates"],
        )
        horizon_rows.append(metrics)
        log.info(
            "Horizon %dd metrics: R2=%.4f XS-R2=%.4f IC=%.4f Acc=%.4f",
            horizon, metrics["oos_r2"], metrics["xs_oos_r2"],
            metrics["ic"], metrics["accuracy"],
        )

        log.info("Fitting fast linear baseline for horizon %dd", horizon)
        ols_pred = _ols_aligned_predictions(d_tft, aligned)
        dm = diebold_mariano_test(aligned["y_test"], ols_pred, pred_ret, horizon=horizon)
        dm.update({"horizon": horizon, "test": "DM OLS vs TFT_D", "model_a": "OLS", "model_b": "TFT_D"})
        dm_rows.append(dm)

        log.info("Running long/short and hurdle backtests for horizon %dd", horizon)
        strategy_rows.extend(_strategy_runs(horizon, pred_ret, aligned))
        log.info("Finished strategy backtests for horizon %dd", horizon)

        if horizon == 3:
            model_for_importance = result["model"]
            d_for_importance = d_tft

    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    horizon_df = pd.DataFrame(horizon_rows)
    strategy_df = pd.DataFrame(strategy_rows)
    dm_df = pd.DataFrame(dm_rows)
    horizon_df.to_csv(tables / "tft_svi_horizon_comparison.csv", index=False)
    strategy_df.to_csv(tables / "tft_svi_strategy_summary.csv", index=False)
    dm_df.to_csv(tables / "tft_svi_dm_vs_ols.csv", index=False)
    _plot_outputs(horizon_df, strategy_df)

    if model_for_importance is not None and d_for_importance is not None:
        _compute_tft_importance(model_for_importance, d_for_importance, feat_ids, lookback)

    log.info("Professional extensions complete.")
    print("Saved:")
    print(f"  {tables / 'tft_svi_horizon_comparison.csv'}")
    print(f"  {tables / 'tft_svi_strategy_summary.csv'}")
    print(f"  {tables / 'tft_svi_dm_vs_ols.csv'}")
    print(f"  {RESULTS_DIR / 'figures' / 'tft_svi_signal_decay.png'}")
    print(f"  {RESULTS_DIR / 'figures' / 'tft_svi_strategy_costs.png'}")
    print(f"  {RESULTS_DIR / 'figures' / 'tft_svi_shap_importance.png'}")


if __name__ == "__main__":
    main()
