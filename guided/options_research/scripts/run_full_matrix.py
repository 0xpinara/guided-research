"""Run full matrix walk-forward + long/short across models and sets.

This script fills the gap left by Stage 6 (which currently runs walk-forward
only for XGBoost). It supports:

  - Models: XGBoost, FFNN, TabNet, TFT
  - Feature sets: A, B, C, candidate_6, D
  - Schemes: expanding and rolling_9m_3m
  - Strategy summaries on concatenated OOS predictions:
      * long_short_decile
      * long_flat_hurdle

Outputs are written to results/tables/:
  - full_matrix_walkforward_windows.csv
  - full_matrix_walkforward_pooled.csv
  - full_matrix_strategy_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep thread pressure low to avoid native-library crashes on some local setups.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from run_pipeline import _resolve_feature_ids, _add_ticker_encoding
from src.evaluation.metrics import evaluate_model, oos_r2
from src.models.ffnn_model import train_ffnn, predict_ffnn
from src.models.tft_model import (
    train_tft,
    predict_tft,
    MultiTickerTimeSeriesDataset,
)
from src.trading.backtest import backtest_long_short_deciles, backtest_long_flat_hurdle
from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR
from src.utils.logger import setup_logger

log = setup_logger("full_matrix")


VALID_MODELS = ("OLS", "ElasticNet", "XGBoost", "FFNN", "TabNet", "TFT")
VALID_SCHEMES = ("expanding", "rolling_9m_3m")
_LINEAR_MODELS = ("OLS", "ElasticNet")

# Module-level accumulators persisted for post-hoc analysis (no retrain needed):
#   - OOS predictions for XGBoost cells -> regime slicing / signal decay
#   - long/short daily P&L for every cell -> PBO / deflated Sharpe
_OOS_PRED_FRAMES: list[pd.DataFrame] = []
_LS_DAILY_FRAMES: list[pd.DataFrame] = []


def _cross_sectional_demean(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Subtract each date's cross-sectional mean from every observation.

    This is contemporaneous (same-day cross-section only, no look-ahead) and is
    the standard target transform for cross-sectional return prediction
    (Gu-Kelly-Xiu): the model then learns the relative spread the long/short
    decile strategy actually trades, and the train mean collapses to ~0 so the
    pooled R^2 is measured against a zero forecast."""
    s = pd.DataFrame({"d": dates, "v": np.asarray(values, dtype=np.float64)})
    return (s["v"] - s.groupby("d")["v"].transform("mean")).to_numpy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full matrix walk-forward runner")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(VALID_MODELS),
        help=f"Subset of models to run (default: all). Choices: {VALID_MODELS}",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["A", "B", "C", "candidate_6", "D"],
        help="Feature sets to run",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=list(VALID_SCHEMES),
        help=f"Walk-forward schemes (default: all). Choices: {VALID_SCHEMES}",
    )
    parser.add_argument(
        "--expanding-min-train",
        type=int,
        default=504,
        help="Expanding: minimum training days",
    )
    parser.add_argument(
        "--expanding-step",
        type=int,
        default=63,
        help="Expanding: test block size and step (days)",
    )
    parser.add_argument(
        "--rolling-train",
        type=int,
        default=189,
        help="Rolling: train window length (days)",
    )
    parser.add_argument(
        "--rolling-test",
        type=int,
        default=63,
        help="Rolling: test block length (days)",
    )
    parser.add_argument(
        "--rolling-step",
        type=int,
        default=63,
        help="Rolling: slide step (days)",
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=0,
        help="Debug mode: cap number of windows per run (0 = no cap)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Target horizons (trading days) to run. Each gets its own "
             "non-overlapping rebalance cadence (rebalance every h days).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing full_matrix_*.csv and skip cells already computed.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    bad_models = [m for m in args.models if m not in VALID_MODELS]
    bad_schemes = [s for s in args.schemes if s not in VALID_SCHEMES]
    if bad_models:
        raise ValueError(f"Unknown models: {bad_models}")
    if bad_schemes:
        raise ValueError(f"Unknown schemes: {bad_schemes}")


def _train_val_masks(train_mask: np.ndarray, dates: np.ndarray, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal train/val masks *within* a window's train block."""
    train_dates = np.sort(np.unique(dates[train_mask]))
    if len(train_dates) < 20:
        return train_mask, np.zeros_like(train_mask, dtype=bool)
    split_idx = max(1, int(len(train_dates) * (1 - val_frac)))
    tr_dates = set(train_dates[:split_idx])
    va_dates = set(train_dates[split_idx:])
    return np.isin(dates, list(tr_dates)), np.isin(dates, list(va_dates))


def _predict_window(
    model_name: str,
    params: dict,
    X: np.ndarray,
    y_ret: np.ndarray,
    y_raw: np.ndarray,
    tickers: np.ndarray,
    dates: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one model on one window and return aligned (pred, demeaned-truth,
    dates, tickers, raw-truth). The model trains on the cross-sectionally
    demeaned target ``y_ret``; ``y_raw`` carries the raw realised return used by
    the strategy P&L. All five arrays share length and ordering."""
    tr_mask, va_mask = _train_val_masks(train_mask, dates)
    if va_mask.sum() == 0:
        va_mask = tr_mask.copy()

    X_tr, y_tr = X[tr_mask], y_ret[tr_mask]
    X_va, y_va = X[va_mask], y_ret[va_mask]
    X_te, y_te = X[test_mask], y_ret[test_mask]
    raw_te = y_raw[test_mask]
    tk_tr, tk_va, tk_te = tickers[tr_mask], tickers[va_mask], tickers[test_mask]

    if model_name == "OLS":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return pred, y_te, dates[test_mask], tk_te, raw_te

    if model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # Lean regularized-linear reference: a small fixed alpha grid, one
        # l1_ratio, loose tol. precompute=False avoids the float32 Gram-matrix
        # failure; StandardScaler is fit on train only (no leakage).
        model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=0.5, alphas=[1e-4, 1e-3, 1e-2, 1e-1], cv=3,
                         precompute=False, random_state=42, n_jobs=1,
                         max_iter=1000, tol=1e-3),
        ).fit(np.asarray(X_tr, dtype=np.float64), y_tr)
        pred = model.predict(np.asarray(X_te, dtype=np.float64))
        return pred, y_te, dates[test_mask], tk_te, raw_te

    if model_name == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.01),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            eval_metric=params.get("eval_metric_reg", "rmse"),
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_te)
        return pred, y_te, dates[test_mask], tk_te, raw_te

    if model_name == "FFNN":
        model, _ = train_ffnn(X_tr, y_tr, X_va, y_va, params, task="regression")
        pred = predict_ffnn(model, X_te, task="regression")
        return pred, y_te, dates[test_mask], tk_te, raw_te

    if model_name == "TabNet":
        try:
            from src.models.tabnet_model import train_tabnet_regressor
        except ImportError as e:
            raise RuntimeError(f"TabNet unavailable: {e}") from e
        model = train_tabnet_regressor(X_tr, y_tr, X_va, y_va, params)
        pred = model.predict(X_te).ravel()
        return pred, y_te, dates[test_mask], tk_te, raw_te

    if model_name == "TFT":
        model, _ = train_tft(
            X_tr, y_tr, X_va, y_va, params, task="regression",
            tickers_train=tk_tr, tickers_val=tk_va,
        )
        pred, _ = predict_tft(
            model, X_te, tickers=tk_te,
            lookback=params.get("lookback_window", 20),
        )
        lookback = params.get("lookback_window", 20)
        ds_te = MultiTickerTimeSeriesDataset(X_te, y_te, tk_te, lookback)
        gidx = np.array(ds_te.global_indices)
        y_aligned = np.array([seq[1].item() for seq in ds_te.sequences], dtype=np.float64)
        dt_aligned = dates[test_mask][gidx]
        # Carry the per-window ticker and raw return for each retained sequence
        # so the cross-sectional decile strategy can rank and price by name.
        tk_aligned = tk_te[gidx]
        raw_aligned = raw_te[gidx]
        return pred, y_aligned, dt_aligned, tk_aligned, raw_aligned

    raise ValueError(f"Unknown model: {model_name}")


def _window_ranges(
    scheme: str,
    unique_dates: np.ndarray,
    expanding_min_train: int,
    expanding_step: int,
    rolling_train: int,
    rolling_test: int,
    rolling_step: int,
) -> list[tuple[int, int, int, int]]:
    """Return list of (train_start, train_end, test_start, test_end) indices."""
    n_dates = len(unique_dates)
    windows = []

    if scheme == "expanding":
        for train_end in range(expanding_min_train, n_dates - expanding_step, expanding_step):
            test_start = train_end
            test_end = min(train_end + expanding_step, n_dates)
            windows.append((0, train_end, test_start, test_end))
        return windows

    if scheme == "rolling_9m_3m":
        test_start = rolling_train
        while test_start + rolling_test <= n_dates:
            windows.append((test_start - rolling_train, test_start, test_start, test_start + rolling_test))
            test_start += rolling_step
        return windows

    raise ValueError(f"Unknown scheme: {scheme}")


def _pooled_metrics(pred: np.ndarray, y_true: np.ndarray, y_train_mean: float) -> dict:
    pred_dir = (pred > 0).astype(int)
    y_dir = (y_true > 0).astype(int)
    ic = np.nan
    if len(pred) > 10:
        ic = float(pd.Series(pred).corr(pd.Series(y_true), method="spearman"))
    return {
        "pooled_r2": float(oos_r2(y_true, pred, y_train_mean)),
        "pooled_rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
        "pooled_accuracy": float((pred_dir == y_dir).mean()),
        "pooled_ic": ic,
    }


def _flush_outputs(window_rows: list[dict], pooled_rows: list[dict], strategy_rows: list[dict]) -> None:
    """Persist intermediate progress so long runs can resume safely."""
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if window_rows:
        pd.DataFrame(window_rows).to_csv(out_dir / "full_matrix_walkforward_windows.csv", index=False)
    if pooled_rows:
        pd.DataFrame(pooled_rows).to_csv(out_dir / "full_matrix_walkforward_pooled.csv", index=False)
    if strategy_rows:
        pd.DataFrame(strategy_rows).to_csv(out_dir / "full_matrix_strategy_summary.csv", index=False)


def _cell_key(row: dict) -> tuple:
    """Identity of one walk-forward cell: (model, feature_set, scheme, horizon)."""
    return (
        row.get("model"),
        row.get("feature_set"),
        row.get("scheme"),
        int(row["horizon"]) if row.get("horizon") is not None else None,
    )


def _load_existing() -> tuple[list[dict], list[dict], list[dict], set]:
    """Reload prior results for --resume. Only horizon-aware files are reused
    so legacy single-horizon outputs cannot contaminate a multi-horizon run."""
    out_dir = RESULTS_DIR / "tables"
    window_rows: list[dict] = []
    pooled_rows: list[dict] = []
    strategy_rows: list[dict] = []
    done: set = set()
    for name, bucket in (
        ("full_matrix_walkforward_windows.csv", "win"),
        ("full_matrix_walkforward_pooled.csv", "pool"),
        ("full_matrix_strategy_summary.csv", "strat"),
    ):
        path = out_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "horizon" not in df.columns:
            log.warning("Ignoring legacy %s (no horizon column) for resume", name)
            continue
        recs = df.to_dict("records")
        if bucket == "win":
            window_rows = recs
        elif bucket == "pool":
            pooled_rows = recs
            done = {_cell_key(r) for r in recs}
        else:
            strategy_rows = recs
    if done:
        log.info("Resume: %d completed cells will be skipped", len(done))
    return window_rows, pooled_rows, strategy_rows, done


def _run_horizon(
    args: argparse.Namespace,
    cfg,
    feat_defs,
    tc_bps: float,
    panel: pd.DataFrame,
    panel_norm: pd.DataFrame,
    ticker_cols: list[str],
    horizon: int,
    window_rows: list[dict],
    pooled_rows: list[dict],
    strategy_rows: list[dict],
    done: set,
) -> None:
    """Run the full model x set x scheme grid for a single target horizon.

    Positions rebalance every ``horizon`` trading days, so the realised
    h-day-forward returns used by the strategies do not overlap.
    """
    ret_col = f"ret_{horizon}d"
    dir_col = f"dir_{horizon}d"
    panel_h = panel.dropna(subset=[ret_col, dir_col]).copy()
    panel_norm_h = panel_norm.dropna(subset=[ret_col, dir_col]).copy()

    for fs_name in args.sets:
        feat_ids = _resolve_feature_ids(feat_defs, fs_name, panel_h.columns.tolist())
        if not feat_ids:
            log.warning("Skipping %s (no resolved features)", fs_name)
            continue

        feat_with_ticker = feat_ids + ticker_cols
        flat_dates = panel_h["date"].to_numpy()
        tft_dates = panel_norm_h["date"].to_numpy()
        flat_raw = panel_h[ret_col].to_numpy(dtype=np.float64, na_value=np.nan)
        tft_raw = panel_norm_h[ret_col].to_numpy(dtype=np.float64, na_value=np.nan)
        set_data = {
            "flat": {
                "X": panel_h.loc[:, feat_with_ticker].to_numpy(dtype=np.float32, na_value=np.nan),
                # Training target = cross-sectionally demeaned return (GKX).
                "y_ret": _cross_sectional_demean(flat_raw, flat_dates),
                # Raw realised return for strategy P&L.
                "y_raw": flat_raw,
                "dates": flat_dates,
                "tickers": panel_h["ticker"].to_numpy(),
            },
            "tft": {
                "X": panel_norm_h.loc[:, feat_ids].to_numpy(dtype=np.float32, na_value=np.nan),
                "y_ret": _cross_sectional_demean(tft_raw, tft_dates),
                "y_raw": tft_raw,
                "dates": tft_dates,
                "tickers": panel_norm_h["ticker"].to_numpy(),
            },
        }
        for key in ("flat", "tft"):
            np.nan_to_num(set_data[key]["X"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(set_data[key]["y_ret"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(set_data[key]["y_raw"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        for model_name in args.models:
            if model_name in _LINEAR_MODELS:
                model_params = {}
            else:
                model_params = vars(getattr(cfg.models, model_name.lower() if model_name != "XGBoost" else "xgboost"))
            key = "tft" if model_name == "TFT" else "flat"
            X_all = set_data[key]["X"]
            y_ret_all = set_data[key]["y_ret"]
            y_raw_all = set_data[key]["y_raw"]
            dates_all = set_data[key]["dates"]
            tickers_all = set_data[key]["tickers"]
            unique_dates = np.sort(np.unique(dates_all))

            for scheme in args.schemes:
                cell = (model_name, fs_name, scheme, int(horizon))
                if cell in done:
                    log.info("Skip (resume) %s", cell)
                    continue

                windows = _window_ranges(
                    scheme, unique_dates,
                    args.expanding_min_train, args.expanding_step,
                    args.rolling_train, args.rolling_test, args.rolling_step,
                )
                if args.limit_windows > 0:
                    windows = windows[: args.limit_windows]
                if not windows:
                    log.warning("No windows for %s/%s/%s", model_name, fs_name, scheme)
                    continue

                log.info("Running %s | set=%s | scheme=%s | h=%dd | windows=%d",
                         model_name, fs_name, scheme, horizon, len(windows))

                pooled_pred, pooled_true = [], []
                pooled_dates, pooled_tickers, pooled_raw = [], [], []
                train_means = []

                for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
                    # Embargo the last `horizon` training dates: their h-day
                    # forward labels are computed from prices that fall inside
                    # the test block (test_start == train_end), so keeping them
                    # leaks test-period returns into training. Purging them is
                    # the standard purged/embargoed walk-forward (Lopez de Prado).
                    train_block = unique_dates[tr_s:tr_e]
                    if horizon > 0 and len(train_block) > horizon:
                        train_block = train_block[:-horizon]
                    train_dates = set(train_block)
                    test_dates = set(unique_dates[te_s:te_e])
                    train_mask = np.isin(dates_all, list(train_dates))
                    test_mask = np.isin(dates_all, list(test_dates))
                    if train_mask.sum() == 0 or test_mask.sum() == 0:
                        continue
                    try:
                        pred, y_true, dt_aligned, tk_aligned, raw_aligned = _predict_window(
                            model_name=model_name,
                            params=model_params,
                            X=X_all,
                            y_ret=y_ret_all,
                            y_raw=y_raw_all,
                            tickers=tickers_all,
                            dates=dates_all,
                            train_mask=train_mask,
                            test_mask=test_mask,
                        )
                    except Exception as e:
                        log.warning(
                            "Window failed (%s/%s/%s/w%d): %s",
                            model_name, fs_name, scheme, w_idx, e,
                        )
                        continue

                    # pred / demeaned-truth / dates / tickers / raw-truth are
                    # returned aligned (1:1, same ordering) for every model.
                    y_dir_aligned = (y_true > 0).astype(int)
                    pred_dir = (pred > 0).astype(int)
                    if not (len(pred) == len(y_true) == len(dt_aligned) == len(tk_aligned) == len(raw_aligned)):
                        log.warning(
                            "Alignment mismatch (%s/%s/%s/w%d): pred=%d y=%d dt=%d tk=%d raw=%d",
                            model_name, fs_name, scheme, w_idx,
                            len(pred), len(y_true), len(dt_aligned), len(tk_aligned), len(raw_aligned),
                        )
                        continue

                    m = evaluate_model(
                        model_name=f"{model_name}_{fs_name}_{scheme}_w{w_idx}",
                        y_test_ret=y_true,
                        y_test_dir=y_dir_aligned,
                        pred_ret=pred,
                        pred_dir=pred_dir,
                        pred_proba=None,
                        y_train_mean=float(y_ret_all[train_mask].mean()),
                        transaction_cost_bps=tc_bps,
                        dates=dt_aligned,
                        horizon=horizon,
                    )
                    m.update({
                        "window": w_idx,
                        "model": model_name,
                        "feature_set": fs_name,
                        "scheme": scheme,
                        "horizon": int(horizon),
                        "train_start": str(unique_dates[tr_s])[:10],
                        "train_end": str(unique_dates[tr_e - 1])[:10],
                        "test_start": str(unique_dates[te_s])[:10],
                        "test_end": str(unique_dates[te_e - 1])[:10],
                        "n_train": int(train_mask.sum()),
                        "n_test": int(len(pred)),
                    })
                    window_rows.append(m)

                    pooled_pred.append(pred)
                    pooled_true.append(y_true)
                    pooled_dates.append(dt_aligned)
                    pooled_tickers.append(tk_aligned)
                    pooled_raw.append(raw_aligned)
                    train_means.append(float(y_ret_all[train_mask].mean()))

                if not pooled_pred:
                    continue

                pred_all = np.concatenate(pooled_pred)
                true_all = np.concatenate(pooled_true)       # cross-sectionally demeaned
                raw_all = np.concatenate(pooled_raw)         # raw realised returns
                dates_concat = np.concatenate(pooled_dates)
                tickers_concat = np.concatenate(pooled_tickers)
                pooled = _pooled_metrics(pred_all, true_all, float(np.mean(train_means)))
                pooled.update({
                    "model": model_name,
                    "feature_set": fs_name,
                    "scheme": scheme,
                    "horizon": int(horizon),
                    "n_windows": len(pooled_pred),
                })
                pooled_rows.append(pooled)

                # Persist OOS predictions for XGBoost (the headline tabular
                # model) so regime slicing and signal decay can be computed
                # post-hoc without retraining.
                if model_name == "XGBoost":
                    _OOS_PRED_FRAMES.append(pd.DataFrame({
                        "model": model_name, "feature_set": fs_name,
                        "scheme": scheme, "horizon": int(horizon),
                        "date": dates_concat, "ticker": tickers_concat,
                        "pred": pred_all, "y_demeaned": true_all, "y_raw": raw_all,
                    }))

                # Strategy layer on concatenated OOS predictions.
                for eff in (0.15, 0.25):
                    ls_daily, ls_summary = backtest_long_short_deciles(
                        dates=dates_concat,
                        tickers=tickers_concat,
                        actual_returns=raw_all,
                        predicted_returns=pred_all,
                        quoted_spread=None,
                        top_n=6,
                        bottom_n=6,
                        effective_spread_fraction=eff,
                        short_fee_bps_annual=50.0,
                        horizon=horizon,
                        rebalance_every=horizon,
                    )
                    if ls_summary:
                        ls_summary.update({
                            "model": model_name,
                            "feature_set": fs_name,
                            "scheme": scheme,
                            "cost_scheme": f"{int(eff * 100)}pct_spread",
                        })
                        strategy_rows.append(ls_summary)
                        # Persist the daily L/S P&L series (15% cost) for PBO.
                        if eff == 0.15 and not ls_daily.empty:
                            d = ls_daily[["date", "portfolio_return"]].copy()
                            d["model"] = model_name; d["feature_set"] = fs_name
                            d["scheme"] = scheme; d["horizon"] = int(horizon)
                            _LS_DAILY_FRAMES.append(d)

                    lf_daily, lf_summary = backtest_long_flat_hurdle(
                        dates=dates_concat,
                        tickers=tickers_concat,
                        actual_returns=raw_all,
                        predicted_returns=pred_all,
                        quoted_spread=None,
                        hurdle_bps=10.0,
                        effective_spread_fraction=eff,
                        horizon=horizon,
                        rebalance_every=horizon,
                    )
                    if lf_summary:
                        lf_summary.update({
                            "model": model_name,
                            "feature_set": fs_name,
                            "scheme": scheme,
                            "cost_scheme": f"{int(eff * 100)}pct_spread",
                        })
                        strategy_rows.append(lf_summary)

                # Save after each model/set/scheme block.
                _flush_outputs(window_rows, pooled_rows, strategy_rows)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    cfg = load_config()
    feat_defs = load_feature_defs()
    tc_bps = cfg.evaluation.transaction_cost_bps

    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    panel_norm = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_normalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel, panel_norm, splits):
        df["date"] = pd.to_datetime(df["date"])

    panel = panel.merge(splits, on=["ticker", "date"], how="inner")
    panel_norm = panel_norm.merge(splits, on=["ticker", "date"], how="inner")

    # Merge the raw-quote interpolated IV grid (model-free, independent of the
    # SVI fit) so repr_grid_raw gives a genuine non-SVI grid representation.
    raw_grid_path = FEATURES_DIR / "resolution_2_surface" / "surface_features_all.parquet"
    if raw_grid_path.exists():
        rg = load_parquet(raw_grid_path)
        rg["date"] = pd.to_datetime(rg["date"])
        rg_cols = [c for c in rg.columns if c.startswith("iv_surf_") or c.startswith("surface_")]
        rg = rg[["ticker", "date"] + rg_cols].drop_duplicates(["ticker", "date"])
        # Merge into both panels so feature-id resolution matches. repr_grid_raw
        # is run only with tabular models (which use the unnormalised panel);
        # the copy in panel_norm is never consumed by the TFT path.
        panel = panel.merge(rg, on=["ticker", "date"], how="left")
        panel_norm = panel_norm.merge(rg, on=["ticker", "date"], how="left")
        log.info("Merged %d raw-grid columns for repr_grid_raw", len(rg_cols))

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel_norm = panel_norm.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_cols = _add_ticker_encoding(panel)
    _add_ticker_encoding(panel_norm)

    if args.resume:
        window_rows, pooled_rows, strategy_rows, done = _load_existing()
    else:
        window_rows, pooled_rows, strategy_rows, done = [], [], [], set()

    log.info(
        "Grid: models=%s sets=%s schemes=%s horizons=%s",
        args.models, args.sets, args.schemes, args.horizons,
    )

    for horizon in args.horizons:
        _run_horizon(
            args, cfg, feat_defs, tc_bps, panel, panel_norm, ticker_cols,
            int(horizon), window_rows, pooled_rows, strategy_rows, done,
        )

    _flush_outputs(window_rows, pooled_rows, strategy_rows)

    out_dir = RESULTS_DIR / "tables"

    def _merge_parquet(frames: list[pd.DataFrame], path: Path, dedup: list[str]) -> None:
        """Append new frames to any existing parquet (dedupe), so chained
        --resume processes accumulate rather than overwrite."""
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)
        if path.exists():
            try:
                new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
            except Exception:
                pass
        new = new.drop_duplicates(subset=dedup, keep="last")
        new.to_parquet(path, index=False)

    _merge_parquet(_OOS_PRED_FRAMES, out_dir / "full_matrix_oos_xgb.parquet",
                   ["model", "feature_set", "scheme", "horizon", "date", "ticker"])
    _merge_parquet(_LS_DAILY_FRAMES, out_dir / "full_matrix_ls_daily.parquet",
                   ["model", "feature_set", "scheme", "horizon", "date"])

    log.info(
        "Done. wrote windows=%d, pooled=%d, strategy=%d rows; oos_frames=%d ls_frames=%d",
        len(window_rows), len(pooled_rows), len(strategy_rows),
        len(_OOS_PRED_FRAMES), len(_LS_DAILY_FRAMES),
    )


if __name__ == "__main__":
    main()
