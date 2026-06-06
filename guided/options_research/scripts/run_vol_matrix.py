"""Volatility-target walk-forward grid (companion to run_full_matrix.py).

Same panel, same feature sets, same embargoed expanding/rolling windows, same
train-only demeaning -- but the target is forward realised volatility instead of
the forward return, and only the CPU models (OLS, ElasticNet, XGBoost) are run.
This produces the returns-vs-volatility dissociation that is the spine of the
paper. No economic backtest (volatility is not traded the same way).

RV target: for each (ticker, date, horizon h),
    RV[t,h] = sqrt( sum_{k=0}^{h-1} ( log(1 + ret_1d[t+k]) )^2 ),
i.e. the realised vol of the *forward* window [t+1, t+h]. It is then
cross-sectionally demeaned per date, identically to the return target, so the
rank-IC is directly comparable across the two tasks.

Outputs (results/tables/):
    vol_walkforward_windows.csv, vol_walkforward_pooled.csv,
    vol_ic_tstats.csv, vol_oos_xgb.parquet
"""
from __future__ import annotations

import os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_pipeline import _resolve_feature_ids, _add_ticker_encoding
from run_full_matrix import (
    _window_ranges, _predict_window, _cross_sectional_demean,
)
from src.evaluation.metrics import oos_r2
from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR
from src.utils.logger import setup_logger

log = setup_logger("vol_matrix")

MODELS = ("OLS", "ElasticNet", "XGBoost")
SETS = ["A", "B", "C", "candidate_6", "D",
        "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]
SCHEMES = ("expanding", "rolling_9m_3m")
HORIZONS = (1, 3, 5)
# window geometry copied verbatim from run_full_matrix defaults
EXP_MIN_TRAIN, EXP_STEP = 504, 63
ROLL_TRAIN, ROLL_TEST, ROLL_STEP = 189, 63, 63
TAB = RESULTS_DIR / "tables"


def _nw_tstat(x: np.ndarray):
    """Mean, Newey-West HAC SE of the mean, t (identical to posthoc_stats)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(np.mean(x)) if n else np.nan, np.nan, np.nan)
    m = float(np.mean(x)); e = x - m
    g0 = float(np.mean(e * e)); L = max(1, int(n ** (1 / 3))); var = g0
    for k in range(1, L + 1):
        var += 2.0 * (1 - k / (L + 1.0)) * float(np.mean(e[k:] * e[:-k]))
    se = np.sqrt(max(var, 0.0) / n)
    return m, se, (m / se if se > 0 else np.nan)


def _build_rv(panel: pd.DataFrame, h: int) -> pd.Series:
    """Forward realised volatility over [t+1, t+h] from forward 1-day returns."""
    lr = np.log1p(panel["ret_1d"].astype(float))
    sq = (lr * lr)
    out = np.full(len(panel), np.nan)
    for _, idx in panel.groupby("ticker").groups.items():
        s = sq.loc[idx]
        # forward rolling sum of h squared daily returns (reverse trick)
        fwd = s[::-1].rolling(h, min_periods=h).sum()[::-1]
        out[panel.index.get_indexer(idx)] = np.sqrt(fwd.to_numpy())
    return pd.Series(out, index=panel.index)


def main():
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass

    cfg = load_config(); feat_defs = load_feature_defs()
    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel, splits):
        df["date"] = pd.to_datetime(df["date"])
    panel = panel.merge(splits, on=["ticker", "date"], how="inner")

    rg_path = FEATURES_DIR / "resolution_2_surface" / "surface_features_all.parquet"
    if rg_path.exists():
        rg = load_parquet(rg_path); rg["date"] = pd.to_datetime(rg["date"])
        cols = [c for c in rg.columns if c.startswith("iv_surf_") or c.startswith("surface_")]
        rg = rg[["ticker", "date"] + cols].drop_duplicates(["ticker", "date"])
        panel = panel.merge(rg, on=["ticker", "date"], how="left")
        log.info("merged %d raw-grid columns", len(cols))

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    ticker_cols = _add_ticker_encoding(panel)

    # resume support
    win_path = TAB / "vol_walkforward_windows.csv"
    window_rows, pooled_rows, done = [], [], set()
    if win_path.exists():
        prev = pd.read_csv(win_path); window_rows = prev.to_dict("records")
        done = {(r["model"], r["feature_set"], r["scheme"], int(r["horizon"]))
                for r in window_rows}
        pp = TAB / "vol_walkforward_pooled.csv"
        if pp.exists():
            pooled_rows = pd.read_csv(pp).to_dict("records")
        log.info("resume: %d cells done", len(done))

    oos_frames = []
    xgmodels = {}

    for h in HORIZONS:
        rv = _build_rv(panel, h)
        ph = panel.assign(rv=rv).dropna(subset=["rv"]).reset_index(drop=True)
        ticker_cols_h = [c for c in ph.columns if c.startswith("ticker_")]
        for fs in SETS:
            feat_ids = _resolve_feature_ids(feat_defs, fs, ph.columns.tolist())
            if not feat_ids:
                log.warning("skip %s (no features)", fs); continue
            cols = feat_ids + ticker_cols_h
            X = ph.loc[:, cols].to_numpy(np.float32, na_value=np.nan)
            dates = ph["date"].to_numpy()
            rv_raw = ph["rv"].to_numpy(np.float64, na_value=np.nan)
            y = _cross_sectional_demean(rv_raw, dates)
            tickers = ph["ticker"].to_numpy()
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            uniq = np.sort(np.unique(dates))
            for model in MODELS:
                params = vars(cfg.models.xgboost) if model == "XGBoost" else {}
                for scheme in SCHEMES:
                    cell = (model, fs, scheme, int(h))
                    if cell in done:
                        continue
                    wins = _window_ranges(scheme, uniq, EXP_MIN_TRAIN, EXP_STEP,
                                          ROLL_TRAIN, ROLL_TEST, ROLL_STEP)
                    if not wins:
                        continue
                    log.info("vol %s | %s | %s | h=%d | %d windows",
                             model, fs, scheme, h, len(wins))
                    p_pred, p_true, p_dt, p_tk = [], [], [], []
                    tmeans = []
                    for wi, (ts, te, vs, ve) in enumerate(wins):
                        tb = uniq[ts:te]
                        if h > 0 and len(tb) > h:
                            tb = tb[:-h]            # embargo last h train dates
                        trm = np.isin(dates, list(set(tb)))
                        tem = np.isin(dates, list(set(uniq[vs:ve])))
                        if trm.sum() == 0 or tem.sum() == 0:
                            continue
                        try:
                            pred, yt, dt, tk, _ = _predict_window(
                                model, params, X, y, rv_raw, tickers, dates, trm, tem)
                        except Exception as e:
                            log.warning("window fail %s w%d: %s", cell, wi, e); continue
                        if len(pred) != len(yt):
                            continue
                        ic = spearmanr(yt, pred)[0] if len(pred) > 10 else np.nan
                        window_rows.append({
                            "model": model, "feature_set": fs, "scheme": scheme,
                            "horizon": int(h), "window": wi, "ic": float(ic),
                            "n_test": int(len(pred))})
                        p_pred.append(pred); p_true.append(yt)
                        p_dt.append(dt); p_tk.append(tk)
                        tmeans.append(float(y[trm].mean()))
                    if not p_pred:
                        continue
                    pa = np.concatenate(p_pred); ta = np.concatenate(p_true)
                    pic = spearmanr(ta, pa)[0]
                    pooled_rows.append({
                        "model": model, "feature_set": fs, "scheme": scheme,
                        "horizon": int(h), "n_windows": len(p_pred),
                        "pooled_ic": float(pic),
                        "pooled_r2": float(oos_r2(ta, pa, float(np.mean(tmeans))))})
                    if model == "XGBoost":
                        oos_frames.append(pd.DataFrame({
                            "model": model, "feature_set": fs, "scheme": scheme,
                            "horizon": int(h),
                            "date": np.concatenate(p_dt),
                            "ticker": np.concatenate(p_tk),
                            "pred": pa, "y_demeaned": ta}))
                    pd.DataFrame(window_rows).to_csv(win_path, index=False)
                    pd.DataFrame(pooled_rows).to_csv(TAB / "vol_walkforward_pooled.csv", index=False)

    # HAC t-stats per cell
    wdf = pd.DataFrame(window_rows)
    rows = []
    for key, g in wdf.groupby(["model", "feature_set", "scheme", "horizon"]):
        m, se, t = _nw_tstat(g["ic"].to_numpy())
        rows.append(dict(zip(["model", "feature_set", "scheme", "horizon"], key)) |
                    {"n_windows": len(g), "mean_ic": m, "nw_se": se, "nw_tstat": t,
                     "survives_5pct": bool(abs(t) > 1.96) if np.isfinite(t) else False,
                     "survives_hlz_t3": bool(abs(t) > 3.0) if np.isfinite(t) else False})
    pd.DataFrame(rows).sort_values("nw_tstat", ascending=False).to_csv(
        TAB / "vol_ic_tstats.csv", index=False)
    if oos_frames:
        pd.concat(oos_frames, ignore_index=True).to_parquet(TAB / "vol_oos_xgb.parquet")
    log.info("DONE vol grid: %d window rows, %d cells", len(window_rows), len(rows))


if __name__ == "__main__":
    main()
