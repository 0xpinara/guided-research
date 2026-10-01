"""Task 0 re-fits: reproduce the EN headline cell and diagnose the OLS rank problem.

Read-only on every BIR-era file; writes only to revision/out/T0/.

(a) ElasticNet / Set D / rolling 9m/3m / 5-day. The BIR-era run saved only
    XGBoost's out-of-sample predictions, so this cell cannot be recomputed "from
    saved predictions". It is re-fitted here with the unchanged BIR-era code path
    (run_full_matrix._window_ranges, the h-date purge, _predict_window), the
    predictions are saved, and the per-window IC is compared with
    results/tables/full_matrix_walkforward_windows.csv. The long/short backtest is
    re-run on the fresh predictions with the BIR-era settings.
(b) OLS numerical rank. run_full_matrix fits sklearn LinearRegression on float32
    X. Three rolling 5-day windows are re-fitted in float32 and float64 for six
    feature sets, recording the effective rank and the window IC.

Run:  python revision/T0/t0_refits.py      (a few minutes; needs data/)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from run_pipeline import _resolve_feature_ids, _add_ticker_encoding  # noqa: E402
from run_full_matrix import _window_ranges, _predict_window, _cross_sectional_demean  # noqa: E402
from posthoc_stats import _nw_tstat  # noqa: E402
from src.evaluation.metrics import information_coefficient  # noqa: E402
from src.trading.backtest import backtest_long_short_deciles  # noqa: E402
from src.utils.config import load_feature_defs  # noqa: E402
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR  # noqa: E402

OUT = ROOT / "revision" / "out" / "T0"
OUT.mkdir(parents=True, exist_ok=True)
TAB = RESULTS_DIR / "tables"
EXP_MIN_TRAIN, EXP_STEP = 504, 63          # run_full_matrix defaults
ROLL_TRAIN, ROLL_TEST, ROLL_STEP = 189, 63, 63


def load_bir_panel() -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the modelling panel exactly as run_full_matrix.main() does."""
    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel, splits):
        df["date"] = pd.to_datetime(df["date"])
    panel = panel.merge(splits, on=["ticker", "date"], how="inner")
    rg = load_parquet(FEATURES_DIR / "resolution_2_surface" / "surface_features_all.parquet")
    rg["date"] = pd.to_datetime(rg["date"])
    rg_cols = [c for c in rg.columns if c.startswith("iv_surf_") or c.startswith("surface_")]
    rg = rg[["ticker", "date"] + rg_cols].drop_duplicates(["ticker", "date"])
    panel = panel.merge(rg, on=["ticker", "date"], how="left")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    ticker_cols = _add_ticker_encoding(panel)
    return panel, ticker_cols


def horizon_arrays(panel, ticker_cols, feat_defs, fs, h):
    """Same array construction as run_full_matrix._run_horizon (flat models)."""
    ph = panel.dropna(subset=[f"ret_{h}d", f"dir_{h}d"]).copy()
    feat_ids = _resolve_feature_ids(feat_defs, fs, ph.columns.tolist())
    X = ph.loc[:, feat_ids + ticker_cols].to_numpy(dtype=np.float32, na_value=np.nan)
    dates = ph["date"].to_numpy()
    raw = ph[f"ret_{h}d"].to_numpy(dtype=np.float64, na_value=np.nan)
    y = _cross_sectional_demean(raw, dates)
    for a in (X, y, raw):
        np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, raw, dates, ph["ticker"].to_numpy()


def window_masks(dates, unique_dates, win, h):
    tr_s, tr_e, te_s, te_e = win
    tb = unique_dates[tr_s:tr_e]
    if h > 0 and len(tb) > h:
        tb = tb[:-h]
    return np.isin(dates, list(set(tb))), np.isin(dates, list(set(unique_dates[te_s:te_e])))


def reproduce_en(panel, ticker_cols, feat_defs) -> list[str]:
    model, fs, scheme, h = "ElasticNet", "D", "rolling_9m_3m", 5
    X, y, raw, dates, tickers = horizon_arrays(panel, ticker_cols, feat_defs, fs, h)
    ud = np.sort(np.unique(dates))
    wins = _window_ranges(scheme, ud, EXP_MIN_TRAIN, EXP_STEP, ROLL_TRAIN, ROLL_TEST, ROLL_STEP)
    rows, frames = [], []
    for wi, win in enumerate(wins):
        trm, tem = window_masks(dates, ud, win, h)
        pred, yt, dt, tk, rt = _predict_window(model, {}, X, y, raw, tickers, dates, trm, tem)
        rows.append({"window": wi, "test_start": str(ud[win[2]])[:10],
                     "ic_fresh": float(information_coefficient(yt, pred))})
        frames.append(pd.DataFrame({"model": model, "feature_set": fs, "scheme": scheme,
                                    "horizon": h, "window": wi, "date": dt, "ticker": tk,
                                    "pred": pred, "y_demeaned": yt, "y_raw": rt}))
    preds = pd.concat(frames, ignore_index=True)
    preds.to_parquet(OUT / "en_D_rolling_h5_predictions.parquet", index=False)

    saved = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
    saved = saved[(saved.model == model) & (saved.feature_set == fs)
                  & (saved.scheme == scheme) & (saved.horizon == h)][["window", "test_start", "ic"]]
    cmp_ = pd.DataFrame(rows).merge(saved.rename(columns={"ic": "ic_saved",
                                                          "test_start": "test_start_saved"}),
                                    on="window", how="outer")
    cmp_["abs_diff"] = (cmp_.ic_fresh - cmp_.ic_saved).abs()
    cmp_.to_csv(OUT / "en_D_rolling_h5_window_ic_check.csv", index=False)
    m_f, se_f, t_f = _nw_tstat(cmp_.ic_fresh.to_numpy())
    m_s, se_s, t_s = _nw_tstat(cmp_.ic_saved.to_numpy())

    # Long/short on the fresh predictions, BIR-era settings (15% spread, 50bp borrow).
    daily, summ = backtest_long_short_deciles(
        dates=preds.date.to_numpy(), tickers=preds.ticker.to_numpy(),
        actual_returns=preds.y_raw.to_numpy(), predicted_returns=preds.pred.to_numpy(),
        quoted_spread=None, top_n=6, bottom_n=6, effective_spread_fraction=0.15,
        short_fee_bps_annual=50.0, horizon=h, rebalance_every=h)
    r = daily.portfolio_return
    arith = float(r.mean() / r.std(ddof=1) * np.sqrt(252 / h))
    out = [
        "(a) ElasticNet / D / rolling_9m_3m / h=5 re-fit with the BIR-era code path",
        f"    windows: fresh {cmp_.ic_fresh.notna().sum()}, saved {cmp_.ic_saved.notna().sum()}; "
        f"test-start dates identical: {bool((cmp_.test_start == cmp_.test_start_saved).all())}",
        f"    max |per-window IC fresh - saved| = {cmp_.abs_diff.max():.3e}",
        f"    mean IC fresh {m_f:.6f} (NW t {t_f:.4f}); saved {m_s:.6f} (NW t {t_s:.4f})",
        f"    long/short on fresh predictions: final $ {1000 * (1 + summ['total_return']):,.2f}, "
        f"Sharpe as coded (geometric) {summ['sharpe']:.4f}, "
        f"arithmetic mean/sd*sqrt(252/5) {arith:.4f}, periods {summ['n_periods']}",
        f"    predictions saved: revision/out/T0/en_D_rolling_h5_predictions.parquet "
        f"({len(preds):,} rows)",
    ]
    return out


def ols_rank(panel, ticker_cols, feat_defs) -> list[str]:
    scheme, h = "rolling_9m_3m", 5
    saved = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
    rows = []
    for fs in ["A", "repr_svi", "D", "repr_grid", "B", "C"]:
        X, y, raw, dates, tickers = horizon_arrays(panel, ticker_cols, feat_defs, fs, h)
        ud = np.sort(np.unique(dates))
        wins = _window_ranges(scheme, ud, EXP_MIN_TRAIN, EXP_STEP, ROLL_TRAIN, ROLL_TEST, ROLL_STEP)
        for wi in (0, 15, 31):
            trm, tem = window_masks(dates, ud, wins[wi], h)
            # run_full_matrix._train_val_masks: OLS is fitted on the first 80% of the
            # purged training dates; the last 20% is the XGBoost early-stopping fold.
            tr_dates = np.sort(np.unique(dates[trm]))
            fit_mask = np.isin(dates, tr_dates[:int(len(tr_dates) * 0.8)])
            rec = {"feature_set": fs, "window": wi, "n_cols": X.shape[1]}
            for tag, dt in (("float32", np.float32), ("float64", np.float64)):
                m = LinearRegression().fit(X[fit_mask].astype(dt), y[fit_mask])
                rec[f"rank_{tag}"] = int(m.rank_)
                rec[f"ic_{tag}"] = float(information_coefficient(y[tem], m.predict(X[tem].astype(dt))))
            s = saved[(saved.model == "OLS") & (saved.feature_set == fs) & (saved.scheme == scheme)
                      & (saved.horizon == h) & (saved.window == wi)]
            rec["ic_saved"] = float(s.ic.iloc[0]) if len(s) else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ols_rank_check.csv", index=False)
    out = ["(b) OLS effective rank, rolling_9m_3m h=5, windows 0/15/31 (float32 = BIR code path)"]
    for _, r in df.iterrows():
        out.append(f"    {r.feature_set:10s} w{r.window:<2d} cols={r.n_cols:3d} | float32 rank {r.rank_float32:3d} "
                   f"IC {r.ic_float32:+.6f} (saved {r.ic_saved:+.6f}) | float64 rank {r.rank_float64:3d} "
                   f"IC {r.ic_float64:+.6f}")
    out.append(f"    float32 fit reproduces the saved OLS IC in "
               f"{int(np.isclose(df.ic_float32, df.ic_saved, atol=1e-7).sum())}/{len(df)} window-cells")
    return out


def main() -> None:
    t0 = time.time()
    feat_defs = load_feature_defs()
    panel, ticker_cols = load_bir_panel()
    lines = [f"Task 0 re-fits (modelling panel: {len(panel):,} rows, {len(ticker_cols)} ticker dummies)"]
    lines += reproduce_en(panel, ticker_cols, feat_defs)
    lines += ols_rank(panel, ticker_cols, feat_defs)
    lines.append(f"run time: {time.time() - t0:.0f} s")
    (OUT / "t0_refits.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
