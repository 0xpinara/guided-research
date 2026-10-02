"""Task 0b.2-0b.5 — corrected walk-forward benchmark for OLS, ElasticNet and XGBoost.

Grid: 2 targets (forward return, forward realised volatility) x 3 horizons x 9 feature sets
x 2 schemes x 3 models = 324 cells, on revision/out/T0b/panel/panel_primary.parquet
(or --panel unfiltered for the robustness comparison).

Per window (same geometry as BIR; last h training dates purged):
  * preprocessing fitted on the training rows only (common.WindowPreprocessor)
  * OLS: float64 LinearRegression on the full purged training block (imputed + indicators + dummies)
  * ElasticNet: StandardScaler + ElasticNetCV(l1_ratio=0.5, alphas 1e-4..1e-1, cv=3, random_state=42,
    max_iter=1000, tol=1e-3), float64, full purged training block
  * XGBoost: BIR hyperparameters (500 trees, depth 6, lr 0.01, subsample 0.8, colsample 0.8, early
    stopping 50, rmse), random_state=42, n_jobs=1; fitted on the first 80% of training dates minus
    the last h (purge), early-stopped on the last 20%; NaN handled natively
  * target: cross-sectionally demeaned per date over the names present
Saves every test prediction and per-window metrics (canonical daily IC and the BIR pooled IC).

Run:  python revision/T0b/run_grid.py [--panel primary|unfiltered] [--targets return volatility]
          [--sets ...] [--horizons ...] [--schemes ...] [--models ...] [--workers 7]
Jobs whose outputs exist are skipped (resumable).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (OUT, SETS, MODELS, SCHEMES, HORIZONS, feature_columns, windows,  # noqa: E402
                    demean_by_date, WindowPreprocessor, daily_ic, pooled_ic, ETFS)

TARGET_COL = {"return": "ret_{h}d", "volatility": "rv_{h}d"}
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                  early_stopping_rounds=50, eval_metric="rmse", random_state=42, n_jobs=1, tree_method="hist")
_PANEL: pd.DataFrame | None = None


def _init(panel_path: str) -> None:
    global _PANEL
    _PANEL = pd.read_parquet(panel_path)
    _PANEL["date"] = pd.to_datetime(_PANEL["date"])
    _PANEL.sort_values(["ticker", "date"], inplace=True, ignore_index=True)


def fit_predict(model, X_tr, y_tr, X_te, tr_dates, h, tickers_tr, tickers_te, X_raw_tr, X_raw_te):
    from sklearn.linear_model import ElasticNetCV, LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    info = {}
    if model == "OLS":
        m = LinearRegression().fit(X_tr, y_tr)
        info["rank"] = int(m.rank_)
        return m.predict(X_te), info
    if model == "ElasticNet":
        m = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=0.5, alphas=[1e-4, 1e-3, 1e-2, 1e-1], cv=3,
                                                          precompute=False, random_state=42, n_jobs=1,
                                                          max_iter=1000, tol=1e-3)).fit(X_tr, y_tr)
        en = m[-1]
        info["alpha"] = float(en.alpha_)
        info["n_nonzero"] = int((en.coef_ != 0).sum())
        return m.predict(X_te), info
    import xgboost as xgb
    ud = np.sort(np.unique(tr_dates))
    cut = int(len(ud) * 0.8)
    fit_dates, val_dates = ud[: max(cut - h, 1)], ud[cut:]
    fm, vm = np.isin(tr_dates, fit_dates), np.isin(tr_dates, val_dates)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_raw_tr[fm], y_tr[fm], eval_set=[(X_raw_tr[vm], y_tr[vm])], verbose=False)
    info["best_iteration"] = int(m.best_iteration)
    return m.predict(X_raw_te), info


def run_job(job: tuple) -> str:
    target, h, fs, schemes, models, tag = job
    pred_path = OUT / f"predictions{tag}" / target / f"h{h}" / f"{fs}.parquet"
    met_path = OUT / f"benchmark{tag}" / "parts" / f"{target}_h{h}_{fs}.csv"
    if pred_path.exists() and met_path.exists():
        return f"skip {target} h{h} {fs}"
    t0 = time.time()
    ycol = TARGET_COL[target].format(h=h)
    p = _PANEL[_PANEL[ycol].notna()].reset_index(drop=True)
    cols = feature_columns(fs, p.columns)
    X_all = p[cols].to_numpy(np.float64)
    dummies = pd.get_dummies(p.ticker, prefix="tkr", dtype=np.float64)
    D_all = dummies.drop(columns=dummies.columns[-1]).to_numpy()
    dates, tickers = p.date.to_numpy(), p.ticker.to_numpy()
    y_raw = p[ycol].to_numpy(np.float64)
    y = demean_by_date(y_raw, dates)
    is_stock = ~np.isin(tickers, ETFS)
    ud = np.sort(np.unique(dates))
    pred_frames, rows = [], []
    for scheme in schemes:
        for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows(scheme, len(ud))):
            tr_block = ud[tr_s:tr_e]
            tr_block = tr_block[:-h] if len(tr_block) > h else tr_block
            trm, tem = np.isin(dates, tr_block), np.isin(dates, ud[te_s:te_e])
            pp_lin = WindowPreprocessor().fit(X_all[trm], tickers[trm], impute=True)
            pp_xgb = WindowPreprocessor().fit(X_all[trm], tickers[trm], impute=False)
            Xl_tr = np.hstack([pp_lin.transform(X_all[trm], tickers[trm]), D_all[trm]])
            Xl_te = np.hstack([pp_lin.transform(X_all[tem], tickers[tem]), D_all[tem]])
            Xx_tr = np.hstack([pp_xgb.transform(X_all[trm], tickers[trm]), D_all[trm]])
            Xx_te = np.hstack([pp_xgb.transform(X_all[tem], tickers[tem]), D_all[tem]])
            for model in models:
                pred, info = fit_predict(model, Xl_tr, y[trm], Xl_te, dates[trm], h, tickers[trm], tickers[tem],
                                         Xx_tr, Xx_te)
                dte, yte = dates[tem], y[tem]
                dic = daily_ic(dte, pred, yte)
                dic_s = daily_ic(dte[is_stock[tem]], pred[is_stock[tem]], yte[is_stock[tem]])
                ss_res = float(np.sum((yte - pred) ** 2))
                ss_tot = float(np.sum((yte - y[trm].mean()) ** 2))
                rows.append({
                    "target": target, "model": model, "feature_set": fs, "scheme": scheme, "horizon": h,
                    "window": wi, "train_start": str(tr_block[0])[:10], "train_end": str(tr_block[-1])[:10],
                    "test_start": str(ud[te_s])[:10], "test_end": str(ud[te_e - 1])[:10],
                    "n_train": int(trm.sum()), "n_test": int(tem.sum()), "n_test_dates": int(len(dic)),
                    "ic_daily": float(dic.ic.mean()), "ic_daily_stocks": float(dic_s.ic.mean()),
                    "n_constant_dates": int(dic.constant.sum()),
                    "ic_pooled": pooled_ic(pred, yte), "oos_r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                    "n_features_kept": int(pp_lin.keep.sum()), "n_indicators": len(pp_lin.ind_cols), **info})
                pred_frames.append(pd.DataFrame({
                    "target": target, "model": model, "feature_set": fs, "scheme": scheme, "horizon": h,
                    "window": wi, "date": dte, "ticker": tickers[tem], "pred": pred, "y": yte,
                    "y_raw": y_raw[tem]}))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    met_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_frames, ignore_index=True).to_parquet(pred_path, index=False, compression="zstd")
    pd.DataFrame(rows).to_csv(met_path, index=False)
    return f"done {target} h{h} {fs} in {time.time() - t0:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="primary", choices=["primary", "unfiltered"])
    ap.add_argument("--targets", nargs="+", default=["return", "volatility"])
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    tag = "" if a.panel == "primary" else "_unfiltered"
    jobs = [(t, h, fs, a.schemes, a.models, tag) for t in a.targets for h in a.horizons for fs in a.sets]
    log = OUT / "logs"
    log.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(str(OUT / "panel" / f"panel_{a.panel}.parquet"),)) as pool:
        for msg in pool.imap_unordered(run_job, jobs):
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            print(line, flush=True)
            with open(log / f"run_grid{tag}.log", "a") as f:
                f.write(line + "\n")
    print(f"all jobs finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
