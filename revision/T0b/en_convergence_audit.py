"""Task 0b — ElasticNet convergence audit (read-only on the canonical outputs).

The canonical run logs sklearn ConvergenceWarnings from ElasticNetCV. ElasticNetCV first fits
the alpha path on 3 CV folds (alphas 1e-1, 1e-2, 1e-3, 1e-4 with warm starts), then refits
the selected alpha on the whole training block. A warning from the CV path does not mean the
selected model failed to converge; this audit separates the two.

For every ElasticNet window of the canonical grid it re-fits the identical model (same panel,
features, per-window preprocessing, scaler and ElasticNetCV arguments) and records
  * warnings_total: ConvergenceWarnings raised during the whole ElasticNetCV fit;
  * final_n_iter / final_dual_gap: iterations and duality gap of the final refit at the
    selected alpha (sklearn stores them in n_iter_ / dual_gap_);
  * final_converged: final_n_iter < max_iter (sklearn stops at max_iter without converging);
  * pred_max_abs_diff: max |re-fit prediction - saved canonical prediction| (expected 0),
    which shows that the audited fit is the canonical fit.
Nothing in the model specification is changed. Writes revision/out/T0b/en_convergence/.
Run after run_grid.py:  python revision/T0b/en_convergence_audit.py [--workers 6]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import OUT, SETS, SCHEMES, HORIZONS, feature_columns, windows, demean_by_date, WindowPreprocessor  # noqa: E402

TARGET_COL = {"return": "ret_{h}d", "volatility": "rv_{h}d"}
MAX_ITER, TOL = 1000, 1e-3
AUD = OUT / "en_convergence"
_PANEL = None


def _init(path):
    global _PANEL
    _PANEL = pd.read_parquet(path)
    _PANEL["date"] = pd.to_datetime(_PANEL["date"])
    _PANEL.sort_values(["ticker", "date"], inplace=True, ignore_index=True)


def audit_job(job):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    target, h, fs = job
    ycol = TARGET_COL[target].format(h=h)
    p = _PANEL[_PANEL[ycol].notna()].reset_index(drop=True)
    cols = feature_columns(fs, p.columns)
    X_all = p[cols].to_numpy(np.float64)
    dummies = pd.get_dummies(p.ticker, prefix="tkr", dtype=np.float64)
    D_all = dummies.drop(columns=dummies.columns[-1]).to_numpy()
    dates, tickers = p.date.to_numpy(), p.ticker.to_numpy()
    y = demean_by_date(p[ycol].to_numpy(np.float64), dates)
    ud = np.sort(np.unique(dates))
    saved = pd.read_parquet(OUT / "predictions" / target / f"h{h}" / f"{fs}.parquet",
                            filters=[("model", "==", "ElasticNet")])
    rows = []
    for scheme in SCHEMES:
        sv = saved[saved.scheme == scheme]
        for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows(scheme, len(ud))):
            tr_block = ud[tr_s:tr_e]
            tr_block = tr_block[:-h] if len(tr_block) > h else tr_block
            trm, tem = np.isin(dates, tr_block), np.isin(dates, ud[te_s:te_e])
            pp = WindowPreprocessor().fit(X_all[trm], tickers[trm], impute=True)
            Xtr = np.hstack([pp.transform(X_all[trm], tickers[trm]), D_all[trm]])
            Xte = np.hstack([pp.transform(X_all[tem], tickers[tem]), D_all[tem]])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                m = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=0.5, alphas=[1e-4, 1e-3, 1e-2, 1e-1], cv=3,
                                                                  precompute=False, random_state=42, n_jobs=1,
                                                                  max_iter=MAX_ITER, tol=TOL)).fit(Xtr, y[trm])
            en = m[-1]
            pred = m.predict(Xte)
            ref = sv[sv.window == wi].set_index(["date", "ticker"]).pred
            got = pd.Series(pred, index=pd.MultiIndex.from_arrays([dates[tem], tickers[tem]])).reindex(ref.index)
            rows.append({"target": target, "feature_set": fs, "scheme": scheme, "horizon": h, "window": wi,
                         "alpha": float(en.alpha_), "n_nonzero": int((en.coef_ != 0).sum()),
                         "warnings_total": sum(issubclass(x.category, ConvergenceWarning) for x in w),
                         "final_n_iter": int(en.n_iter_), "final_dual_gap": float(en.dual_gap_),
                         "final_converged": bool(en.n_iter_ < MAX_ITER),
                         "pred_max_abs_diff": float(np.nanmax(np.abs(got.to_numpy() - ref.to_numpy())))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    AUD.mkdir(parents=True, exist_ok=True)
    jobs = [(t, h, fs) for t in ("return", "volatility") for h in HORIZONS for fs in SETS]
    with Pool(a.workers, initializer=_init, initargs=(str(OUT / "panel" / "panel_primary.parquet"),)) as pool:
        rows = [r for part in pool.imap_unordered(audit_job, jobs) for r in part]
    df = pd.DataFrame(rows).sort_values(["target", "feature_set", "scheme", "horizon", "window"])
    df.to_csv(AUD / "en_fits.csv", index=False)
    lines = [f"ElasticNet fits audited: {len(df)} windows in {df.groupby(['target', 'feature_set', 'scheme', 'horizon']).ngroups} cells",
             f"re-fit reproduces the saved canonical predictions: max |diff| {df.pred_max_abs_diff.max():.3e}",
             f"ConvergenceWarnings raised during ElasticNetCV fits: {int(df.warnings_total.sum())} in "
             f"{int((df.warnings_total > 0).sum())} of {len(df)} window fits",
             f"final refits at the selected alpha that did NOT converge (n_iter = max_iter = {MAX_ITER}): "
             f"{int((~df.final_converged).sum())} of {len(df)}"]
    for (tgt,), g in df.groupby(["target"]):
        lines.append(f"  {tgt}: warnings {int(g.warnings_total.sum())} in {int((g.warnings_total > 0).sum())} fits; "
                     f"final fits not converged {int((~g.final_converged).sum())}/{len(g)}; selected alpha counts "
                     f"{g.alpha.value_counts().sort_index().to_dict()}")
    nc = df[~df.final_converged]
    if len(nc):
        lines.append("non-converged final fits by selected alpha: " + str(nc.alpha.value_counts().to_dict()))
        lines.append("their max final duality gap: " + f"{nc.final_dual_gap.max():.3e}")
    lines.append(f"final-fit iterations: median {df.final_n_iter.median():.0f}, max {df.final_n_iter.max()}")
    (AUD / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
