"""Task 0b — benchmark tables, old-vs-new comparison and IC-definition comparison.

Reads the per-window metrics written by run_grid.py and the BIR-era saved outputs (read-only).
Writes revision/out/T0b/benchmark/*.csv and benchmark/summary.txt.

IC definitions compared (0b.4)
  daily     canonical: mean over the window's test dates of the cross-sectional Spearman IC;
            a date with constant predictions scores 0
  pooled    BIR: one Spearman over all ticker-days of the window; a window with constant
            predictions is NaN and is dropped from the cell mean (as in the BIR tables)
  pooled0   the BIR pooled IC with constant-prediction windows scored 0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import OUT, ROOT, SETS, MODELS, nw_tstat, daily_ic, pooled_ic  # noqa: E402

BEN = OUT / "benchmark"
TAB = ROOT / "results" / "tables"
CELL = ["target", "model", "feature_set", "scheme", "horizon"]
LINES: list[str] = []


def emit(s=""):
    LINES.append(str(s))
    print(s)


def cell_stats(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in w.groupby(CELL):
        g = g.sort_values("window")
        d = nw_tstat(g.ic_daily.to_numpy())
        ds = nw_tstat(g.ic_daily_stocks.to_numpy())
        p = nw_tstat(g.ic_pooled.to_numpy())
        p0 = nw_tstat(g.ic_pooled.fillna(0).to_numpy())
        rows.append(dict(zip(CELL, k)) | {
            "n_windows": len(g), "ic": d[0], "nw_se": d[1], "t": d[2],
            "ic_stocks": ds[0], "t_stocks": ds[2],
            "ic_pooled": p[0], "t_pooled": p[2], "n_windows_pooled_defined": int(g.ic_pooled.notna().sum()),
            "ic_pooled0": p0[0], "t_pooled0": p0[2],
            "share_constant_dates": g.n_constant_dates.sum() / g.n_test_dates.sum(),
            "oos_r2_mean": g.oos_r2.mean(), "indicators_mean": g.n_indicators.mean()})
    return pd.DataFrame(rows)


def table3(C: pd.DataFrame) -> pd.DataFrame:
    s = C[(C.scheme == "rolling_9m_3m") & (C.horizon == 5)]
    out = []
    for fs in SETS:
        row = {"feature_set": fs}
        for tgt in ("return", "volatility"):
            for m in MODELS:
                r = s[(s.target == tgt) & (s.model == m) & (s.feature_set == fs)].iloc[0]
                row[f"{tgt}_{m}"] = r.ic
                row[f"{tgt}_{m}_t"] = r.t
        out.append(row)
    return pd.DataFrame(out)


def paired_lift(w: pd.DataFrame, target: str, fs: str, models=("ElasticNet", "XGBoost")) -> dict:
    sub = w[(w.target == target) & w.model.isin(models)]
    pv = sub.pivot_table(index=["model", "scheme", "horizon", "window"], columns="feature_set", values="ic_daily")
    d = (pv[fs] - pv["A"]).dropna()
    cells = d.groupby(level=[0, 1, 2]).mean()
    ts = np.array([nw_tstat(g.to_numpy())[2] for _, g in d.groupby(level=[0, 1, 2])], dtype=float)
    ok = np.isfinite(ts)        # t undefined where both fits are null models in every window (lift = 0, SE = 0)
    return {"lift": float(cells.mean()), "n_cells": len(cells), "n_cells_t_undefined": int((~ok).sum()),
            "cell_t_min": float(np.min(ts[ok])) if ok.any() else np.nan,
            "cell_t_median": float(np.median(ts[ok])) if ok.any() else np.nan,
            "cell_t_max": float(np.max(ts[ok])) if ok.any() else np.nan}


def bir_windows_from_dates(dates: pd.Series) -> pd.Series:
    """BIR test windows are consecutive 63-date blocks of the sorted test dates."""
    ud = np.sort(dates.unique())
    return dates.map(dict(zip(ud, np.arange(len(ud)) // 63)))


def three_definitions(pred: pd.DataFrame) -> dict:
    """pred: window, date, pred, y.  Returns cell means and NW t under the three definitions."""
    per = []
    for wi, g in pred.groupby("window"):
        per.append({"daily": daily_ic(g.date.to_numpy(), g.pred.to_numpy(), g.y.to_numpy()).ic.mean(),
                    "pooled": pooled_ic(g.pred.to_numpy(), g.y.to_numpy())})
    per = pd.DataFrame(per)
    out = {}
    for name, x in (("pooled", per.pooled), ("pooled0", per.pooled.fillna(0)), ("daily", per.daily)):
        m, _, t = nw_tstat(x.to_numpy())
        out[name] = (m, t, int(x.notna().sum()))
    return out


def ic_definition_comparison(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # BIR pipeline predictions
    en = pd.read_parquet(ROOT / "revision/out/T0/en_D_rolling_h5_predictions.parquet").rename(columns={"y_demeaned": "y"})
    xg = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet",
                         filters=[("feature_set", "==", "D"), ("scheme", "==", "expanding"), ("horizon", "==", 3)])
    xg = xg.rename(columns={"y_demeaned": "y"})
    xg["date"] = pd.to_datetime(xg["date"])
    xg["window"] = bir_windows_from_dates(xg.date)
    new = {}
    for (m, fs, s, h) in (("ElasticNet", "D", "rolling_9m_3m", 5), ("XGBoost", "D", "expanding", 3)):
        f = OUT / "predictions" / "return" / f"h{h}" / f"{fs}.parquet"
        p = pd.read_parquet(f, filters=[("model", "==", m), ("scheme", "==", s)])
        new[(m, fs, s, h)] = p
    for (cell, bir) in ((("ElasticNet", "D", "rolling_9m_3m", 5), en), (("XGBoost", "D", "expanding", 3), xg)):
        for pipe, df in (("BIR pipeline (saved / T0 re-fit predictions)", bir), ("corrected pipeline (Task 0b)", new[cell])):
            r = three_definitions(df)
            for defn, (m_, t_, n_) in r.items():
                rows.append({"cell": "/".join(map(str, cell)), "pipeline": pipe, "definition": defn,
                             "mean_ic": m_, "nw_t": t_, "windows_with_defined_ic": n_})
    return pd.DataFrame(rows)


def constant_prediction_stats(W: pd.DataFrame) -> None:
    """How often predictions are constant across names on a test date (scored IC = 0)."""
    W = W.assign(all_constant=W.n_constant_dates == W.n_test_dates)
    g = W.groupby(["target", "model", "scheme", "horizon"]).agg(
        windows=("window", "size"), test_dates=("n_test_dates", "sum"), constant_dates=("n_constant_dates", "sum"),
        windows_all_constant=("all_constant", "sum"))
    g["share_constant_dates"] = g.constant_dates / g.test_dates
    g.reset_index().to_csv(BEN / "constant_predictions.csv", index=False)
    emit("\nConstant-prediction dates (scored IC = 0), share of test dates")
    tot = W.groupby(["target", "model"]).agg(td=("n_test_dates", "sum"), cd=("n_constant_dates", "sum"),
                                              w=("window", "size"), wa=("all_constant", "sum"))
    for (t, m), r in tot.iterrows():
        emit(f"  {t:10s} {m:10s}: {r.cd / r.td:6.1%} of test dates; windows constant throughout {int(r.wa)}/{int(r.w)}")


def headline_table(C: pd.DataFrame, L: pd.DataFrame) -> None:
    """Paper (BIR) headline numbers next to their corrected values; economics read from economics/."""
    def c(tgt, m, fs, s, h):
        return C[(C.target == tgt) & (C.model == m) & (C.feature_set == fs) & (C.scheme == s) & (C.horizon == h)].iloc[0]

    def lift(tgt, fs):
        return L[(L.target == tgt) & (L.feature_set == fs)].iloc[0]
    r3, v3 = C[C.target == "return"], C[C.target == "volatility"]
    se = r3.nw_se.median()
    rows = [
        ("return cells with t > 3 (of 162)", "7", f"{int((r3.t > 3).sum())}"),
        ("volatility cells with t > 3 (of 162)", "158", f"{int((v3.t > 3).sum())}"),
        ("largest volatility HAC t", "58.7", f"{v3.t.max():.1f}"),
        ("median HAC SE of the mean return IC", "0.0064 (198 cells incl. FFNN/TFT)", f"{se:.4f} (162 cells)"),
        ("t>3 detection threshold (3 x SE)", "0.019", f"{3 * se:.4f}"),
        ("return lift D over A (EN+XGB, full grid)", "+0.0027 (paper: about 0.003)", f"{lift('return', 'D').lift:+.4f}"),
        ("volatility lift D over A (EN+XGB, full grid)", "0.0362 ('more than ten SEs')",
         f"{lift('volatility', 'D').lift:.4f} (per-cell paired t {lift('volatility', 'D').cell_t_min:.1f}..{lift('volatility', 'D').cell_t_max:.1f})"),
        ("volatility lift repr_svi over A", "0.0235", f"{lift('volatility', 'repr_svi').lift:.4f}"),
        ("Set A volatility IC, EN rolling 5d", "0.520", f"{c('volatility', 'ElasticNet', 'A', 'rolling_9m_3m', 5).ic:.3f}"),
    ]
    for (m, fs, s, h, old) in (("ElasticNet", "repr_grid", "rolling_9m_3m", 5, "0.0346 (t 4.28)"),
                               ("ElasticNet", "D", "rolling_9m_3m", 5, "0.0331 (t 4.81)"),
                               ("ElasticNet", "A", "rolling_9m_3m", 5, "0.0311 (t 3.21)"),
                               ("ElasticNet", "repr_bkm", "rolling_9m_3m", 5, "0.0312 (t 3.24)"),
                               ("XGBoost", "A", "expanding", 1, "0.0136 (t 3.64)"),
                               ("XGBoost", "D", "expanding", 3, "0.0118 (t 1.47)"),
                               ("XGBoost", "D", "rolling_9m_3m", 5, "-0.0025 (t -0.39)")):
        x = c("return", m, fs, s, h)
        rows.append((f"return IC {m}/{fs}/{s}/h{h}", old, f"{x.ic:+.4f} (t {x.t:+.2f})"))
    econ = OUT / "economics" / "strategy_summary.csv"
    if econ.exists():
        S = pd.read_csv(econ)
        hl = S[(S.model == "ElasticNet") & (S.feature_set == "D") & (S.scheme == "rolling_9m_3m") & (S.horizon == 5)]
        st, al = hl[hl.universe == "stocks"].iloc[0], hl[hl.universe == "all_names"].iloc[0]
        fam = S[S.universe == "stocks"]
        pbo = pd.read_csv(OUT / "economics" / "pbo.csv")
        rows += [
            ("headline long/short Sharpe (EN/D/rolling/5d)", "1.15 (geometric formula, all 60 names)",
             f"{st.sharpe:.3f} stocks only, arithmetic ({al.sharpe:.3f} all names)"),
            ("headline final $ from $1,000", "5,228", f"{st.final_dollars:,.0f} stocks ({al.final_dollars:,.0f} all names)"),
            ("headline periods flat (all predictions tied)", "0 (ties broken arbitrarily)", f"{st.share_flat:.1%}"),
            ("ETF-free headline (Sharpe / $)", "1.07 / 4,585 (geometric)", f"{st.sharpe:.3f} / {st.final_dollars:,.0f}"),
            ("SR0 (expected-maximum Sharpe)", "1.153 (N = 198)", f"{st.sr0_family:.3f} (N = {len(fam)})"),
            ("configurations above SR0", "2 of 198", f"{int(fam.clears_sr0.sum())} of {len(fam)}"),
            ("DSR of the headline cell", "about 0.497 (referee_fixes.txt)", f"{st.dsr:.3f}"),
            ("PBO", "0.571 (expanding/3d group)",
             f"{pbo.iloc[0].pbo:.3f} (family of 162); headline group rolling h5 "
             f"{pbo[pbo.family.str.startswith('rolling_9m_3m h=5')].pbo.iloc[0]:.3f}"),
        ]
    H = pd.DataFrame(rows, columns=["quantity", "paper (BIR pipeline)", "corrected (Task 0b)"])
    H.to_csv(BEN / "headline_numbers.csv", index=False)
    emit("\nHeadline numbers: paper vs corrected")
    for r in H.itertuples():
        emit(f"  {r.quantity:48s} | {r._2:40s} | {r._3}")


def main() -> None:
    parts = sorted((BEN / "parts").glob("*.csv"))
    W = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    W.to_csv(BEN / "windows.csv", index=False)
    C = cell_stats(W)
    C.to_csv(BEN / "cells.csv", index=False)
    emit(f"cells: {len(C)} (expected 324); windows: {len(W):,}")
    for tgt in ("return", "volatility"):
        c = C[C.target == tgt]
        emit(f"{tgt}: t>3 cells {int((c.t > 3).sum())}/{len(c)} (canonical daily IC); max t {c.t.max():.1f}; "
             f"median NW SE {c.nw_se.median():.4f} -> 3xSE {3 * c.nw_se.median():.4f}; "
             f"share of test dates with constant predictions (EN) "
             f"{c[c.model == 'ElasticNet'].share_constant_dates.mean():.1%}")
    sig = C[(C.target == "return") & (C.t > 3)].sort_values("t", ascending=False)
    emit("return cells with t > 3 (canonical IC): " + "; ".join(
        f"{r.model}/{r.feature_set}/{r.scheme}/h{r.horizon} IC {r.ic:+.4f} t {r.t:.2f}" for r in sig.itertuples()))
    T3 = table3(C)
    T3.to_csv(BEN / "table3_corrected.csv", index=False)
    emit("\nCorrected Table 3 (rolling 9m/3m, h=5; mean daily cross-sectional IC; * = t > 3)")
    emit(f"{'set':14s}" + "".join(f"{('ret ' if t == 'return' else 'vol ') + m[:3]:>11s}" for t in ('return', 'volatility') for m in MODELS))
    for r in T3.itertuples():
        cells = []
        for t in ("return", "volatility"):
            for m in MODELS:
                v, tt = getattr(r, f"{t}_{m}"), getattr(r, f"{t}_{m}_t")
                cells.append(f"{v:+.4f}{'*' if tt > 3 else ' '}" if t == "return" else f"{v:+.3f}{'*' if tt > 3 else ' '}")
        emit(f"{r.feature_set:14s}" + "".join(f"{c:>11s}" for c in cells))

    # lifts over Set A (EN + XGB, full grid), as in the paper's dissociation numbers
    emit("\nMarginal lift over Set A (mean over EN+XGB x 2 schemes x 3 horizons; canonical IC)")
    lift_rows = []
    for tgt in ("return", "volatility"):
        for fs in [s for s in SETS if s != "A"]:
            r = paired_lift(W, tgt, fs)
            lift_rows.append({"target": tgt, "feature_set": fs} | r)
            emit(f"  {tgt:10s} {fs:14s} lift {r['lift']:+.4f}; per-cell paired HAC t {r['cell_t_min']:+.1f} .. "
                 f"{r['cell_t_max']:+.1f} (median {r['cell_t_median']:+.1f}"
                 + (f"; t undefined in {r['n_cells_t_undefined']} cells: lift 0 in every window" if r['n_cells_t_undefined'] else "")
                 + ")")
    pd.DataFrame(lift_rows).to_csv(BEN / "lifts_over_A.csv", index=False)

    # old (BIR) vs new
    old_r = pd.read_csv(TAB / "posthoc_ic_tstats.csv").assign(target="return")
    old_v = pd.read_csv(TAB / "vol_ic_tstats.csv").assign(target="volatility")
    old = pd.concat([old_r, old_v])
    old = old[old.model.isin(MODELS)].rename(columns={"mean_ic": "bir_ic", "nw_tstat": "bir_t"})
    ON = C.merge(old[CELL + ["bir_ic", "bir_t"]], on=CELL, how="left")
    ON.to_csv(BEN / "old_vs_new_cells.csv", index=False)
    head = ON[ON.feature_set.isin(["A", "D", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"])]
    head[CELL + ["bir_ic", "bir_t", "ic", "t", "ic_pooled", "t_pooled", "share_constant_dates"]].to_csv(
        BEN / "old_vs_new_headline_cells.csv", index=False)
    for tgt in ("return", "volatility"):
        h_ = head[head.target == tgt]
        emit(f"old vs new ({tgt}, 108 headline cells): corr(BIR IC, new IC) {h_[['bir_ic', 'ic']].corr().iloc[0, 1]:.3f}; "
             f"t>3 BIR {int((h_.bir_t > 3).sum())} -> new {int((h_.t > 3).sum())}")

    constant_prediction_stats(W)
    headline_table(C, pd.DataFrame(lift_rows))

    IC = ic_definition_comparison(W)
    IC.to_csv(BEN / "ic_definition_comparison.csv", index=False)
    emit("\nIC-definition comparison (two headline cells)")
    for r in IC.itertuples():
        emit(f"  {r.cell:32s} {r.pipeline:46s} {r.definition:8s} IC {r.mean_ic:+.4f}  t {r.nw_t:+.2f}  "
             f"(windows {r.windows_with_defined_ic})")
    (BEN / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
