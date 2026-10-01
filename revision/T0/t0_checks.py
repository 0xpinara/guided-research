"""Task 0 audit checks: recompute every number cited in revision/out/T0/README.md.

Read-only on all BIR-era files (results/tables, data/); writes only to
revision/out/T0/. Run t0_refits.py first (section S3 reads its EN predictions).

Run:  python revision/T0/t0_checks.py        (about 5 minutes; needs data/)
Seeds: regime moving-block bootstrap 1 (per cell/conditioner/block length);
       meta-analysis quarter bootstrap 7 (return_extras.py default);
       published i.i.d. regime bootstrap 1 (return_extras.py default).
"""
from __future__ import annotations

import importlib
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.stats import norm  # noqa: E402

import return_extras as rx  # noqa: E402  (BIR-era code, unchanged)
from posthoc_stats import _nw_tstat, pbo_cscv  # noqa: E402
from src.trading.backtest import backtest_long_short_deciles  # noqa: E402

TAB = ROOT / "results" / "tables"
DATA = ROOT / "data"
OUT = ROOT / "revision" / "out" / "T0"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["model", "feature_set", "scheme", "horizon"]
M3 = ["OLS", "ElasticNet", "XGBoost"]
ETFS = {"SPY", "QQQ", "IWM"}
LINES: list[str] = []


def emit(s: str = "") -> None:
    LINES.append(str(s))
    print(s)


def head(title: str) -> None:
    emit("")
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def tstats(windows: pd.DataFrame, nan_as_zero: bool = False) -> pd.DataFrame:
    rows = []
    for k, g in windows.groupby(KEYS):
        x = g.sort_values("window")["ic"].to_numpy(float)
        n_nan = int(np.isnan(x).sum())
        if nan_as_zero:
            x = np.nan_to_num(x, nan=0.0)
        m, se, t = _nw_tstat(x)
        rows.append(dict(zip(KEYS, k)) | {"n_windows": len(x), "n_nan": n_nan,
                                          "mean_ic": m, "nw_se": se, "nw_t": t})
    return pd.DataFrame(rows)


W = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
VW = pd.read_csv(TAB / "vol_walkforward_windows.csv")
RT, VT = tstats(W), tstats(VW)
RT0 = tstats(W, nan_as_zero=True)


# ---------------------------------------------------------------- S1 tables
PAPER_T3 = {  # set: return OLS, EN, XGB | volatility OLS, EN, XGB  (rolling 9m/3m, h=5)
    "A": (0.0018, 0.0311, 0.0032, 0.252, 0.520, 0.501),
    "B": (0.0128, 0.0290, 0.0119, 0.044, 0.545, 0.534),
    "C": (0.0128, 0.0233, 0.0038, 0.044, 0.556, 0.550),
    "candidate_6": (0.0182, 0.0101, 0.0125, 0.511, 0.522, 0.503),
    "D": (0.0104, 0.0331, -0.0025, 0.414, 0.553, 0.555),
    "repr_svi": (0.0018, 0.0066, 0.0074, 0.252, 0.536, 0.532),
    "repr_grid": (0.0104, 0.0346, 0.0053, 0.414, 0.550, 0.554),
    "repr_grid_raw": (0.0063, 0.0095, -0.0004, 0.322, 0.555, 0.556),
    "repr_bkm": (0.0043, 0.0312, 0.0037, 0.349, 0.520, 0.513),
}
DAGGER_RET = {("ElasticNet", s) for s in ("A", "D", "repr_grid", "repr_bkm")}
DAGGER_VOL_MISS = {("OLS", "B"), ("OLS", "C")}
PAPER_T4 = {
    ("ElasticNet", "rolling_9m_3m", 5): {"A": (0.0311, 3.21), "repr_svi": (0.0066, 0.47),
                                         "repr_bkm": (0.0312, 3.24), "repr_grid_raw": (0.0095, 0.69),
                                         "repr_grid": (0.0346, 4.28), "D": (0.0331, 4.81)},
    ("XGBoost", "expanding", 1): {"A": (0.0136, 3.64), "repr_svi": (0.0041, 0.91),
                                  "repr_bkm": (0.0056, 1.92), "repr_grid_raw": (0.0052, 1.56),
                                  "repr_grid": (0.0122, 2.34), "D": (0.0053, 1.02)},
}


def cell(df, m, fs, s, h):
    r = df[(df.model == m) & (df.feature_set == fs) & (df.scheme == s) & (df.horizon == h)]
    return r.iloc[0] if len(r) else None


def s1_tables():
    head("S1  Tables 3 and 4 recomputed from the saved per-window IC files")
    saved = pd.read_csv(TAB / "posthoc_ic_tstats.csv").merge(RT, on=KEYS, suffixes=("_saved", ""))
    emit(f"posthoc_ic_tstats.csv vs recomputation from windows: max |dIC| "
         f"{(saved.mean_ic_saved - saved.mean_ic).abs().max():.2e}, max |dt| "
         f"{(saved.nw_tstat - saved.nw_t).abs().max():.2e}")
    rows = []
    for fs, vals in PAPER_T3.items():
        for tgt, df, nd, vv in (("return", RT, 4, vals[:3]), ("volatility", VT, 3, vals[3:])):
            for m, pv in zip(M3, vv):
                r = cell(df, m, fs, "rolling_9m_3m", 5)
                dag_paper = ((m, fs) in DAGGER_RET) if tgt == "return" else ((m, fs) not in DAGGER_VOL_MISS)
                rows.append({"table": "T3", "target": tgt, "model": m, "feature_set": fs,
                             "paper": pv, "recomputed": r.mean_ic, "nw_t": r.nw_t,
                             "value_match": abs(round(r.mean_ic, nd) - pv) < 1e-9,
                             "dagger_match": bool(r.nw_t > 3) == dag_paper})
    for (m, s, h), d in PAPER_T4.items():
        a = cell(RT, m, "A", s, h).mean_ic
        for fs, (pic, pt) in d.items():
            r = cell(RT, m, fs, s, h)
            rows.append({"table": "T4", "target": "return", "model": m, "feature_set": fs,
                         "paper": pic, "recomputed": r.mean_ic, "nw_t": r.nw_t,
                         "value_match": abs(round(r.mean_ic, 4) - pic) < 1e-9 and abs(round(r.nw_t, 2) - pt) < 1e-9,
                         "dagger_match": True, "lift_recomputed": r.mean_ic - a})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "S1_tables_check.csv", index=False)
    for t in ("T3", "T4"):
        g = df[df.table == t]
        emit(f"{t}: values match {int(g.value_match.sum())}/{len(g)}; t>3 marks match "
             f"{int(g.dagger_match.sum())}/{len(g)}")


# ---------------------------------------------------------------- S2 counts / SE
def s2_counts():
    head("S2  t>3 counts and the median HAC SE behind the 0.019 threshold")
    g3 = RT[RT.model.isin(M3)]
    gv = VT[VT.model.isin(M3)]
    emit(f"return grid cells {len(g3)}; t>3: {int((g3.nw_t > 3).sum())}; "
         f"vol grid cells {len(gv)}; t>3: {int((gv.nw_t > 3).sum())}; max vol t {gv.nw_t.max():.1f}")
    se_all, se_3 = RT.nw_se.median(), g3.nw_se.median()
    emit(f"median NW SE: all {len(RT)} cells (incl. {int((~RT.model.isin(M3)).sum())} FFNN/TFT) "
         f"{se_all:.4f} -> 3xSE {3 * se_all:.4f}; OLS/EN/XGB 162 cells {se_3:.4f} -> 3xSE {3 * se_3:.4f}")
    for lift in (0.002, 0.003, 0.004):
        emit(f"  names needed for lift {lift}: {60 * (3 * se_all / lift) ** 2:,.0f} (paper SE) vs "
             f"{60 * (3 * se_3 / lift) ** 2:,.0f} (162-cell SE)")
    pd.DataFrame([{"cells": "all_198", "median_se": se_all, "threshold": 3 * se_all},
                  {"cells": "ols_en_xgb_162", "median_se": se_3, "threshold": 3 * se_3}]
                 ).to_csv(OUT / "S2_se_threshold.csv", index=False)


# ---------------------------------------------------------------- S3 constant forecasts
def s3_constant_predictions():
    head("S3  Windows with constant (null-model) predictions: IC is NaN and silently dropped")
    by_model = RT.groupby("model").agg(cells=("n_nan", "size"), cells_with_nan=("n_nan", lambda s: int((s > 0).sum())),
                                       nan_windows=("n_nan", "sum"), windows=("n_windows", "sum"))
    emit(by_model.to_string())
    emit(f"volatility grid NaN windows: {int(VT.n_nan.sum())} of {int(VT.n_windows.sum())}")
    both = RT.merge(RT0[KEYS + ["mean_ic", "nw_t"]], on=KEYS, suffixes=("", "_nan0"))
    both.to_csv(OUT / "S3_constant_prediction_windows.csv", index=False)
    g3 = both[both.model.isin(M3)]
    emit(f"return cells with t>3: {int((g3.nw_t > 3).sum())} as published (NaN dropped); "
         f"{int((g3.nw_t_nan0 > 3).sum())} if a constant forecast counts as IC = 0")
    for _, r in g3[(g3.nw_t > 3) | (g3.nw_t_nan0 > 3)].sort_values("nw_t", ascending=False).iterrows():
        emit(f"  {r.model}/{r.feature_set}/{r.scheme}/h{r.horizon}: valid windows {r.n_windows - r.n_nan}/{r.n_windows}; "
             f"published IC {r.mean_ic:+.4f} t {r.nw_t:.2f} | NaN->0 IC {r.mean_ic_nan0:+.4f} t {r.nw_t_nan0:.2f}")
    p = OUT / "en_D_rolling_h5_predictions.parquet"
    if not p.exists():
        emit("  (run t0_refits.py first for the long/short decomposition)")
        return
    pr = pd.read_parquet(p)
    const_dates = set(pr.loc[pr.groupby("window").pred.transform("std") < 1e-12, "date"])
    daily, _ = backtest_long_short_deciles(
        dates=pr.date.to_numpy(), tickers=pr.ticker.to_numpy(), actual_returns=pr.y_raw.to_numpy(),
        predicted_returns=pr.pred.to_numpy(), quoted_spread=None, top_n=6, bottom_n=6,
        effective_spread_fraction=0.15, short_fee_bps_annual=50.0, horizon=5, rebalance_every=5)
    daily["constant_window"] = daily.date.isin(const_dates)
    rows = []
    for k, g in daily.groupby("constant_window"):
        r = g.portfolio_return
        rows.append({"constant_window": k, "periods": len(r), "mean": r.mean(),
                     "arith_sharpe": r.mean() / r.std() * np.sqrt(252 / 5), "growth": (1 + r).prod()})
        emit(f"  headline L/S periods in {'constant' if k else 'informative'} windows: {len(r)}, "
             f"arithmetic Sharpe {rows[-1]['arith_sharpe']:+.3f}, growth x{rows[-1]['growth']:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "S3_headline_ls_by_window_type.csv", index=False)
    tie = pr[pr.date == min(const_dates)].sort_values("pred", ascending=False)
    emit(f"  tie-broken book on {min(const_dates).date()}: long {tie.head(6).ticker.tolist()} "
         f"short {tie.tail(6).ticker.tolist()}")


# ---------------------------------------------------------------- S4 Appendix A
def s4_appendix_a():
    head("S4  Appendix A (OLS, rolling 9m/3m, h=5)")
    paper = {"A": (0.001813, 0.238), "repr_svi": (0.001812, 0.238), "D": (0.010418, 1.671),
             "repr_grid": (0.010385, 1.664), "B": (0.012799, 1.864), "C": (0.012799, 1.864)}
    for fs, (pic, pt) in paper.items():
        r = cell(RT, "OLS", fs, "rolling_9m_3m", 5)
        emit(f"  {fs:10s} IC {r.mean_ic:.6f} (paper {pic:.6f}) t {r.nw_t:.3f} (paper {pt:.3f})")
    for tgt, wdf, df in (("return", W, RT), ("vol", VW, VT)):
        b = wdf[(wdf.model == "OLS") & (wdf.feature_set == "B") & (wdf.scheme == "rolling_9m_3m") & (wdf.horizon == 5)].sort_values("window")
        c = wdf[(wdf.model == "OLS") & (wdf.feature_set == "C") & (wdf.scheme == "rolling_9m_3m") & (wdf.horizon == 5)].sort_values("window")
        d = np.abs(b.ic.to_numpy() - c.ic.to_numpy())
        emit(f"  {tgt}: B mean {b.ic.mean():.9f}, C mean {c.ic.mean():.9f}; per-window max |B-C| {d.max():.2e}; "
             f"identical windows {int((d == 0).sum())}/{len(d)}")
    emit("  mechanism: see t0_refits.txt / ols_rank_check.csv (float32 rank 1-8 of 80-114 columns)")


# ---------------------------------------------------------------- S5 volatility lift
def s5_vol_lift():
    head("S5  Volatility lift of D over A (EN+XGB, 2 schemes x 3 horizons)")
    rows = []
    for fs in ["B", "C", "D", "candidate_6", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]:
        for tgt, df in (("return", RT), ("vol", VT)):
            d = df[df.model.isin(["ElasticNet", "XGBoost"])].pivot_table(index=["model", "scheme", "horizon"],
                                                                         columns="feature_set", values="mean_ic")
            L = (d[fs] - d["A"]).dropna()
            rows.append({"target": tgt, "feature_set": fs, "lift": L.mean(), "n_cells": len(L),
                         "se_ddof0": L.std(ddof=0) / np.sqrt(len(L)), "se_ddof1": L.std(ddof=1) / np.sqrt(len(L))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "S5_lifts.csv", index=False)
    v = df[(df.target == "vol") & (df.feature_set == "D")].iloc[0]
    emit(f"  vol D-A lift {v.lift:.4f}; SE ddof0 {v.se_ddof0:.4f} (ratio {v.lift / v.se_ddof0:.1f}); "
         f"SE ddof1 {v.se_ddof1:.4f} (ratio {v.lift / v.se_ddof1:.1f})")
    pv = VW[VW.model.isin(["ElasticNet", "XGBoost"])].pivot_table(index=KEYS[:1] + KEYS[2:] + ["window"],
                                                                  columns="feature_set", values="ic")
    ts = [_nw_tstat(g.to_numpy())[2] for _, g in (pv["D"] - pv["A"]).dropna().groupby(level=[0, 1, 2])]
    emit(f"  per-cell paired HAC t of per-window (D - A): min {min(ts):.1f}, median {np.median(ts):.1f}, max {max(ts):.1f}")
    r = df[(df.target == "return") & (df.feature_set == "D")].iloc[0]
    emit(f"  return D-A lift {r.lift:+.4f}; repr_svi vol lift "
         f"{df[(df.target == 'vol') & (df.feature_set == 'repr_svi')].lift.iloc[0]:.4f}")


# ---------------------------------------------------------------- S6 meta-analysis
def meta(wdf: pd.DataFrame) -> dict:
    wdf = wdf.copy()
    wdf["test_start"] = pd.to_datetime(wdf["test_start"])
    reps = ["B", "C", "D", "candidate_6", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]
    pivot = wdf.pivot_table(index=["model", "scheme", "horizon", "window"], columns="feature_set",
                            values="ic").dropna(subset=["A"])
    quarter = (wdf.drop_duplicates(["scheme", "horizon", "window"]).set_index(["scheme", "horizon", "window"])
               ["test_start"].dt.to_period("Q").astype(str).to_dict())
    rows = []
    for rep in reps:
        for (m, s, h, w), r in pivot[[rep, "A"]].dropna().iterrows():
            rows.append({"rep": rep, "model": m, "quarter": quarter.get((s, int(h), int(w))),
                         "finewin": f"{s}|{h}|{w}", "lift": r[rep] - r["A"]})
    L = pd.DataFrame(rows)
    ci, p, nq = rx.block_bootstrap(L, "quarter")
    cif, pf, _ = rx.block_bootstrap(L, "finewin")
    return {"pooled": L.lift.mean(), "ci_lo": ci[0], "ci_hi": ci[1], "p": p, "quarters": nq,
            "lifts": len(L), "fine_ci_lo": cif[0], "fine_ci_hi": cif[1], "fine_p": pf}


def s6_meta():
    head("S6  Meta-analysis (calendar-quarter joint bootstrap, seed 7)")
    rows = [{"models": "all (as published, incl. FFNN/TFT)"} | meta(W),
            {"models": "OLS/EN/XGB only"} | meta(W[W.model.isin(M3)])]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "S6_meta_analysis.csv", index=False)
    for _, r in df.iterrows():
        emit(f"  {r.models:38s} lift {r.pooled:+.4f} CI [{r.ci_lo:+.4f},{r.ci_hi:+.4f}] p {r.p:.3f} "
             f"({r.lifts} lifts, {r.quarters} quarters); fine blocks CI [{r.fine_ci_lo:+.4f},{r.fine_ci_hi:+.4f}] p {r.fine_p:.3f}")
    emit("  note: per-window lifts where A or the representation has a NaN (constant-forecast) IC are dropped")


# ---------------------------------------------------------------- S7 grid vs raw grid
def s7_grid_vs_raw():
    head("S7  repr_grid (SVI-evaluated) vs repr_grid_raw (raw-quote grid)")
    g = RT[RT.model.isin(M3)].pivot_table(index=["model", "scheme", "horizon"], columns="feature_set", values="mean_ic")
    d = (g["repr_grid"] - g["repr_grid_raw"]).dropna()
    emit(f"  benchmark IC, 18 cells: max |diff| {d.abs().max():.4f} at {d.abs().idxmax()}; median signed {d.median():+.4f}; "
         f"grid higher in {int((d > 0).sum())}/{len(d)}; XGB-only median {d.xs('XGBoost').median():+.4f}")
    txt = (TAB / "referee_fixes.txt").read_text()
    dd = [float(x) for x in re.findall(r"diff\(SVI-raw\)=([+-]\d+\.\d+)", txt)]
    emit(f"  referee_fixes.txt (XGBoost, daily-IC definition, 6 cells): diffs {dd} -> median {np.median(dd):+.4f} "
         f"(this is the paper's 0.0008)")


# ---------------------------------------------------------------- S8 regime
def mbb(high: pd.Series, low: pd.Series, dates: np.ndarray, L: int, B: int = 4000, seed: int = 1):
    n = len(dates)
    H, Lo = high.reindex(dates).to_numpy(), low.reindex(dates).to_numpy()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    hm, dm = np.empty(B), np.empty(B)
    for b in range(B):
        idx = (rng.integers(0, n - L + 1, size=nb)[:, None] + np.arange(L)).ravel()[:n]
        hm[b] = np.nanmean(H[idx])
        dm[b] = hm[b] - np.nanmean(Lo[idx])
    return np.percentile(hm, 2.5), np.percentile(hm, 97.5), 2 * min((dm <= 0).mean(), (dm >= 0).mean())


def s8_regime():
    head("S8  Regime conditioning: published i.i.d.-day bootstrap vs moving-block bootstrap")
    oos = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet",
                          filters=[("feature_set", "==", "D")])
    oos["date"] = pd.to_datetime(oos["date"])
    panel = pd.read_parquet(DATA / "features/resolution_1_scalar/panel_unnormalized.parquet",
                            columns=["ticker", "date", "feat_38", "feat_12", "feat_43"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.rename(columns={"feat_38": "vix", "feat_12": "vrp", "feat_43": "days_earn"})
    g = panel.dropna(subset=["vrp"]).groupby("date")["vrp"]
    emit(f"  VRP (feat_12) dispersion: median within-date SD {g.std().median():.4f} vs SD of daily means "
         f"{g.mean().std():.4f}  -> the split is per ticker-day, not per date")
    emit(f"  VIX (feat_38) max within-date SD {panel.groupby('date').vix.std().max():.1f} -> per-date variable")
    thr = rx.causal_vrp_threshold(panel)
    rows = []
    for scheme in ["expanding", "rolling_9m_3m"]:
        for h in [3, 5]:
            d = oos[(oos.scheme == scheme) & (oos.horizon == h)].merge(panel, on=["ticker", "date"], how="left")
            d["thr"] = d["date"].map(thr)
            dates = np.sort(d.date.unique())
            for key, hi, lo in (("VIX", d[d.vix >= 25], d[d.vix <= 15]),
                                ("VRP", d[d.vrp > d.thr], d[d.vrp <= d.thr]),
                                ("EARN", d[d.days_earn <= 7], d[d.days_earn > 7])):
                ih, il = rx.daily_ic_series(hi), rx.daily_ic_series(lo)
                mh, cih, ml, _, diff, _, p_iid = rx.bootstrap_diff(ih, il)
                rec = {"scheme": scheme, "horizon": h, "conditioner": key,
                       "unconditional_daily_ic": rx.daily_ic_series(d).mean(), "high_ic": mh, "low_ic": ml,
                       "diff": diff, "iid_high_ci_lo": cih[0], "iid_high_ci_hi": cih[1], "iid_diff_p": p_iid}
                for L in (10, 21, 63):
                    lo_, hi_, p = mbb(ih, il, dates, L)
                    rec |= {f"mbb{L}_high_ci_lo": lo_, f"mbb{L}_high_ci_hi": hi_, f"mbb{L}_diff_p": p}
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "S8_regime_bootstrap.csv", index=False)
    bonf = 0.05 / 12
    for key in ("VRP", "VIX", "EARN"):
        s = df[df.conditioner == key]
        msg = (f"  {key}: high CI>0 iid {int((s.iid_high_ci_lo > 0).sum())}/4, "
               + ", ".join(f"L={L} {int((s[f'mbb{L}_high_ci_lo'] > 0).sum())}/4" for L in (10, 21, 63))
               + f" | diff p<Bonferroni iid {int((s.iid_diff_p < bonf).sum())}/4, "
               + ", ".join(f"L={L} {int((s[f'mbb{L}_diff_p'] < bonf).sum())}/4" for L in (10, 21, 63)))
        emit(msg)
    v = df[df.conditioner == "VIX"]
    emit(f"  VIX diff p-values (paper says all > 0.8): {[round(x, 3) for x in v.iid_diff_p]}")
    u = df.drop_duplicates(["scheme", "horizon"])
    emit(f"  unconditional mean daily IC, XGB/D, 4 cells: {[round(x, 4) for x in u.unconditional_daily_ic]} "
         f"-> mean {u.unconditional_daily_ic.mean():+.4f} (paper: 'near zero')")
    for _, r in df[df.conditioner == "VRP"].iterrows():
        emit(f"    VRP {r.scheme:13s} h{r.horizon}: high {r.high_ic:+.4f} iid CI [{r.iid_high_ci_lo:+.4f},{r.iid_high_ci_hi:+.4f}] "
             f"p {r.iid_diff_p:.3f} | L=21 CI [{r.mbb21_high_ci_lo:+.4f},{r.mbb21_high_ci_hi:+.4f}] p {r.mbb21_diff_p:.3f}")


# ---------------------------------------------------------------- S9 PBO
def pbo_local(scheme, h, S=8, conv="published"):
    ls = pd.read_parquet(TAB / "full_matrix_ls_daily.parquet")
    ls = ls[(ls.scheme == scheme) & (ls.horizon == h)].copy()
    ls["trial"] = ls.model + "/" + ls.feature_set
    R = ls.pivot_table(index="date", columns="trial", values="portfolio_return").dropna().to_numpy()
    T, N = R.shape
    blocks = np.array_split(np.arange(T), S)
    from itertools import combinations
    logits = []
    for is_idx in combinations(range(S), S // 2):
        isr = np.concatenate([blocks[b] for b in is_idx])
        oos = np.concatenate([blocks[b] for b in range(S) if b not in is_idx])
        def sr(rows):
            x = R[rows]; mu = x.mean(0); sd = x.std(0, ddof=1)
            return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
        s_is, s_oos = sr(isr), sr(oos)
        below = (s_oos < s_oos[int(np.argmax(s_is))]).sum()
        w = below / N if conv == "published" else (below + 1) / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    return float((np.array(logits) <= 0).mean()), N


def s9_pbo():
    head("S9  PBO (CSCV, S=8, 70 splits) by scheme x horizon group")
    rows = []
    for s in ["expanding", "rolling_9m_3m"]:
        for h in [1, 3, 5]:
            pub = pbo_cscv(scheme=s, horizon=h)["pbo"]
            loc, n = pbo_local(s, h)
            std, _ = pbo_local(s, h, conv="bailey")
            rows.append({"scheme": s, "horizon": h, "trials": n, "pbo_published_code": pub,
                         "pbo_local_check": loc, "pbo_rank_over_N_plus_1": std})
            emit(f"  {s:13s} h{h}: trials {n}, PBO {pub:.3f} (rank/(N+1) convention {std:.3f})"
                 + ("   <- value printed in the paper" if (s, h) == ("expanding", 3) else "")
                 + ("   <- headline cell's group" if (s, h) == ("rolling_9m_3m", 5) else ""))
    pd.DataFrame(rows).to_csv(OUT / "S9_pbo_by_group.csv", index=False)


# ---------------------------------------------------------------- S10 Sharpe / SR0
def sr0(sr):
    sr = np.asarray(sr, float); sr = sr[np.isfinite(sr)]; n = len(sr)
    e = 0.5772156649015329
    return float(np.sqrt(np.var(sr, ddof=1)) * ((1 - e) * norm.ppf(1 - 1 / n) + e * norm.ppf(1 - 1 / (n * np.e)))), n


def s10_sharpe():
    head("S10  Sharpe convention and the expected-maximum benchmark SR0")
    s = pd.read_csv(TAB / "full_matrix_strategy_summary.csv")
    s = s[(s.strategy == "long_short_decile") & s.cost_scheme.astype(str).str.contains("15")].copy()
    ppy = 252 / s.rebalance_every
    s["sharpe_arith"] = s.mean_return * ppy / s.annualized_vol
    hl = s[(s.model == "ElasticNet") & (s.feature_set == "D") & (s.scheme == "rolling_9m_3m") & (s.horizon == 5)].iloc[0]
    emit(f"  headline EN/D/rolling/h5: Sharpe as coded (geometric) {hl.sharpe:.4f}; arithmetic {hl.sharpe_arith:.4f}; "
         f"final $ {1000 * (1 + hl.total_return):,.2f}")
    rows = []
    for fam, mask in (("all 198", s.model.notna()), ("no TFT (192)", s.model != "TFT"),
                      ("OLS/EN/XGB (162)", s.model.isin(M3))):
        for conv in ("sharpe", "sharpe_arith"):
            b, n = sr0(s.loc[mask, conv])
            surv = int((s.loc[mask, conv] > b).sum())
            rows.append({"family": fam, "convention": conv, "N": n, "SR0": b, "n_clear": surv,
                         "headline_excess": hl[conv] - b})
            emit(f"  {fam:17s} {'geometric' if conv == 'sharpe' else 'arithmetic':10s}: N {n}, SR0 {b:.4f}, "
                 f"configs above SR0 {surv}, headline excess {hl[conv] - b:+.4f}")
    pd.DataFrame(rows).to_csv(OUT / "S10_sr0.csv", index=False)
    emit(f"  strategy rows by model (15% cost): {s.model.value_counts().to_dict()}")


# ---------------------------------------------------------------- S11 data
def s11_data():
    head("S11  Data-layer checks")
    crsp = pd.read_parquet(DATA / "raw/wrds/crsp/stock_daily.parquet")
    crsp["date"] = pd.to_datetime(crsp["date"])
    seg = (crsp.groupby(["ticker", "permno"]).agg(first=("date", "min"), last=("date", "max"), n=("date", "size"),
                                                   median_abs_prc=("prc", lambda x: x.abs().median())).reset_index())
    multi = seg[seg.ticker.isin(seg.groupby("ticker").permno.nunique().loc[lambda s: s > 1].index)]
    multi.to_csv(OUT / "S11_ticker_permno_collisions.csv", index=False)
    emit(f"  CRSP rows {len(crsp):,}; tickers {crsp.ticker.nunique()}; dates {crsp.date.min().date()}..{crsp.date.max().date()}")
    emit(f"  tickers mapped to >1 PERMNO: {sorted(multi.ticker.unique())}; PERMNOs with >1 ticker: "
         f"{int((crsp.groupby('permno').ticker.nunique() > 1).sum())}")
    for _, r in multi.iterrows():
        emit(f"    {r.ticker} permno {r.permno}: {r['first'].date()}..{r['last'].date()} ({r.n} rows, median |prc| {r.median_abs_prc:.2f})")
    cnt = crsp.groupby("ticker").size()
    emit(f"  rows per ticker: {cnt.value_counts().to_dict()}")
    c = crsp.sort_values(["ticker", "date"]).copy()
    c["pr"] = c.prc.abs() / c.groupby("ticker").prc.shift().abs()
    c["implied"] = c.pr / (1 + c.ret)
    ev = c[np.abs(np.log(c.implied)) > np.log(1.2)][["ticker", "date", "permno", "ret", "pr", "implied"]]
    ev.to_csv(OUT / "S11_split_or_identifier_events.csv", index=False)
    emit(f"  days where |price ratio| disagrees with (1+ret) by >20% (splits / identifier changes): {len(ev)}")
    for _, r in ev.iterrows():
        emit(f"    {r.ticker:5s} {r.date.date()} price ratio {r.pr:.3f} vs 1+ret {1 + r.ret:.3f}")
    panel = pd.read_parquet(DATA / "features/resolution_1_scalar/panel_unnormalized.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    stocks = panel[~panel.ticker.isin(ETFS)]
    emit(f"  feat_50 == feat_01: stocks {np.isclose(stocks.feat_50, stocks.feat_01).mean():.4f}, "
         f"ETFs {np.isclose(panel[panel.ticker.isin(ETFS)].feat_50, panel[panel.ticker.isin(ETFS)].feat_01).mean():.4f}")
    sf = pd.read_parquet(DATA / "features/stock_features.parquet", columns=["ticker", "date", "feat_28", "feat_29", "feat_36"])
    sf["date"] = pd.to_datetime(sf["date"])
    nv = sf[(sf.ticker == "NVDA") & sf.date.between("2024-06-06", "2024-06-12")].merge(
        panel[["ticker", "date", "feat_28"]], on=["ticker", "date"], suffixes=("_raw", "_panel"))
    emit("  NVDA around its 10:1 split (raw = stock_features.parquet, panel = winsorised):")
    for _, r in nv.iterrows():
        emit(f"    {r.date.date()} feat_28 raw {r.feat_28_raw:+.3f} panel {r.feat_28_panel:+.3f}; feat_29 raw {r.feat_29:+.3f}; feat_36 raw {r.feat_36:+.3f}")
    goog = panel[panel.ticker == "GOOG"].feat_43
    earn = pd.read_parquet(DATA / "raw/wrds/compustat/earnings_dates.parquet")
    tcol = [x for x in earn.columns if "tic" in x.lower()][0]
    emit(f"  GOOG feat_43 distinct values {goog.nunique()} ({goog.unique()[:3]}); Compustat tickers like GOOG: "
         f"{sorted(t for t in earn[tcol].astype(str).unique() if 'GOOG' in t)}")
    splits = pd.read_parquet(DATA / "splits/split_indices.parquet")
    splits["date"] = pd.to_datetime(splits["date"])
    model_keys = splits[["ticker", "date"]]
    thin_rows = []
    for f in sorted((DATA / "interim/merged").glob("*.parquet")):
        m = pd.read_parquet(f, columns=[c for c in ("date", "n_contracts", "thin_options_day") if c in pq.read_schema(f).names])
        if "thin_options_day" in m.columns:
            m["date"] = pd.to_datetime(m["date"])
            m["ticker"] = f.stem
            in_model = m.merge(model_keys, on=["ticker", "date"])
            thin_rows.append({"ticker": f.stem, "thin_days": int(m.thin_options_day.sum()), "days": len(m),
                              "thin_days_in_modelling_panel": int(in_model.thin_options_day.sum())})
    th = pd.DataFrame(thin_rows)
    th.to_csv(OUT / "S11_thin_option_days.csv", index=False)
    emit(f"  ticker-days with fewer than 50 contracts (flagged, never filtered): {int(th.thin_days.sum()):,} "
         f"in data/interim/merged, {int(th.thin_days_in_modelling_panel.sum()):,} of them in the modelling panel, "
         f"across {int((th.thin_days > 0).sum())} tickers")
    cov = []
    for f in sorted((DATA / "raw/github_options").glob("*_options.parquet")):
        t = pd.to_datetime(pq.read_table(f, columns=["date"]).column("date").to_pandas())
        cov.append({"ticker": f.stem.replace("_options", "").upper(), "first": t.min().date(), "last": t.max().date()})
    cov = pd.DataFrame(cov)
    cov.to_csv(OUT / "S11_option_snapshot_coverage.csv", index=False)
    emit(f"  option snapshot: last date {cov['last'].min()}..{cov['last'].max()}; tickers starting after 2013-01-02: "
         f"{cov[cov['first'] > pd.Timestamp('2013-01-02').date()].ticker.tolist()}")
    tr = splits[splits.split == "train"].date
    emit(f"  panel rows {len(panel):,}; modelling rows after split join {len(splits):,} "
         f"(dates dropped: {panel.date.nunique() - splits.date.nunique()}); winsorise/impute 'train' block "
         f"{tr.min().date()}..{tr.max().date()}")
    cal = np.sort(panel.date.unique())
    pos = pd.Series(np.arange(len(cal)), index=cal)
    mp = panel.merge(splits[["ticker", "date"]], on=["ticker", "date"]).sort_values(["ticker", "date"])
    spliced = 0
    for _, g in mp.groupby("ticker"):
        ix = pos.loc[g.date].to_numpy()
        if len(ix) >= 5:
            spliced += int(((ix[4:] - ix[:-4]) != 4).sum())
    emit(f"  5-day RV windows in run_vol_matrix that splice non-adjacent days (built after the split join): {spliced}")
    import yaml
    fd = yaml.safe_load(open(ROOT / "config" / "feature_definitions.yaml"))
    surf = [c for c in fd["feature_sets"]["D"]["features"] if not str(c).startswith("feat_")]
    emit(f"  Set-D surface columns (not winsorised or imputed; NaN -> 0 in run_full_matrix): {len(surf)}, "
         f"NaN share median {panel[surf].isna().mean().median():.3f}, max {panel[surf].isna().mean().max():.3f}")


# ---------------------------------------------------------------- S12 injection
def s12_injection():
    head("S12  Signal-injection calibration: target delta vs realised oracle IC")
    si = pd.read_csv(TAB / "signal_injection.csv")
    g = si.groupby("delta").oracle_ic.mean()
    for d, o in g.items():
        emit(f"  delta {d:.3f} -> oracle IC {o:.4f}" + (f"  (ratio {o / d:.2f})" if d > 0 else ""))
    g.to_csv(OUT / "S12_injection_oracle.csv")


def s13_env():
    head("S13  Environment")
    emit(f"  python {platform.python_version()} on {platform.platform()}")
    for p in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "pyarrow", "matplotlib", "torch"):
        try:
            emit(f"  {p} {importlib.import_module(p).__version__}")
        except Exception as e:  # noqa: BLE001
            emit(f"  {p} not importable ({e.__class__.__name__})")


if __name__ == "__main__":
    t0 = time.time()
    for f in (s1_tables, s2_counts, s3_constant_predictions, s4_appendix_a, s5_vol_lift, s6_meta,
              s7_grid_vs_raw, s8_regime, s9_pbo, s10_sharpe, s11_data, s12_injection, s13_env):
        f()
    emit(f"\nrun time {time.time() - t0:.0f} s")
    (OUT / "t0_checks.txt").write_text("\n".join(LINES) + "\n")
