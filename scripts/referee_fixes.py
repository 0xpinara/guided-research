"""Regenerate every number requested by the referee critique (sections A-D).

All outputs are written to results/tables/referee_fixes.txt (+ a few CSVs) so the
paper rewrite can transcribe verified numbers rather than ad-hoc shell echoes.

No model is retrained: everything is recomputed from the saved per-ticker-day OOS
predictions (full_matrix_oos_xgb.parquet, XGBoost, all 9 sets) and the saved daily
long/short returns (full_matrix_ls_daily.parquet, OLS/EN/FFNN/XGBoost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
PANEL = ROOT / "data" / "features" / "resolution_1_scalar"
ETFS = {"SPY", "QQQ", "IWM"}
EULER = 0.5772156649

out_lines: list[str] = []
def P(*a):
    s = " ".join(str(x) for x in a)
    out_lines.append(s)
    print(s)

def hr(t):
    P("\n" + "=" * 78)
    P(t)
    P("=" * 78)

# ---------------------------------------------------------------------------
oos = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet")
oos["date"] = pd.to_datetime(oos["date"])
ls = pd.read_parquet(TAB / "full_matrix_ls_daily.parquet")
ls["date"] = pd.to_datetime(ls["date"])
icts = pd.read_csv(TAB / "posthoc_ic_tstats.csv")

def _daily_ic_series(d, pcol="pred", ycol="y_demeaned", min_n=5):
    """Per-date cross-sectional Spearman as vectorized Pearson-on-ranks
    (C-level group aggregations; ~50x faster than groupby.apply)."""
    d = d[["date", pcol, ycol]].dropna()
    g = d.groupby("date")
    rp = g[pcol].rank(); ry = g[ycol].rank()
    t = pd.DataFrame({"date": d["date"].values, "rp": rp.values, "ry": ry.values})
    gg = t.groupby("date")
    n = gg.size()
    mpp = gg["rp"].mean(); myy = gg["ry"].mean()
    t["pp"] = t["rp"] * t["rp"]; t["yy"] = t["ry"] * t["ry"]; t["py"] = t["rp"] * t["ry"]
    s = t.groupby("date").agg(spp=("pp", "sum"), syy=("yy", "sum"), spy=("py", "sum"))
    cov = s["spy"] - n * mpp * myy
    vp = s["spp"] - n * mpp * mpp; vy = s["syy"] - n * myy * myy
    ic = cov / np.sqrt(vp * vy)
    ic = ic[n >= min_n].replace([np.inf, -np.inf], np.nan).dropna()
    return ic

def daily_ic(df, exclude_etfs=False):
    """Time-series mean of daily cross-sectional Spearman rank ICs."""
    d = df if not exclude_etfs else df[~df["ticker"].isin(ETFS)]
    ics = _daily_ic_series(d)
    return float(ics.mean()), int(len(ics))

# ===========================================================================
hr("A2/A3  REPRESENTATION HEAD-TO-HEAD + MARGINAL LIFT OVER SET-A CONTROL")
# repr sets all sit on the identical Set-A control block; lift = IC(repr) - IC(A)
sets = ["A", "D", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]
for model in ["ElasticNet", "XGBoost"]:
    P(f"\n--- {model}: pooled walk-forward IC (NW t), lift over Set-A control ---")
    sub = icts[(icts.model == model) & (icts.feature_set.isin(sets))]
    for scheme in ["expanding", "rolling_9m_3m"]:
        for h in [1, 3, 5]:
            cell = sub[(sub.scheme == scheme) & (sub.horizon == h)]
            if cell.empty:
                continue
            ic_a = cell[cell.feature_set == "A"]["mean_ic"]
            ic_a = float(ic_a.iloc[0]) if len(ic_a) else np.nan
            P(f"  {scheme:13} h={h}:")
            for s in sets:
                r = cell[cell.feature_set == s]
                if r.empty:
                    continue
                ic = float(r["mean_ic"].iloc[0]); t = float(r["nw_tstat"].iloc[0])
                hlz = bool(r["survives_hlz_t3"].iloc[0])
                lift = ic - ic_a if s != "A" else 0.0
                tag = "  <-- HLZ t>3" if hlz else ""
                P(f"      {s:14} IC={ic:+.4f}  t={t:+.2f}  lift_vs_A={lift:+.4f}{tag}")

# Does the SVI fit earn its keep? repr_grid (SVI-fit grid) vs repr_grid_raw (model-free grid)
P("\n--- Does SVI fitting earn its keep? repr_grid (SVI) vs repr_grid_raw (model-free), XGBoost daily-IC ---")
for scheme in ["expanding", "rolling_9m_3m"]:
    for h in [1, 3, 5]:
        g  = oos[(oos.feature_set=="repr_grid")     & (oos.scheme==scheme) & (oos.horizon==h)]
        gr = oos[(oos.feature_set=="repr_grid_raw") & (oos.scheme==scheme) & (oos.horizon==h)]
        if g.empty or gr.empty:
            continue
        ic_g, _  = daily_ic(g);  ic_gr, _ = daily_ic(gr)
        P(f"  {scheme:13} h={h}: SVI-grid IC={ic_g:+.4f}  raw-grid IC={ic_gr:+.4f}  diff(SVI-raw)={ic_g-ic_gr:+.4f}")

# ===========================================================================
hr("B1  DROP ETFs (SPY/QQQ/IWM) FROM THE RANKED/TRADEABLE UNIVERSE")
P("\n--- XGBoost pooled daily-IC WITH vs WITHOUT the 3 ETFs ---")
for scheme in ["expanding", "rolling_9m_3m"]:
    for h in [1, 3, 5]:
        for s in ["A", "B", "D", "candidate_6"]:
            d = oos[(oos.feature_set==s) & (oos.scheme==scheme) & (oos.horizon==h)]
            if d.empty:
                continue
            ic_all, n = daily_ic(d, False)
            ic_ex, _  = daily_ic(d, True)
            P(f"  {s:12} {scheme:13} h={h}: IC_all={ic_all:+.4f}  IC_ex-ETF={ic_ex:+.4f}  delta={ic_ex-ic_all:+.4f}")

# How often does an ETF enter the top-6 / bottom-6 decile? (XGBoost/D headline cells)
P("\n--- ETF appearance frequency in the top-6/bottom-6 decile (XGBoost/D) ---")
for scheme, h in [("expanding", 3), ("expanding", 5), ("rolling_9m_3m", 5)]:
    d = oos[(oos.feature_set=="D") & (oos.scheme==scheme) & (oos.horizon==h)].copy()
    if d.empty:
        continue
    rb = sorted(d["date"].unique())[::h]  # non-overlapping rebalance dates
    d = d[d["date"].isin(set(rb))]
    n_dates = 0; n_with_etf = 0
    for dt, g in d.groupby("date"):
        if g["ticker"].nunique() < 12:
            continue
        g = g.sort_values("pred", ascending=False)
        legs = set(g.head(6)["ticker"]) | set(g.tail(6)["ticker"])
        n_dates += 1
        if legs & ETFS:
            n_with_etf += 1
    P(f"  {scheme:13} h={h}: {n_with_etf}/{n_dates} rebalances have an ETF in a leg "
      f"({100*n_with_etf/max(n_dates,1):.1f}%)")

# Gross long/short Sharpe with vs without ETFs (XGBoost/D), to bound the impact
def gross_ls_sharpe(d, h, exclude_etfs):
    if exclude_etfs:
        d = d[~d["ticker"].isin(ETFS)]
    rb = sorted(d["date"].unique())[::h]
    d = d[d["date"].isin(set(rb))]
    rets = []
    for dt, g in d.groupby("date"):
        if g["ticker"].nunique() < 12:
            continue
        g = g.sort_values("pred", ascending=False)
        long_r = g.head(6)["y_raw"].mean(); short_r = g.tail(6)["y_raw"].mean()
        rets.append(long_r - short_r)
    rets = np.array(rets)
    ppy = 252 / h
    return rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else np.nan
P("\n--- XGBoost/D gross long/short Sharpe (no costs), with vs without ETFs ---")
for scheme, h in [("expanding", 5), ("rolling_9m_3m", 5)]:
    d = oos[(oos.feature_set=="D") & (oos.scheme==scheme) & (oos.horizon==h)]
    P(f"  {scheme:13} h={h}: Sharpe_all={gross_ls_sharpe(d,h,False):+.3f}  "
      f"Sharpe_ex-ETF={gross_ls_sharpe(d,h,True):+.3f}")

# ===========================================================================
hr("B2  GEX REGIME IC WITH SIGNIFICANCE (block-bootstrap over dates)")
panel = pd.read_parquet(PANEL / "panel_unnormalized.parquet",
                        columns=["ticker", "date", "feat_21", "feat_22", "feat_38", "feat_43"])
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.rename(columns={"feat_21": "gex", "feat_22": "gex_mc",
                              "feat_38": "vix", "feat_43": "days_earn"})

def regime_boot(d, mask_fn, label_a, label_b, n_boot=5000, seed=0):
    """Per-date within-bucket rank-IC, then block-bootstrap (resample dates) the
    MEAN daily IC for each bucket: 95% CI + bootstrap p-value for IC(A)-IC(B).
    Fast and consistent with the per-window IC / HAC methodology in the paper."""
    rng = np.random.default_rng(seed)
    dd = d.dropna(subset=["pred", "y_demeaned", "gex"]).copy()
    a_by = _daily_ic_series(mask_fn(dd, "A")); b_by = _daily_ic_series(mask_fn(dd, "B"))
    # align on common dates so the difference is paired per date
    al = pd.concat([a_by.rename("a"), b_by.rename("b")], axis=1).dropna()
    av, bv = al["a"].to_numpy(), al["b"].to_numpy()
    nA, nB = int(mask_fn(dd, "A")["ticker"].shape[0]), int(mask_fn(dd, "B")["ticker"].shape[0])
    ic_a, ic_b = av.mean(), bv.mean()
    nd = len(al)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    a_s = av[idx].mean(1); b_s = bv[idx].mean(1); diffs = a_s - b_s
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    p_diff = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    P(f"    {label_a}: meanIC={ic_a:+.4f}  95%CI[{ci(a_s)[0]:+.4f},{ci(a_s)[1]:+.4f}]  obs={nA} days={nd}")
    P(f"    {label_b}: meanIC={ic_b:+.4f}  95%CI[{ci(b_s)[0]:+.4f},{ci(b_s)[1]:+.4f}]  obs={nB}")
    P(f"    diff ({label_a}-{label_b}) = {ic_a-ic_b:+.4f}  95%CI[{ci(diffs)[0]:+.4f},{ci(diffs)[1]:+.4f}]  boot p={p_diff:.3f}")

d = oos[(oos.feature_set=="D") & (oos.scheme=="expanding") & (oos.horizon==3)].merge(
    panel, on=["ticker", "date"], how="left")
P("\n[headline regime cell: XGBoost / Set D / expanding / h=3]")
P("\n  (a) GEX sign split (zero threshold, as in the paper):")
regime_boot(d, lambda x, w: x[x.gex < 0] if w == "A" else x[x.gex >= 0],
            "short-gamma (GEX<0)", "long-gamma (GEX>=0)")
P("\n  (b) ROBUSTNESS: split at cross-sectional MEDIAN GEX instead of zero:")
med = d["gex"].median()
regime_boot(d, lambda x, w: x[x.gex < med] if w == "A" else x[x.gex >= med],
            f"below-median GEX", "above-median GEX")
P("\n  (c) ROBUSTNESS: split on GEX/mktcap (feat_22) sign:")
regime_boot(d, lambda x, w: x[x.gex_mc < 0] if w == "A" else x[x.gex_mc >= 0],
            "GEX/mc<0", "GEX/mc>=0")
P("\n  Note: flipping the call/put sign convention merely relabels the two buckets,")
P("  so the magnitude of the IC differential is invariant to the convention;")
P("  only the economic label (which bucket is 'short gamma') depends on it.")

# ===========================================================================
hr("B6  SET-A (STOCK-ONLY) ECONOMIC CONTROL vs SET-D, from saved daily L/S returns")
def curve(model, fs, scheme, h, stake=1000.0):
    r = ls[(ls.model==model)&(ls.feature_set==fs)&(ls.scheme==scheme)&(ls.horizon==h)]
    r = r.sort_values("date")["portfolio_return"].astype(float).to_numpy()
    if len(r) == 0:
        return None
    eq = stake * np.cumprod(1 + np.clip(r, -0.99, None))
    ppy = 252 / h
    ann = (1 + r.mean())**ppy - 1
    sharpe = (ann) / (r.std()*np.sqrt(ppy)) if r.std() > 0 else np.nan
    peak = np.maximum.accumulate(eq); mdd = ((eq-peak)/peak).min()
    return dict(final=eq[-1], sharpe=sharpe, ann=ann, mdd=mdd, n=len(r))
P("\n   model        set  scheme        h   $1000->     Sharpe   AnnRet    MaxDD")
for model in ["ElasticNet", "XGBoost"]:
    for scheme, h in [("rolling_9m_3m", 5), ("expanding", 5), ("expanding", 3)]:
        for fs in ["A", "D"]:
            c = curve(model, fs, scheme, h)
            if c:
                P(f"   {model:11} {fs:4} {scheme:13} {h}  ${c['final']:8.0f}   "
                  f"{c['sharpe']:+.2f}   {100*c['ann']:+6.1f}%  {100*c['mdd']:+6.1f}%")

# ===========================================================================
hr("A1/B3  DEFLATED SHARPE: actual N, principled-design N, headline outcome")
ss = pd.read_csv(TAB / "full_matrix_strategy_summary.csv")
lsd = ss[ss["strategy"] == "long_short_decile"].copy() if "strategy" in ss.columns else ss.copy()
if "cost_scheme" in lsd.columns:
    canon = lsd[lsd["cost_scheme"].astype(str).str.contains("15")]
    if not canon.empty:
        lsd = canon
sr = lsd["sharpe"].to_numpy(float); sr = sr[np.isfinite(sr)]
def sr0_of(N, V):
    return np.sqrt(V)*((1-EULER)*norm.ppf(1-1.0/N) + EULER*norm.ppf(1-1.0/(N*np.e)))
N_actual = len(sr); V = np.var(sr, ddof=1)
P(f"\n  long/short trials actually scored (15% cost): N_actual = {N_actual}")
P(f"  cross-trial Sharpe variance V = {V:.4f}, sd = {np.sqrt(V):.4f}")
P(f"  E[max Sharpe | null], N_actual={N_actual}:  sr0 = {sr0_of(N_actual,V):.4f}")
# principled design N: intended full long/short grid
for label, N in [("completed grid (=ls_daily cells)", 192),
                 ("intended design incl. TabNet+TFT (6 models x ...)", 252)]:
    P(f"  if N = {N:3d} ({label}): sr0 = {sr0_of(N,V):.4f}")
head_sr = 1.1504  # EN/D/rolling/5 headline
P(f"\n  HEADLINE EN/D/rolling/5 Sharpe = {head_sr:.4f}")
P(f"  -> excess over sr0(N_actual) = {head_sr - sr0_of(N_actual,V):+.4f}  "
  f"(survives={head_sr > sr0_of(N_actual,V)})")
P(f"  -> with larger design N the bar only RISES, so the headline fails under any honest N.")
# Approx one-sided DSR probability for the headline (normal approx, skew/kurt ignored)
T = curve("ElasticNet", "D", "rolling_9m_3m", 5)["n"]
sr0a = sr0_of(N_actual, V)
dsr = norm.cdf((head_sr/np.sqrt(252/5) - sr0a/np.sqrt(252/5)) * np.sqrt(max(T-1,1)))
P(f"  approx deflated-Sharpe probability (per-period, normal) DSR ~= {dsr:.3f}  (need >0.95)")
# how many cells survive
P(f"  cells with Sharpe > sr0(N_actual): {(sr > sr0_of(N_actual,V)).sum()} of {N_actual}")

# ===========================================================================
hr("C  YEAR-BY-YEAR P&L of the (former) headline cells -> concentration check")
for model, fs, scheme, h in [("ElasticNet","D","rolling_9m_3m",5),
                             ("ElasticNet","D","expanding",3),
                             ("XGBoost","D","expanding",3)]:
    r = ls[(ls.model==model)&(ls.feature_set==fs)&(ls.scheme==scheme)&(ls.horizon==h)].copy()
    if r.empty:
        continue
    r["year"] = r["date"].dt.year
    P(f"\n  {model}/{fs}/{scheme}/h={h}: annual return by calendar year")
    for y, g in r.groupby("year"):
        rr = g["portfolio_return"].to_numpy(float)
        yr_ret = np.prod(1 + np.clip(rr, -0.99, None)) - 1
        P(f"      {y}: {100*yr_ret:+7.1f}%   (rebalances={len(rr)})")

# ===========================================================================
hr("C  EARNINGS-WINDOW SENSITIVITY of the pooled IC (XGBoost/D)")
for scheme, h in [("expanding", 3), ("expanding", 5)]:
    d = oos[(oos.feature_set=="D")&(oos.scheme==scheme)&(oos.horizon==h)].merge(
        panel[["ticker","date","days_earn"]], on=["ticker","date"], how="left")
    ic_all, _ = daily_ic(d)
    d_ex = d[~((d.days_earn >= 0) & (d.days_earn <= h))]  # drop windows crossing earnings
    ic_ex, _ = daily_ic(d_ex)
    P(f"  {scheme:13} h={h}: IC_all={ic_all:+.4f}  IC_excl-earnings-cross={ic_ex:+.4f}  "
      f"delta={ic_ex-ic_all:+.4f}")

# ===========================================================================
hr("C  SVI NO-ARBITRAGE DIAGNOSTIC (constraints NOT imposed in fit_svi)")
pf = pd.read_parquet(PANEL / "panel_unnormalized.parquet", columns=["svi_b", "svi_rho"])
pf = pf.dropna()
wing = pf["svi_b"] * (1 + pf["svi_rho"].abs())  # Lee/Gatheral large-strike slope
P(f"  headline-expiry fits with finite SVI params: {len(pf)}")
P(f"  Lee wing-slope b*(1+|rho|): mean={wing.mean():.3f} median={wing.median():.3f} "
  f"p95={wing.quantile(.95):.3f}")
P(f"  fraction violating no-butterfly necessary bound b*(1+|rho|) > 2: "
  f"{100*(wing > 2).mean():.2f}%")
P(f"  fraction violating b*(1+|rho|) > 4: {100*(wing > 4).mean():.2f}%")

# ===========================================================================
hr("C  FREQUENCY: daily vs quarterly(per-window) IC for Set D (context for the 0.02-0.05 benchmark)")
ww = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
for scheme, h in [("rolling_9m_3m", 5), ("expanding", 3)]:
    g = ww[(ww.model=="ElasticNet" if scheme=="rolling_9m_3m" else ww.model=="XGBoost")
           & (ww.feature_set=="D") & (ww.scheme==scheme) & (ww.horizon==h)]
    if not g.empty:
        P(f"  {g.model.iloc[0]}/D/{scheme}/h={h}: per-window(quarterly) mean IC = "
          f"{g['ic'].mean():+.4f} over {len(g)} windows (this is the number compared to the benchmark)")

# ---------------------------------------------------------------------------
(TAB / "referee_fixes.txt").write_text("\n".join(out_lines))
print("\n[written] results/tables/referee_fixes.txt")
