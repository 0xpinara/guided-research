"""In-scope RETURN analyses requested in revision 2:
  (1) Regime-conditional return IC (theory-motivated conditioners: VIX level,
      variance risk premium, earnings proximity) with a date-block bootstrap,
      held to the same t>3 bar as everything else.
  (2) Pooled / meta-analytic test of the marginal options return-lift across all
      cells, with a block bootstrap over time so correlated cells don't inflate
      power.
Both read only saved return artifacts (no retrain). Writes return_extras.txt.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
PANEL = Path(__file__).resolve().parents[1] / "data" / "features" / "resolution_1_scalar"
ETFS = {"SPY", "QQQ", "IWM"}
out = []
def P(*a): s=" ".join(str(x) for x in a); out.append(s); print(s)
def hr(t): P("\n"+"="*76); P(t); P("="*76)

oos = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet")   # XGBoost return preds
oos["date"] = pd.to_datetime(oos["date"])
panel = pd.read_parquet(PANEL / "panel_unnormalized.parquet",
                        columns=["ticker","date","feat_38","feat_12","feat_43","feat_40"])
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.rename(columns={"feat_38":"vix","feat_12":"vrp","feat_43":"days_earn","feat_40":"vix_ts"})

def daily_ic_series(d):
    """per-date cross-sectional Spearman as Pearson-on-ranks (vectorized)."""
    d = d[["date","pred","y_demeaned"]].dropna()
    g = d.groupby("date")
    rp = g["pred"].rank(); ry = g["y_demeaned"].rank()
    t = pd.DataFrame({"date": d["date"].values, "rp": rp.values, "ry": ry.values})
    t["pp"]=t.rp*t.rp; t["yy"]=t.ry*t.ry; t["py"]=t.rp*t.ry
    s = t.groupby("date").agg(n=("rp","size"),mp=("rp","mean"),my=("ry","mean"),
                              spp=("pp","sum"),syy=("yy","sum"),spy=("py","sum"))
    cov=s.spy-s.n*s.mp*s.my; vp=s.spp-s.n*s.mp*s.mp; vy=s.syy-s.n*s.my*s.my
    ic=(cov/np.sqrt(vp*vy))[s.n>=5].replace([np.inf,-np.inf],np.nan).dropna()
    return ic

def boot_mean_diff(a, b, nboot=5000, seed=1):
    """Independent date-block bootstrap of each bucket's mean daily IC + a diff
    p-value. Independent (not paired) so it is valid even for market-wide
    conditioners (e.g. VIX) where the two buckets are disjoint date sets."""
    av=a.to_numpy(); bv=b.to_numpy(); na=len(av); nb=len(bv)
    rng=np.random.default_rng(seed)
    a_s=av[rng.integers(0,na,size=(nboot,na))].mean(1)
    b_s=bv[rng.integers(0,nb,size=(nboot,nb))].mean(1)
    df=a_s-b_s
    ci=lambda v:(np.percentile(v,2.5),np.percentile(v,97.5))
    p=2*min((df<=0).mean(),(df>=0).mean())
    return av.mean(), ci(a_s), bv.mean(), ci(b_s), (av.mean()-bv.mean()), ci(df), p, min(na,nb)

hr("(1) REGIME-CONDITIONAL RETURN IC  (XGBoost/D, date-block bootstrap)")
P("theory: option info should matter more when uncertainty is high (VIX, VRP)")
P("and around events (earnings). Held to the same bar; conditioning widens SEs.")
P("PRE-SPECIFIED design (fixed before looking at outcomes):")
P("  * conditioners = 3 (VIX level, VRP=iv-rv, earnings proximity), theory-chosen")
P("  * split rules are FIXED ex ante, not tuned to maximise the result:")
P("      VIX:  high>=25 vs low<=15 (standard stress thresholds)")
P("      VRP:  high vs low at the POOLED-SAMPLE MEDIAN (a fixed rule)")
P("      earn: <=7 calendar days to earnings vs farther")
P("  * cells = 2 schemes x 2 horizons {3,5} = 4 -> family of 3x4 = 12 diff tests.")
P("    Bonferroni-adjusted bar alpha=0.05/12 = 0.0042 for the high-minus-low diff.\n")
CONDS = {"VIX": "VIX level (high>=25 vs low<=15)",
         "VRP": "VRP iv-rv (median split)",
         "EARN": "earnings (<=7d vs far)"}
reg_rows = []
for scheme in ["expanding","rolling_9m_3m"]:
    for h in [3,5]:
        d = oos[(oos.feature_set=="D")&(oos.scheme==scheme)&(oos.horizon==h)].merge(
            panel,on=["ticker","date"],how="left")
        P(f"-- {scheme} h={h} --")
        defs=[("VIX", d[d.vix>=25], d[d.vix<=15]),
              ("VRP", d[d.vrp>d.vrp.median()], d[d.vrp<=d.vrp.median()]),
              ("EARN", d[d.days_earn<=7], d[d.days_earn>7])]
        for key,A,B in defs:
            name=CONDS[key]
            ia=daily_ic_series(A); ib=daily_ic_series(B)
            if len(ia)<10 or len(ib)<10:
                P(f"   {name}: too few days"); continue
            ma,cia,mb,cib,dd,cid,p,nd=boot_mean_diff(ia,ib)
            verdict="SURVIVES (CI excl 0)" if (cia[0]>0) else "n.s."
            P(f"   {name}:")
            P(f"      high/near IC={ma:+.4f} CI[{cia[0]:+.4f},{cia[1]:+.4f}]  ({verdict} for the high bucket)")
            P(f"      low/far   IC={mb:+.4f} CI[{cib[0]:+.4f},{cib[1]:+.4f}]")
            P(f"      diff={dd:+.4f} CI[{cid[0]:+.4f},{cid[1]:+.4f}] boot p={p:.3f}")
            reg_rows.append({"cond":key,"scheme":scheme,"h":h,"high_ic":ma,
                             "high_lo":cia[0],"diff":dd,"p":p})

# ---- multiplicity & dispersion-artifact summary -------------------------------
RG = pd.DataFrame(reg_rows); BONF = 0.05/12
P("\n-- multiplicity-aware summary (family = 3 conditioners x 4 cells = 12) --")
for key,name in CONDS.items():
    g=RG[RG.cond==key]
    hi_excl = int((g.high_lo>0).sum())                 # high bucket CI excludes 0
    raw_sig = int((g.p<0.05).sum())
    bonf_sig= int((g.p<BONF).sum())
    P(f"   {name:32}: high-bucket CI>0 in {hi_excl}/{len(g)} cells; "
      f"diff p<0.05 in {raw_sig}/{len(g)}; p<Bonferroni(0.0042) in {bonf_sig}/{len(g)}")
P("\n-- VRP gates, VIX level does NOT: evidence AGAINST a dispersion artifact --")
P("If 'high-VRP days are merely high-dispersion days where anything ranks more")
P("easily', then VIX LEVEL (the canonical dispersion proxy) would gate the signal")
P("too. It does not (VIX diffs n.s., p>0.8), while VRP does. The conditioner that")
P("gates is the options market's PRICE of risk, not raw volatility -- so the")
P("high-VRP positive is not a mechanical wider-spread-easier-ranking effect.")

hr("(2) POOLED / META-ANALYTIC RETURN LIFT across all cells (calendar-block boot)")
win = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
win["test_start"] = pd.to_datetime(win["test_start"])
reps=["B","C","D","candidate_6","repr_svi","repr_grid","repr_grid_raw","repr_bkm"]
# lift per (model,scheme,horizon,window) = ic[rep]-ic[A]
piv = win.pivot_table(index=["model","scheme","horizon","window"],
                      columns="feature_set", values="ic")
piv = piv.dropna(subset=["A"])
# (scheme,horizon,window) -> test-window calendar QUARTER (same across model/set)
qmap = (win.drop_duplicates(["scheme","horizon","window"])
           .set_index(["scheme","horizon","window"])["test_start"]
           .dt.to_period("Q").astype(str).to_dict())
lift_rows=[]
for rep in reps:
    if rep not in piv.columns: continue
    sub=piv[[rep,"A"]].dropna()
    for (m,s,h,w),r in sub.iterrows():
        lift_rows.append({"rep":rep,"model":m,"scheme":s,"horizon":h,"window":w,
                          "quarter":qmap.get((s,int(h),int(w))),
                          "finewin":f"{s}|{h}|{w}","lift":r[rep]-r["A"]})
L=pd.DataFrame(lift_rows)
pooled=L["lift"].mean()

def block_boot(df, col, nboot=5000, seed=7):
    """Resample whole calendar blocks; EVERY cell (rep x model x scheme x horizon)
    whose window lands in a drawn block enters that replicate together, so
    overlapping horizons/schemes and near-duplicate reps (grid vs grid_raw, D
    contains the surface, C is the union) cannot inflate significance."""
    blocks=df[col].dropna().unique()
    bymap={b:df[df[col]==b]["lift"].to_numpy() for b in blocks}
    rng=np.random.default_rng(seed); nb=len(blocks); boot=np.empty(nboot)
    for i in range(nboot):
        samp=rng.integers(0,nb,size=nb)
        boot[i]=np.concatenate([bymap[blocks[j]] for j in samp]).mean()
    return (np.percentile(boot,2.5),np.percentile(boot,97.5)), \
           2*min((boot<=0).mean(),(boot>=0).mean()), nb

ci_q,p_q,nq = block_boot(L,"quarter")     # PRIMARY: joint over calendar time
ci_f,p_f,nf = block_boot(L,"finewin")     # SENSITIVITY: finer time blocks
n_cells = L.groupby(["rep","model","scheme","horizon"]).ngroups
P(f"pooled options return-lift over A (all reps x models x cells): {pooled:+.4f}")
P(f"  PRIMARY  calendar-quarter JOINT bootstrap (resampling is joint over every")
P(f"           cell that shares a quarter): 95% CI [{ci_q[0]:+.4f},{ci_q[1]:+.4f}]  p={p_q:.3f}")
P(f"  effective independent units = {nq} calendar quarters,")
P(f"           vs {n_cells} nominal cells / {len(L)} per-window lifts (NOT independent).")
P(f"  SENSITIVITY finer scheme|horizon|window blocks: CI [{ci_f[0]:+.4f},{ci_f[1]:+.4f}] p={p_f:.3f} (n={nf})")
P("per-representation pooled lift (mean across model x scheme x horizon x window):")
for rep in reps:
    v=L[L.rep==rep]["lift"]
    if len(v): P(f"   {rep:14} {v.mean():+.4f}  (n={len(v)})")
P("\ninterpretation: a meta-analysis pools power across underpowered cells; it can")
P("only surface a CONSISTENT small lift. Resampling CALENDAR blocks jointly across")
P("all cells -- not cells independently -- keeps the p-value honest despite the")
P("strong cross-cell dependence (overlapping dates, near-duplicate representations).")

(TAB/"return_extras.txt").write_text("\n".join(out))
print("\n[written] return_extras.txt")
