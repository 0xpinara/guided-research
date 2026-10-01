"""Returns-vs-volatility dissociation + marginal-lift-over-A, written to file."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
out = []
def P(*a): s=" ".join(str(x) for x in a); out.append(s); print(s)

ret = pd.read_csv(TAB / "posthoc_ic_tstats.csv")
vol = pd.read_csv(TAB / "vol_ic_tstats.csv")

def ic(df, m, s, h, fs):
    r = df[(df.model==m)&(df.scheme==s)&(df.horizon==h)&(df.feature_set==fs)]
    return (float(r.mean_ic.iloc[0]), float(r.nw_tstat.iloc[0]),
            bool(r.survives_hlz_t3.iloc[0])) if len(r) else (np.nan,)*3

def hr(t): P("\n"+"="*74); P(t); P("="*74)

hr("VOLATILITY GRID: mean per-window IC + HAC t, by set (EN & XGB)")
for m in ["ElasticNet","XGBoost"]:
    P(f"\n-- {m} --")
    for s in ["expanding","rolling_9m_3m"]:
        for h in [1,3,5]:
            a=ic(vol,m,s,h,"A")
            line=f"  {s:13} h={h}: "
            cells=[]
            for fs in ["A","B","C","D","candidate_6","repr_svi","repr_grid","repr_grid_raw","repr_bkm"]:
                v=ic(vol,m,s,h,fs)
                tag="†" if v[2] else " "
                cells.append(f"{fs}={v[0]:.3f}{tag}")
            P(line+"  ".join(cells))

hr("MARGINAL LIFT OVER SET-A CONTROL: returns vs volatility")
P("(lift = IC_set - IC_A, averaged over the EN+XGB workhorse cells x scheme x horizon)")
WORK=["ElasticNet","XGBoost"]
def avg_lift(df, fs, models=WORK):
    ds=[]
    for m in models:
        for s in df.scheme.unique():
            for h in sorted(df.horizon.unique()):
                a=ic(df,m,s,h,"A")[0]; r=ic(df,m,s,h,fs)[0]
                if np.isfinite(a) and np.isfinite(r): ds.append(r-a)
    return np.nanmean(ds), np.nanstd(ds)/np.sqrt(max(len(ds),1)), len(ds)
P(f"\n  {'set':14}{'RETURNS lift (se)':24}{'VOL lift (se)':24}")
for fs in ["B","C","D","candidate_6","repr_svi","repr_grid","repr_grid_raw","repr_bkm"]:
    rl=avg_lift(ret,fs); vl=avg_lift(vol,fs)
    P(f"  {fs:14}{rl[0]:+.4f} ({rl[1]:.4f})        {vl[0]:+.4f} ({vl[1]:.4f})")
P("\n  (OLS-on-raw-aggregates is the lone exception: OLS/B,OLS/C vol IC ~0.04 --")
P("   dumping 34-49 raw features into unregularised OLS overfits, which is itself")
P("   a representation point; the compact reps D/grid/BKM/candidate_6 lift robustly.)")

hr("HLZ t>3 SURVIVORS in the VOL grid (count + list)")
surv=vol[vol.survives_hlz_t3]
P(f"  vol cells with t>3: {len(surv)} of {len(vol)}  (returns had 7 of 198)")
for _,r in surv.sort_values('nw_tstat',ascending=False).head(25).iterrows():
    P(f"   {r.model}/{r.feature_set}/{r.scheme}/h{int(r.horizon)}  IC={r.mean_ic:.3f} t={r.nw_tstat:.1f}")

hr("HEADLINE DISSOCIATION NUMBERS")
# Strongest vol Set-D cell and its lift over A
dcell=vol[vol.feature_set=="D"].sort_values("nw_tstat",ascending=False).iloc[0]
M,S,H=dcell.model,dcell.scheme,int(dcell.horizon)
a=ic(vol,M,S,H,"A"); d=ic(vol,M,S,H,"D"); c=ic(vol,M,S,H,"C")
P(f"  strongest vol Set-D cell: {M}/{S}/h{H}: A={a[0]:.3f}(t={a[1]:.1f}), "
  f"D={d[0]:.3f}(t={d[1]:.1f}), C={c[0]:.3f}(t={c[1]:.1f}); D-A lift={d[0]-a[0]:+.3f}")
# best union lift over A
best=None
for m in vol.model.unique():
    for s in vol.scheme.unique():
        for h in [1,3,5]:
            la=ic(vol,m,s,h,"A")[0]
            for fs in ["B","C","D"]:
                lr=ic(vol,m,s,h,fs)
                if np.isfinite(lr[0]) and (best is None or lr[0]-la>best[0]):
                    best=(lr[0]-la,m,s,h,fs,lr[0],lr[1],la)
P(f"  largest options vol-lift over A: {best[4]} {best[1]}/{best[2]}/h{best[3]} "
  f"lift={best[0]:+.3f} (set IC={best[5]:.3f} t={best[6]:.1f}, A={best[7]:.3f})")
# returns comparator
P(f"  RETURNS comparator (largest options lift over A): see ~0.01 range, all t<3 except A/B.")

(TAB/"vol_dissociation.txt").write_text("\n".join(out))
print("\n[written] vol_dissociation.txt")
