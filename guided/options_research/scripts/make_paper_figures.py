"""All figures for the paper, in one consistent house style.

Reads only saved result tables; nothing is retrained. Volatility figures are
produced only if vol_ic_tstats.csv exists, so this can be run before and after
the vol grid finishes.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
# Write next to the paper if that folder exists (the author's setup); otherwise
# fall back to results/figures/ so the script is self-contained after a fresh clone.
_paper_fig = ROOT.parent / "EasyChair3.5" / "figures"
FIG = _paper_fig if _paper_fig.parent.exists() else ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- house style -----------------------------------------------------------
INK   = "#1d2333"   # near-black ink
NAVY  = "#264653"
TEAL  = "#2a9d8f"
GOLD  = "#e9a23b"
BRICK = "#a4243b"
GREY  = "#9aa0a6"
MUTED = "#6b7280"
PALETTE = [NAVY, BRICK, TEAL, GOLD, "#7b6d8d", GREY]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10.5, "font.family": "sans-serif",
    "axes.edgecolor": "#cfd3da", "axes.linewidth": 0.9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eceef1", "grid.linewidth": 0.9,
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
    "legend.frameon": False, "legend.fontsize": 8.6, "text.color": INK,
})

def finish(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)

icts = pd.read_csv(TAB / "posthoc_ic_tstats.csv")     # RETURN cells
ls = pd.read_parquet(TAB / "full_matrix_ls_daily.parquet")
ls["date"] = pd.to_datetime(ls["date"])
VOL = (TAB / "vol_ic_tstats.csv")
vol = pd.read_csv(VOL) if VOL.exists() else None

def cell_ic(df, model, scheme, h, fs):
    r = df[(df.model==model)&(df.scheme==scheme)&(df.horizon==h)&(df.feature_set==fs)]
    return (float(r.mean_ic.iloc[0]), float(r.nw_tstat.iloc[0]),
            bool(r.survives_hlz_t3.iloc[0])) if len(r) else (np.nan, np.nan, False)

print("Figures ->", FIG)

# ===========================================================================
# FIG: representation vs control (RETURNS) -- ElasticNet rolling 5d
# ===========================================================================
order = ["A", "repr_svi", "repr_bkm", "repr_grid_raw", "repr_grid", "D"]
lab = ["A\ncontrol", "SVI\nparams", "BKM\nmoments", "grid\n(model-free)",
       "grid\n(SVI)", "D\nfull surface"]
ic = [cell_ic(icts,"ElasticNet","rolling_9m_3m",5,s) for s in order]
vals=[v[0] for v in ic]; ts=[v[1] for v in ic]; hlz=[v[2] for v in ic]
fig, ax = plt.subplots(figsize=(7.2, 3.5))
cols=[GREY if s=="A" else (BRICK if s=="repr_svi" else NAVY) for s in order]
b=ax.bar(range(len(order)), vals, color=cols, width=0.66, zorder=3)
ax.axhline(vals[0], color=GREY, ls=(0,(4,3)), lw=1.3, zorder=2,
           label=f"Set-A control = {vals[0]:.4f}")
DAG = "$^\\dagger$"; NL = "\n"
for i,bar in enumerate(b):
    dg = DAG if hlz[i] else ""
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0009,
            f"{vals[i]:.3f}{NL}t={ts[i]:.1f}{dg}",
            ha="center", va="bottom", fontsize=7.6, color=INK)
ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("Mean walk-forward rank IC")
ax.set_title("Returns: surface barely lifts over a stock-only control "
             "(ElasticNet, rolling, 5-day)")
ax.set_ylim(0, max(vals)*1.30); ax.legend(loc="upper left")
ax.margins(x=0.02)
finish(fig, "fig_repr_lift.pdf")

# ===========================================================================
# FIG: SVI grid vs model-free grid (RETURNS), all linear/XGB cells
# ===========================================================================
cells=[("ElasticNet","expanding",1),("ElasticNet","expanding",3),("ElasticNet","expanding",5),
       ("ElasticNet","rolling_9m_3m",1),("ElasticNet","rolling_9m_3m",3),("ElasticNet","rolling_9m_3m",5),
       ("XGBoost","expanding",1),("XGBoost","expanding",3),("XGBoost","expanding",5),
       ("XGBoost","rolling_9m_3m",1),("XGBoost","rolling_9m_3m",3),("XGBoost","rolling_9m_3m",5)]
svi=[cell_ic(icts,m,s,h,"repr_grid")[0] for m,s,h in cells]
raw=[cell_ic(icts,m,s,h,"repr_grid_raw")[0] for m,s,h in cells]
x=np.arange(len(cells)); w=0.4
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.bar(x-w/2, svi, w, label="SVI-fit grid", color=BRICK, zorder=3)
ax.bar(x+w/2, raw, w, label="model-free grid", color=NAVY, zorder=3)
ax.axhline(0, color="#cfd3da", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(
    [f"{m[:2]}/{'ex' if s=='expanding' else 'ro'}/{h}" for m,s,h in cells],
    rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Mean walk-forward rank IC")
ax.set_title("Fitting SVI buys no edge: SVI grid vs. model-free grid trade wins")
ax.legend(ncol=2, loc="upper right")
finish(fig, "fig_svi_vs_raw.pdf")

# ===========================================================================
# FIG: regime CI (from referee_fixes.txt verified values)
# ===========================================================================
reg=[("short-gamma\n(GEX<0)",0.0153,-0.0059,0.0356),("long-gamma\n(GEX>=0)",0.0131,-0.0010,0.0278)]
fig, ax = plt.subplots(figsize=(4.4, 3.3))
xs=np.arange(2); means=[r[1] for r in reg]
lo=[r[1]-r[2] for r in reg]; hi=[r[3]-r[1] for r in reg]
ax.bar(xs, means, 0.52, color=[BRICK, NAVY], zorder=3)
ax.errorbar(xs, means, yerr=[lo,hi], fmt="none", ecolor=INK, capsize=7, lw=1.4, zorder=4)
ax.axhline(0, color="#cfd3da", lw=0.9)
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in reg], fontsize=9)
ax.set_ylabel("Mean daily rank IC")
ax.set_title("Dealer-gamma regime: n.s.\n$\\Delta=+0.0022$, 95% CI [-0.021,+0.024], $p=0.84$",
             fontsize=9.5)
finish(fig, "fig_regime_ci.pdf")

# ===========================================================================
# FIG: Set-A vs Set-D equity, EN & XGB (model-dependent economics)
# ===========================================================================
def equity(model, fs, scheme="rolling_9m_3m", h=5, stake=1000.):
    r=ls[(ls.model==model)&(ls.feature_set==fs)&(ls.scheme==scheme)&(ls.horizon==h)].sort_values("date")
    rr=r["portfolio_return"].astype(float).to_numpy()
    return r["date"].to_numpy(), stake*np.cumprod(1+np.clip(rr,-0.99,None))
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.6,3.1),sharey=True)
for ax,model in [(a1,"ElasticNet"),(a2,"XGBoost")]:
    dD,eD=equity(model,"D"); dA,eA=equity(model,"A")
    ax.plot(dD,eD,color=BRICK,lw=1.9,label="Set D (surface)")
    ax.plot(dA,eA,color=NAVY,lw=1.9,label="Set A (stock only)")
    ax.axhline(1000,color=GREY,ls=(0,(4,3)),lw=1)
    ax.set_title(f"{model}  (rolling, 5-day)")
    ax.legend(loc="upper left")
a1.set_ylabel("Value of \\$1{,}000 (after costs)")
fig.suptitle("Economic edge of the surface is model-dependent (helps EN, reverses for XGB)",
             fontsize=9.5, y=1.02, color=INK, weight="bold")
finish(fig,"fig_control_equity.pdf")

# ===========================================================================
# FIG: walk-forward IC by set (XGB h3) -- matches Table 4
# ===========================================================================
wsets=["A","B","candidate_6","D"]; wlab=["A (control)","B (options)","cand_6","D (surface)"]
exp=[cell_ic(icts,"XGBoost","expanding",3,s)[0] for s in wsets]
rol=[cell_ic(icts,"XGBoost","rolling_9m_3m",3,s)[0] for s in wsets]
x=np.arange(len(wsets)); w=0.38
fig,ax=plt.subplots(figsize=(6.4,3.1))
ax.bar(x-w/2,exp,w,label="expanding",color=NAVY,zorder=3)
ax.bar(x+w/2,rol,w,label="rolling 9m/3m",color=GOLD,zorder=3)
ax.axhline(0,color="#cfd3da",lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(wlab,fontsize=8.5)
ax.set_ylabel("Mean per-window IC")
ax.set_title("Returns walk-forward: options positive but not separated from control")
ax.legend(ncol=2)
finish(fig,"fig_walkforward_ic.pdf")

# ===========================================================================
# FIG: statistical power -- MDE vs universe size
# ===========================================================================
pc=pd.read_csv(TAB/"power_curve.csv")
fig,ax=plt.subplots(figsize=(6.6,3.6))
ax.plot(pc.N, pc.mde_t3, color=NAVY, lw=2.2, zorder=3, label="MDE at $t>3$  (= $3\\times$HAC SE)")
ax.axhspan(0.002,0.004, color=GOLD, alpha=0.25, zorder=1,
           label="observed options lift (0.002--0.004)")
ax.axvline(60, color=BRICK, ls=(0,(4,3)), lw=1.3, zorder=2)
ax.text(66, ax.get_ylim()[1]*0.78, "this study\n(60 names)", color=BRICK, fontsize=8)
# crossing at lift 0.004
ncross=60*(pc.mde_t3.iloc[0]/0.004)**2
ax.axvline(ncross, color=TEAL, ls=":", lw=1.6, zorder=2)
ax.text(ncross+40, ax.get_ylim()[1]*0.55, f"~{ncross:.0f} names\nto detect 0.004",
        color=TEAL, fontsize=8)
ax.set_xlabel("Cross-section size $N$ (tradeable names)")
ax.set_ylabel("Minimum detectable IC")
ax.set_title("Why a 60-name study cannot resolve a 0.004 lift")
ax.set_xlim(0, pc.N.max()); ax.set_ylim(0, pc.mde_t3.iloc[0]*1.05)
ax.legend(loc="upper right")
finish(fig,"fig_power.pdf")

# ===========================================================================
# VOL FIGURES (only if vol grid finished) ------------------------------------
# ===========================================================================
if vol is not None and len(vol):
    # representative cell: XGBoost expanding 5d (also used for returns lift)
    def vlift(df, model, scheme, h):
        a=cell_ic(df,model,scheme,h,"A")[0]
        return {fs: cell_ic(df,model,scheme,h,fs)[0]-a for fs in ["B","D","repr_grid","repr_bkm","candidate_6"]}, a

    # ---- FIG dissociation: options lift over control, RETURNS vs VOL ----
    reps=["B","D","repr_grid","repr_bkm"]; rlab=["B\noptions","D\nsurface","grid\n(SVI)","BKM"]
    def avg_lift(df):
        # average lift over A across the EN+XGB workhorse cells (OLS-on-raw
        # overfits vol and is reported separately, not in the headline figure)
        out={}
        for fs in reps:
            ds=[]
            for m in ["ElasticNet","XGBoost"]:
                for s in df.scheme.unique():
                    for h in sorted(df.horizon.unique()):
                        a=cell_ic(df,m,s,h,"A")[0]; r=cell_ic(df,m,s,h,fs)[0]
                        if np.isfinite(a) and np.isfinite(r): ds.append(r-a)
            out[fs]=(np.nanmean(ds), np.nanstd(ds)/np.sqrt(max(len(ds),1)))
        return out
    rl=avg_lift(icts); vl=avg_lift(vol)
    x=np.arange(len(reps)); w=0.38
    fig,ax=plt.subplots(figsize=(7.0,3.7))
    ax.bar(x-w/2,[rl[f][0] for f in reps],w,yerr=[rl[f][1] for f in reps],
           capsize=4, color=GREY, label="Returns target", zorder=3,
           error_kw=dict(ecolor=INK,lw=1))
    ax.bar(x+w/2,[vl[f][0] for f in reps],w,yerr=[vl[f][1] for f in reps],
           capsize=4, color=TEAL, label="Volatility target", zorder=3,
           error_kw=dict(ecolor=INK,lw=1))
    ax.axhline(0,color="#cfd3da",lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(rlab,fontsize=8.5)
    ax.set_ylabel("Marginal rank-IC lift over stock-only control")
    ax.set_title("The dissociation: option information lifts VOLATILITY prediction,\n"
                 "not RETURN prediction (lift over identical Set-A control, avg across cells)")
    ax.legend(loc="upper left")
    finish(fig,"fig_dissociation.pdf")

    # ---- FIG vol representation vs control (level), strong cell ----
    order=["A","repr_svi","repr_bkm","repr_grid_raw","repr_grid","D"]
    lab=["A\ncontrol","SVI\nparams","BKM","grid\n(model-free)","grid\n(SVI)","D\nfull surface"]
    # pick the model/scheme/horizon with the largest Set-D vol t-stat
    cand=vol[(vol.feature_set=="D")].sort_values("nw_tstat",ascending=False).iloc[0]
    M,S,H=cand.model,cand.scheme,int(cand.horizon)
    ic=[cell_ic(vol,M,S,H,s) for s in order]
    vals=[v[0] for v in ic]; ts=[v[1] for v in ic]; hlz=[v[2] for v in ic]
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    cols=[GREY if s=="A" else (BRICK if s=="repr_svi" else TEAL) for s in order]
    b=ax.bar(range(len(order)),vals,color=cols,width=0.66,zorder=3)
    ax.axhline(vals[0],color=GREY,ls=(0,(4,3)),lw=1.3,label=f"Set-A control = {vals[0]:.3f}")
    for i,bar in enumerate(b):
        dg = DAG if hlz[i] else ""
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.004,
                f"{vals[i]:.3f}{NL}t={ts[i]:.0f}{dg}",
                ha="center",va="bottom",fontsize=7.6,color=INK)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab,fontsize=8)
    ax.set_ylabel("Mean walk-forward rank IC (volatility)")
    sl={"expanding":"expanding","rolling_9m_3m":"rolling"}[S]
    ax.set_title(f"Volatility: the surface adds real forward-looking information "
                 f"({M}, {sl}, {H}-day)")
    ax.set_ylim(0,max(vals)*1.22); ax.legend(loc="upper left")
    finish(fig,"fig_vol_lift.pdf")
    print("VOL FIGURES: done")
else:
    print("VOL FIGURES: skipped (vol_ic_tstats.csv not present yet)")

# ===========================================================================
# FIG: VRP-conditional RETURN IC -- the in-scope positive (values from
# results/tables/return_extras.txt; date-block bootstrap 95% CIs)
# ===========================================================================
# (cell, high mean, high lo, high hi, low mean, low lo, low hi)
vrp=[("exp 3d", 0.0308,0.0161,0.0458, -0.0034,-0.0179,0.0111),
     ("exp 5d", 0.0294,0.0150,0.0435,  0.0136,-0.0008,0.0279),
     ("roll 3d",0.0241,0.0118,0.0365, -0.0093,-0.0221,0.0034),
     ("roll 5d",0.0207,0.0078,0.0335, -0.0172,-0.0303,-0.0038)]
fig,ax=plt.subplots(figsize=(6.8,3.5)); x=np.arange(len(vrp)); w=0.38
hm=[v[1] for v in vrp]; hlo=[v[1]-v[2] for v in vrp]; hhi=[v[3]-v[1] for v in vrp]
lm=[v[4] for v in vrp]; llo=[v[4]-v[5] for v in vrp]; lhi=[v[6]-v[4] for v in vrp]
ax.bar(x-w/2,hm,w,yerr=[hlo,hhi],capsize=4,color=TEAL,label="high VRP (rich options)",
       zorder=3,error_kw=dict(ecolor=INK,lw=1))
ax.bar(x+w/2,lm,w,yerr=[llo,lhi],capsize=4,color=GREY,label="low VRP",
       zorder=3,error_kw=dict(ecolor=INK,lw=1))
ax.axhline(0,color="#cfd3da",lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([v[0] for v in vrp])
ax.set_ylabel("Return rank IC (XGBoost / Set D)")
ax.set_title("In-scope positive: option features predict returns when the\n"
             "variance risk premium is high (CI excludes zero in every cell)")
ax.legend(loc="upper right")
finish(fig,"fig_regime_return.pdf")

# ===========================================================================
# FIG: TreeSHAP attribution -- returns vs volatility
# ===========================================================================
ts=pd.read_csv(TAB/"treeshap_importance.csv") if (TAB/"treeshap_importance.csv").exists() else None
if ts is not None:
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.8))
    for ax,tgt,col in [(a1,"return",GREY),(a2,"volatility",TEAL)]:
        d=ts[ts.target==tgt].head(10).iloc[::-1]
        ax.barh(range(len(d)), d["share"].to_numpy()*100, color=col, zorder=3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["label"], fontsize=7.4)
        ax.set_title(("Returns" if tgt=="return" else "Volatility"))
        ax.set_xlabel("share of |TreeSHAP| (%)")
    fig.suptitle("What the model uses: diffuse macro/technical for returns vs.\n"
                 "IV-level surface + VIX + earnings for volatility",
                 fontsize=9.5, y=1.04, weight="bold", color=INK)
    finish(fig,"fig_treeshap.pdf")

# ===========================================================================
# FIG: meta-analysis forest -- per-representation pooled return lift + pooled CI
# ===========================================================================
if (TAB/"return_extras.txt").exists():
    # per-rep pooled lift (from return_extras.txt), pooled est + 95% CI
    rep_lift={"candidate_6":0.0069,"B":0.0040,"repr_grid":0.0038,"C":0.0026,
              "repr_grid_raw":0.0026,"D":0.0021,"repr_svi":0.0000,"repr_bkm":-0.0006}
    # PRIMARY: calendar-quarter JOINT bootstrap (honest cross-cell + cross-time
    # dependence); the finer per-window blocking (p=0.002) is only a sensitivity.
    pooled, plo, phi = 0.0029, -0.0000, 0.0059
    fig,ax=plt.subplots(figsize=(6.4,3.4))
    ys=list(range(len(rep_lift)))
    items=sorted(rep_lift.items(), key=lambda kv: kv[1])
    ax.scatter([v for _,v in items], ys, color=NAVY, zorder=4, s=34)
    ax.set_yticks(ys); ax.set_yticklabels([k for k,_ in items], fontsize=8)
    ax.axvspan(plo,phi,color=TEAL,alpha=0.22,zorder=1,
               label=f"pooled lift {pooled:+.4f}\ncalendar-block 95% CI [{plo:+.4f},{phi:+.4f}], p=0.051")
    ax.axvline(pooled,color=TEAL,lw=1.6,zorder=2)
    ax.axvline(0,color=BRICK,ls=(0,(4,3)),lw=1.2,zorder=2)
    ax.set_xlabel("pooled marginal return-IC lift over control")
    ax.set_title("Meta-analysis: a tiny, borderline-consistent return lift\n"
                 "(pooled across cells; far below the 0.019 detection bar)")
    ax.legend(loc="lower right", fontsize=7.8)
    finish(fig,"fig_meta.pdf")

# ===========================================================================
# FIG: signal-injection recovery -- the power curve as an experiment
# ===========================================================================
if (TAB/"signal_injection.csv").exists():
    si=pd.read_csv(TAB/"signal_injection.csv")
    MDE=0.019
    # XGBoost is the headline (the workhorse used for the return table); OLS, lacking
    # regularisation against the 59 ticker dummies, dilutes a thin factor in its OOS
    # variance and is reported in text, not plotted.
    agg=(si[si.model=="XGBoost"].groupby("delta")
           .agg(oracle=("oracle_ic","mean"), rec=("recovered_ic","mean"),
                t=("nw_t","mean"), surv=("survives_t3","sum"),
                nseed=("survives_t3","size")).reset_index().sort_values("oracle"))
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.5))
    xmax=agg.oracle.max()*1.05
    # left: recovered vs injected IC, with detection floor + diagonal
    a1.axhspan(0,MDE,color=GREY,alpha=0.16,zorder=0)
    a1.plot([0,xmax],[0,xmax],color="#cfd3da",lw=1,ls=(0,(4,3)),zorder=1,label="perfect recovery")
    a1.axhline(MDE,color=BRICK,lw=1.3,ls=(0,(3,2)),zorder=2,label=f"MDE ($t{{>}}3$) = {MDE}")
    a1.plot(agg.oracle,agg.rec,color=NAVY,lw=1.9,marker="o",ms=5,zorder=3,label="XGBoost recovered")
    a1.set_xlabel("Injected (oracle) cross-sectional IC")
    a1.set_ylabel("Recovered walk-forward IC")
    a1.set_title("Recovery tracks the injected effect")
    a1.legend(loc="upper left",fontsize=7.4); a1.set_xlim(0,xmax)
    # right: NW t vs injected IC, with t>3 hurdle -- the money plot
    a2.axhspan(0,3,color=GREY,alpha=0.16,zorder=0)
    a2.axhline(3,color=BRICK,lw=1.3,ls=(0,(3,2)),zorder=2,label="$t>3$ hurdle")
    # colour points by survival
    a2.plot(agg.oracle,agg.t,color=NAVY,lw=1.9,zorder=3)
    a2.scatter(agg.oracle,agg.t,s=42,zorder=4,
               c=[NAVY if s==3 else (GOLD if s>0 else GREY) for s in agg.surv],
               edgecolor=INK,linewidth=0.5)
    for xv,txt in [(0.0037,"observed\nlift 0.003"),(0.0611,"vol-scale\n0.05")]:
        a2.axvline(xv,color=TEAL,lw=1,ls=":",zorder=1)
        a2.text(xv,a2.get_ylim()[1]*0.50,txt,fontsize=7,color=TEAL,rotation=90,va="center",ha="right")
    a2.set_xlabel("Injected (oracle) cross-sectional IC")
    a2.set_ylabel("Recovered Newey--West $t$")
    a2.set_title("Detected only above the power floor")
    a2.legend(loc="upper left",fontsize=7.4); a2.set_xlim(0,xmax)
    fig.suptitle("Signal-injection recovery (XGBoost workhorse): a 0.05 signal is caught at $t\\gg3$, "
                 "a 0.003 one is not,\nand the detection threshold coincides with the analytic MDE",
                 fontsize=9.0,y=1.05,weight="bold",color=INK)
    finish(fig,"fig_injection.pdf")

print("EXTRA FIGURES done")
