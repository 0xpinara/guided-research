"""Figures for the rewritten (referee-responsive) options paper.

Outputs PDFs into ../EasyChair3.5/figures/. All inputs are the saved result
tables; nothing is retrained. Numbers match results/tables/referee_fixes.txt.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT.parent / "EasyChair3.5" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

NAVY, BRICK, GOLD, GREY = "#1f3b57", "#9b2226", "#bb9457", "#888888"

icts = pd.read_csv(TAB / "posthoc_ic_tstats.csv")
ls = pd.read_parquet(TAB / "full_matrix_ls_daily.parquet")
ls["date"] = pd.to_datetime(ls["date"])

# ---------------------------------------------------------------------------
# FIG 1: representation IC + marginal lift over the identical Set-A control.
# Strongest cell where anything survives: ElasticNet / rolling / h=5.
# ---------------------------------------------------------------------------
order = ["A", "repr_svi", "repr_bkm", "repr_grid_raw", "repr_grid", "D"]
labels = ["A\n(stock only,\ncontrol)", "repr_svi\n(SVI params)", "repr_bkm\n(BKM 2 mom.)",
          "repr_grid_raw\n(model-free grid)", "repr_grid\n(SVI grid)", "D\n(full surface)"]
cell = icts[(icts.model == "ElasticNet") & (icts.scheme == "rolling_9m_3m") & (icts.horizon == 5)]
ic = [float(cell[cell.feature_set == s]["mean_ic"].iloc[0]) for s in order]
t  = [float(cell[cell.feature_set == s]["nw_tstat"].iloc[0]) for s in order]
ic_a = ic[0]
fig, ax = plt.subplots(figsize=(7.4, 3.5))
cols = [GREY if s == "A" else (BRICK if "svi" in s and "grid" not in s else NAVY) for s in order]
bars = ax.bar(range(len(order)), ic, color=cols, edgecolor="black", linewidth=0.5)
ax.axhline(ic_a, color=GREY, ls="--", lw=1.2, label=f"Set-A control IC = {ic_a:.4f}")
for i, (b, tv) in enumerate(zip(bars, t)):
    star = r"$\dagger$" if abs(tv) > 3 else ""
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0008,
            f"{ic[i]:.3f}\nt={tv:.1f}{star}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Pooled walk-forward IC")
ax.set_title("Representation vs. identical stock-only control (ElasticNet, rolling, 5-day)\n"
             r"$\dagger$ = survives Harvey--Liu--Zhu $t>3$.  SVI parameters add nothing; "
             "surface lift over control is tiny.", fontsize=8.5)
ax.set_ylim(0, max(ic) * 1.25)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_repr_lift.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# FIG 2: does the SVI fit earn its keep? repr_grid (SVI) vs repr_grid_raw (model-free)
# across all scheme x horizon cells, both linear and XGBoost.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.3))
cells = [("ElasticNet","expanding",1),("ElasticNet","expanding",3),("ElasticNet","expanding",5),
         ("ElasticNet","rolling_9m_3m",1),("ElasticNet","rolling_9m_3m",3),("ElasticNet","rolling_9m_3m",5),
         ("XGBoost","expanding",1),("XGBoost","expanding",3),("XGBoost","expanding",5),
         ("XGBoost","rolling_9m_3m",1),("XGBoost","rolling_9m_3m",3),("XGBoost","rolling_9m_3m",5)]
def getic(m, s, h, fs):
    r = icts[(icts.model==m)&(icts.scheme==s)&(icts.horizon==h)&(icts.feature_set==fs)]
    return float(r["mean_ic"].iloc[0]) if len(r) else np.nan
svi = [getic(m,s,h,"repr_grid") for m,s,h in cells]
raw = [getic(m,s,h,"repr_grid_raw") for m,s,h in cells]
x = np.arange(len(cells)); w = 0.4
ax.bar(x - w/2, svi, w, label="repr_grid (SVI-fit grid)", color=BRICK, edgecolor="black", lw=0.4)
ax.bar(x + w/2, raw, w, label="repr_grid_raw (model-free grid)", color=NAVY, edgecolor="black", lw=0.4)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"{m[:2]}/{'exp' if s=='expanding' else 'rol'}/{h}" for m,s,h in cells],
                   rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Pooled walk-forward IC")
ax.set_title("SVI fit does not earn its keep: SVI-grid vs. model-free grid trade wins across cells",
             fontsize=8.5)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_svi_vs_raw.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# FIG 3: GEX regime IC with block-bootstrap 95% CI (from referee_fixes.txt).
# Shows the short-/long-gamma difference is NOT significant.
# ---------------------------------------------------------------------------
# values transcribed from results/tables/referee_fixes.txt (XGBoost/D/expanding/h=3)
reg = [("short-gamma\n(GEX<0)", 0.0153, -0.0059, 0.0356),
       ("long-gamma\n(GEX>=0)", 0.0131, -0.0010, 0.0278)]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
xs = np.arange(len(reg))
means = [r[1] for r in reg]
lo = [r[1]-r[2] for r in reg]; hi = [r[3]-r[1] for r in reg]
ax.bar(xs, means, 0.5, color=[BRICK, NAVY], edgecolor="black", lw=0.5)
ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=6, lw=1.3)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in reg], fontsize=8)
ax.set_ylabel("Mean daily rank IC")
ax.set_title("Dealer-gamma regime: difference\n+0.0022, 95% CI [-0.021,+0.024], boot p=0.84 (n.s.)",
             fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig_regime_ci.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# FIG 4: Set-A (stock-only) vs Set-D economic control, ElasticNet and XGBoost,
# rolling 5-day. Shows the surface's $ edge is model-dependent (EN: D>A; XGB: A>D).
# ---------------------------------------------------------------------------
def equity(model, fs, scheme="rolling_9m_3m", h=5, stake=1000.0):
    r = ls[(ls.model==model)&(ls.feature_set==fs)&(ls.scheme==scheme)&(ls.horizon==h)].sort_values("date")
    rr = r["portfolio_return"].astype(float).to_numpy()
    return r["date"].to_numpy(), stake*np.cumprod(1+np.clip(rr,-0.99,None))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
for ax, model in [(a1,"ElasticNet"), (a2,"XGBoost")]:
    dA, eA = equity(model, "A"); dD, eD = equity(model, "D")
    ax.plot(dD, eD, color=BRICK, lw=1.6, label="Set D (full surface)")
    ax.plot(dA, eA, color=NAVY, lw=1.6, label="Set A (stock only)")
    ax.axhline(1000, color=GREY, ls="--", lw=1)
    ax.set_title(f"{model}  (rolling, 5-day)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left"); ax.set_xlabel("")
a1.set_ylabel("Value of \\$1{,}000 (after costs)")
fig.suptitle("Economic control: the surface's edge over stock-only is model-dependent "
             "(adds value for ElasticNet, reverses for XGBoost)", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig_control_equity.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# FIG 5: walk-forward mean IC by feature set (XGBoost, h=3), both schemes,
# regenerated to match Table 4 exactly (mean per-window IC).
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.2))
wsets = ["A", "B", "candidate_6", "D"]
wlab = ["A (control)", "B (options)", "candidate_6", "D (surface)"]
def micc(s, sch):
    r = icts[(icts.model=="XGBoost")&(icts.feature_set==s)&(icts.scheme==sch)&(icts.horizon==3)]
    return float(r["mean_ic"].iloc[0]) if len(r) else np.nan
exp = [micc(s, "expanding") for s in wsets]
rol = [micc(s, "rolling_9m_3m") for s in wsets]
x = np.arange(len(wsets)); w = 0.38
ax.bar(x - w/2, exp, w, label="expanding", color=NAVY, edgecolor="black", lw=0.4)
ax.bar(x + w/2, rol, w, label="rolling 9m/3m", color=GOLD, edgecolor="black", lw=0.4)
ax.axhline(0, color="black", lw=0.6)
ax.axhline(micc("A","expanding"), color=GREY, ls="--", lw=1, label="Set-A expanding")
ax.set_xticks(x); ax.set_xticklabels(wlab, fontsize=8)
ax.set_ylabel("Mean per-window IC")
ax.set_title("Walk-forward IC by feature set (XGBoost, 3-day): options blocks "
             "positive but not separated from the control", fontsize=8.3)
ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig_walkforward_ic.pdf"); plt.close(fig)

print("wrote: fig_repr_lift.pdf fig_svi_vs_raw.pdf fig_regime_ci.pdf "
      "fig_control_equity.pdf fig_walkforward_ic.pdf")
