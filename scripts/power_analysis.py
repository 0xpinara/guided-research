"""Statistical power analysis for the return-prediction task.

Pure calculation from the existing return walk-forward outputs (no compute):
how large a cross-section would be needed to resolve the observed options
marginal-lift at the Harvey-Liu-Zhu t>3 discovery hurdle.

Writes results/tables/power_analysis.txt and ...power_curve.csv (for the figure).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
N0 = 60                      # current tradeable cross-section
out = []
def P(*a):
    s = " ".join(str(x) for x in a); out.append(s); print(s)

icts = pd.read_csv(TAB / "posthoc_ic_tstats.csv")            # return cells
win = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")

# 1) representative HAC SE of the per-window mean IC across the return grid
se_med = float(icts["nw_se"].median())
se_surf = float(icts[icts.feature_set.isin(["B", "D", "repr_grid"])]["nw_se"].median())
mde_t3 = 3.0 * se_med
P("== Statistical power of the 60-name return study ==")
P(f"median HAC SE of per-window mean IC (198 cells): {se_med:.4f}")
P(f"  ... over options/surface cells (B,D,repr_grid): {se_surf:.4f}")
P(f"minimum detectable effect at t>3 (MDE = 3*SE): {mde_t3:.4f}")

# 2) per-window IC dispersion vs the cross-sectional sampling floor 1/sqrt(N-1)
icsd = win.groupby(["model", "feature_set", "scheme", "horizon"])["ic"].std()
P(f"median per-window IC std (within cell): {float(icsd.median()):.4f}")
P(f"single-cross-section IC sampling sd 1/sqrt(N-1), N={N0}: {1/np.sqrt(N0-1):.4f}")

# 3) observed marginal options lift over the Set-A control (typical magnitude)
#    use the significant rolling-5d cell + the broad average of |lift| for B/D/grid
def lift(model, scheme, h, fs):
    a = icts[(icts.model==model)&(icts.scheme==scheme)&(icts.horizon==h)&(icts.feature_set=="A")]
    r = icts[(icts.model==model)&(icts.scheme==scheme)&(icts.horizon==h)&(icts.feature_set==fs)]
    if len(a) and len(r):
        return float(r.mean_ic.iloc[0]) - float(a.mean_ic.iloc[0])
    return np.nan
lifts = [lift("ElasticNet","rolling_9m_3m",5,"D"),
         lift("ElasticNet","rolling_9m_3m",5,"repr_grid"),
         lift("XGBoost","expanding",3,"D"),
         lift("XGBoost","expanding",5,"D")]
lift_typ = float(np.nanmean([abs(x) for x in lifts]))
P(f"observed options marginal lift over A: rolling-5d D=+{lifts[0]:.4f}, "
  f"grid=+{lifts[1]:.4f}; broad |lift| avg={lift_typ:.4f}")

# 4) required universe: SE scales as 1/sqrt(N) (cross-sectional sampling), so to
#    push the MDE down to a target lift, N must grow by (MDE/lift)^2.
P("required cross-section N to resolve a given marginal lift at t>3:")
HEAD = 0.004
for lf in (0.002, 0.003, 0.004, 0.006):
    Nr = N0 * (mde_t3 / lf) ** 2
    tag = "   <-- headline (paper states lift 0.002-0.004)" if abs(lf-HEAD) < 1e-9 else ""
    P(f"   lift={lf:.3f}:  MDE/lift={mde_t3/lf:4.1f}x   N_req ~= {Nr:5.0f} names{tag}")
N_req = N0 * (mde_t3 / HEAD) ** 2
P(f"  => at the literature-relevant lift of {HEAD:.3f}, need ~{N_req:.0f} names;")
P(f"     large-universe sort papers use thousands -> their positive and this")
P(f"     60-name null are reconciled by POWER, not contradiction.")

# 5) MDE as a function of universe size N, for the figure
Ns = np.unique(np.r_[np.arange(60, 600, 20), np.arange(600, 3200, 100)])
mde = mde_t3 * np.sqrt(N0 / Ns)
pd.DataFrame({"N": Ns, "mde_t3": mde}).to_csv(TAB / "power_curve.csv", index=False)
P(f"[written] power_curve.csv ({len(Ns)} points), lift={lift_typ:.4f}, "
  f"MDE0={mde_t3:.4f}, N_req={N_req:.0f}")

(TAB / "power_analysis.txt").write_text("\n".join(out) +
    f"\n\n# machine-readable\nMDE_t3={mde_t3:.6f}\nlift_head=0.004\n"
    f"N_req_head={N_req:.1f}\nse_med={se_med:.6f}\n")
