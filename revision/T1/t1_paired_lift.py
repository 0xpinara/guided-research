"""Task 1 — paired inference for the marginal lift of option representations over Set A.

Inputs (read-only): the corrected Task 0b predictions, revision/out/T0b/predictions/
(git tag task0b-corrected-baseline-2026-10-02; checksums in MANIFEST_large_files.sha256).

Canonical IC (Task 0b): for each window, the mean over its test dates of the daily
cross-sectional Spearman IC across names; a date whose predictions are constant scores 0.
It is recomputed here from the saved predictions and checked against benchmark/windows.csv.

Estimand: Delta_w(X) = IC_w(X) - IC_w(A), for the same target, model, scheme, horizon and
window w. Per cell:
  mean IC(A), mean IC(X), HAC SE of each, corr over windows of IC_w(X) and IC_w(A);
  mean Delta; HAC SE of Delta (Newey-West, Bartlett kernel, lag floor(n^(1/3)), on the
  per-window Delta series, as in Task 0b); paired t = mean / SE;
  DT = 3 x SE (the t > 3 detection threshold) and MDE80 = (3 + 0.8416) x SE;
  raw two-sided p from Student t with n - 1 degrees of freedom (n = windows; the normal p is
  also reported); Holm-adjusted p within the pre-specified family.
Families (fixed before computing any result):
  primary    nested sets {C, D, repr_svi, repr_grid, repr_grid_raw, repr_bkm} x {ElasticNet,
             XGBoost} x 2 schemes x 3 horizons = 72 return tests (the plan's "nested regularised family")
  ols        the same nested sets with OLS: 36 return tests
  nonnested  B and candidate_6 versus A, all three models: 36 return tests. These are not
             marginal lifts: B and candidate_6 do not contain Set A.
  vol_*      the same three families on the volatility target (positive control)
A degenerate cell, where Delta = 0 in every window because both fits are intercept-only,
carries no evidence against H0: its t is undefined and its p is set to 1.
The t > 3 hurdle and Holm significance are separate criteria and are reported separately.

Run:  python revision/T1/t1_paired_lift.py      (about 2 minutes)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "T0b"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, t as tdist  # noqa: E402

from common import nw_tstat, daily_ic, SETS, HORIZONS  # noqa: E402

T0B = ROOT / "revision" / "out" / "T0b"
OUT = ROOT / "revision" / "out" / "T1"
OUT.mkdir(parents=True, exist_ok=True)
NESTED = ["C", "D", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]
NONNESTED = ["B", "candidate_6"]
Z80 = norm.ppf(0.80)          # 0.8416


def window_ics() -> pd.DataFrame:
    rows = []
    for target in ("return", "volatility"):
        for h in HORIZONS:
            for fs in SETS:
                p = pd.read_parquet(T0B / "predictions" / target / f"h{h}" / f"{fs}.parquet",
                                    columns=["model", "scheme", "window", "date", "ticker", "pred", "y"])
                for (m, s, w), g in p.groupby(["model", "scheme", "window"], sort=True):
                    d = daily_ic(g.date.to_numpy(), g.pred.to_numpy(), g.y.to_numpy())
                    rows.append({"target": target, "model": m, "feature_set": fs, "scheme": s, "horizon": h,
                                 "window": int(w), "ic": float(d.ic.mean())})
    return pd.DataFrame(rows)


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return pd.Series(adj).reindex(p.index)


def main() -> None:
    t0 = time.time()
    W = window_ics()
    W.to_csv(OUT / "T1_window_ics.csv", index=False)
    ref = pd.read_csv(T0B / "benchmark" / "windows.csv")
    chk = W.merge(ref[["target", "model", "feature_set", "scheme", "horizon", "window", "ic_daily"]],
                  on=["target", "model", "feature_set", "scheme", "horizon", "window"])
    max_diff = float((chk.ic - chk.ic_daily).abs().max())
    assert len(chk) == len(W) == len(ref), (len(chk), len(W), len(ref))

    pv = W.pivot_table(index=["target", "model", "scheme", "horizon", "window"], columns="feature_set", values="ic")
    rows = []
    for (target, model, scheme, h), g in pv.groupby(level=[0, 1, 2, 3]):
        a = g["A"].to_numpy()
        for fs in NESTED + NONNESTED:
            x = g[fs].to_numpy()
            d = x - a
            n = len(d)
            ma, sea, _ = nw_tstat(a)
            mx, sex, _ = nw_tstat(x)
            md, sed, td = nw_tstat(d)
            degenerate = not (sed > 0)
            if degenerate:
                td, p_t, p_n = np.nan, 1.0, 1.0
            else:
                p_t = float(2 * tdist.sf(abs(td), df=n - 1))
                p_n = float(2 * norm.sf(abs(td)))
            fam = ("primary" if model != "OLS" else "ols") if fs in NESTED else "nonnested"
            rows.append({"target": target, "family": fam if target == "return" else f"vol_{fam}",
                         "model": model, "feature_set": fs, "nested_on_A": fs in NESTED, "scheme": scheme,
                         "horizon": int(h), "n_windows": n, "mean_ic_A": ma, "se_ic_A": sea, "mean_ic_X": mx,
                         "se_ic_X": sex, "corr_icX_icA": float(np.corrcoef(a, x)[0, 1]) if np.std(a) > 0 and np.std(x) > 0 else np.nan,
                         "mean_dic": md, "se_dic": sed, "t_dic": td, "dt_3se": 3 * sed, "mde80": (3 + Z80) * sed,
                         "p_raw": p_t, "p_raw_normal": p_n, "degenerate": degenerate})
    T = pd.DataFrame(rows)
    T["p_holm"] = np.nan
    for fam, g in T.groupby("family"):
        T.loc[g.index, "p_holm"] = holm(g.p_raw)
    T["clears_t3"] = T.t_dic > 3                      # the paper's discovery hurdle (positive lift)
    T["holm_sig_5pct"] = T.p_holm < 0.05               # family-wise 5%, two-sided
    T = T.sort_values(["target", "family", "feature_set", "model", "scheme", "horizon"]).reset_index(drop=True)
    T.to_csv(OUT / "T1_lift_table.csv", index=False)

    L = [f"Task 1 paired lift; per-window ICs recomputed from the Task 0b predictions "
         f"(max |diff| vs Task 0b windows.csv: {max_diff:.2e}; {len(W):,} window ICs)"]
    for fam in ["primary", "ols", "nonnested", "vol_primary", "vol_ols", "vol_nonnested"]:
        g = T[T.family == fam]
        L.append(f"[{fam}] tests {len(g)}; degenerate {int(g.degenerate.sum())}; t > 3: {int(g.clears_t3.sum())}; "
                 f"Holm p < 0.05: {int(g.holm_sig_5pct.sum())}; raw p < 0.05: {int((g.p_raw < 0.05).sum())}; "
                 f"mean dIC {g.mean_dic.mean():+.4f}; t range {np.nanmin(g.t_dic):+.2f} .. {np.nanmax(g.t_dic):+.2f}; "
                 f"min Holm p {g.p_holm.min():.3f}")
    pr = T[T.family == "primary"]
    L.append(f"primary family SEs (median): SE(dIC) {pr.se_dic.median():.4f} vs SE(IC_X) {pr.se_ic_X.median():.4f}, "
             f"SE(IC_A) {pr.se_ic_A.median():.4f}; median corr(IC_X, IC_A) {pr.corr_icX_icA.median():.2f}; "
             f"median DT_delta {pr.dt_3se.median():.4f}, median MDE80_delta {pr.mde80.median():.4f}")
    for m in ("ElasticNet", "XGBoost"):
        g = pr[pr.model == m]
        L.append(f"  {m}: median SE(dIC) {g.se_dic.median():.4f}, SE(IC_X) {g.se_ic_X.median():.4f}, "
                 f"SE(IC_A) {g.se_ic_A.median():.4f}, median corr {g.corr_icX_icA.median():.2f}")
    top = T[T.target == "return"].sort_values("t_dic", ascending=False).head(8)
    L.append("largest return paired t (all families): " + "; ".join(
        f"{r.family}:{r.model}/{r.feature_set}/{r.scheme}/h{r.horizon} dIC {r.mean_dic:+.4f} t {r.t_dic:.2f} "
        f"p {r.p_raw:.4f} Holm {r.p_holm:.3f}" for r in top.itertuples()))
    L.append(f"run time {time.time() - t0:.0f} s")
    (OUT / "T1_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
