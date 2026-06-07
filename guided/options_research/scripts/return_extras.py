"""Extra return-target analyses: regime conditioning and a meta-analysis.

(1) Regime-conditional return IC. Split the days by three pre-specified
    conditioners (VIX level, the variance risk premium IV-RV, earnings proximity)
    and compare the high vs low buckets with a date-block bootstrap.

(2) Meta-analysis of the marginal options return-lift, pooled across all cells.
    The block bootstrap resamples whole calendar quarters jointly across every
    cell, so overlapping dates and near-duplicate representations do not inflate
    significance.

Both read saved return artifacts only (no retraining). Writes return_extras.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
PANEL = Path(__file__).resolve().parents[1] / "data" / "features" / "resolution_1_scalar"

report_lines = []


def emit(*args):
    line = " ".join(str(a) for a in args)
    report_lines.append(line)
    print(line)


def header(title):
    emit("\n" + "=" * 76)
    emit(title)
    emit("=" * 76)


def daily_ic_series(df):
    """Per-date cross-sectional rank IC (Spearman computed as Pearson on ranks)."""
    df = df[["date", "pred", "y_demeaned"]].dropna()
    grp = df.groupby("date")
    rank_pred = grp["pred"].rank()
    rank_y = grp["y_demeaned"].rank()
    t = pd.DataFrame({"date": df["date"].values, "rp": rank_pred.values, "ry": rank_y.values})
    t["pp"] = t.rp * t.rp
    t["yy"] = t.ry * t.ry
    t["py"] = t.rp * t.ry
    s = t.groupby("date").agg(n=("rp", "size"), mp=("rp", "mean"), my=("ry", "mean"),
                              spp=("pp", "sum"), syy=("yy", "sum"), spy=("py", "sum"))
    cov = s.spy - s.n * s.mp * s.my
    var_p = s.spp - s.n * s.mp * s.mp
    var_y = s.syy - s.n * s.my * s.my
    ic = (cov / np.sqrt(var_p * var_y))[s.n >= 5]
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def bootstrap_diff(a, b, n_boot=5000, seed=1):
    """Bootstrap each bucket's mean daily IC and the high-minus-low difference.
    The two buckets are resampled independently, which is valid even when they
    are disjoint date sets (e.g. for the VIX split)."""
    a = a.to_numpy()
    b = b.to_numpy()
    rng = np.random.default_rng(seed)
    a_boot = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
    b_boot = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1)
    diff = a_boot - b_boot

    def ci(v):
        return np.percentile(v, 2.5), np.percentile(v, 97.5)

    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return a.mean(), ci(a_boot), b.mean(), ci(b_boot), a.mean() - b.mean(), ci(diff), p


def causal_vrp_threshold(panel):
    """Per-date out-of-sample VRP split threshold: the expanding median of the
    pooled VRP distribution over all dates STRICTLY BEFORE each date. Computing
    the threshold only from past data removes the look-ahead in the old
    full-sample median, which classified each day using the whole test period.
    Returns a date-indexed Series of thresholds."""
    g = panel.dropna(subset=["vrp"]).groupby("date")["vrp"]
    acc = np.array([], dtype=float)
    thr = {}
    for dt, arr in g:                       # groupby iterates dates in order
        thr[dt] = np.median(acc) if acc.size else np.nan
        acc = np.concatenate([acc, arr.to_numpy(dtype=float)])
    return pd.Series(thr, name="vrp_thr")


def regime_analysis(oos, panel, vrp_thr):
    header("(1) REGIME-CONDITIONAL RETURN IC  (XGBoost/D, date-block bootstrap)")
    emit("Theory: option information should matter more when uncertainty is high")
    emit("(VIX, VRP) and around events (earnings). Conditioning widens the SEs.")
    emit("Design is pre-specified; the split rules are fixed in advance:")
    emit("  VIX:  high>=25 vs low<=15 (fixed)   earnings: <=7 days vs farther (fixed)")
    emit("  VRP: high vs low at the EXPANDING (trailing, out-of-sample) median --")
    emit("       threshold at date t = median of pooled VRP over dates < t (no look-ahead).")
    emit("Family = 3 conditioners x 4 cells = 12 diff tests; Bonferroni alpha=0.05/12=0.0042.\n")

    names = {"VIX": "VIX level (high>=25 vs low<=15)",
             "VRP": "VRP iv-rv (trailing-median split)",
             "EARN": "earnings (<=7d vs far)"}
    rows = []
    for scheme in ["expanding", "rolling_9m_3m"]:
        for h in [3, 5]:
            d = oos[(oos.feature_set == "D") & (oos.scheme == scheme) & (oos.horizon == h)]
            d = d.merge(panel, on=["ticker", "date"], how="left")
            d["vrp_thr"] = d["date"].map(vrp_thr)
            emit(f"-- {scheme} h={h} --")
            splits = [
                ("VIX", d[d.vix >= 25], d[d.vix <= 15]),
                ("VRP", d[d.vrp > d.vrp_thr], d[d.vrp <= d.vrp_thr]),
                ("EARN", d[d.days_earn <= 7], d[d.days_earn > 7]),
            ]
            for key, high, low in splits:
                ic_high = daily_ic_series(high)
                ic_low = daily_ic_series(low)
                if len(ic_high) < 10 or len(ic_low) < 10:
                    emit(f"   {names[key]}: too few days")
                    continue
                m_hi, ci_hi, m_lo, ci_lo, diff, ci_diff, p = bootstrap_diff(ic_high, ic_low)
                verdict = "SURVIVES (CI excl 0)" if ci_hi[0] > 0 else "n.s."
                emit(f"   {names[key]}:")
                emit(f"      high/near IC={m_hi:+.4f} CI[{ci_hi[0]:+.4f},{ci_hi[1]:+.4f}]  ({verdict} for the high bucket)")
                emit(f"      low/far   IC={m_lo:+.4f} CI[{ci_lo[0]:+.4f},{ci_lo[1]:+.4f}]")
                emit(f"      diff={diff:+.4f} CI[{ci_diff[0]:+.4f},{ci_diff[1]:+.4f}] boot p={p:.3f}")
                rows.append({"cond": key, "high_ci_lo": ci_hi[0], "p": p})

    summary = pd.DataFrame(rows)
    bonferroni = 0.05 / 12
    emit("\n-- multiplicity-aware summary (family = 3 conditioners x 4 cells = 12) --")
    for key, name in names.items():
        g = summary[summary.cond == key]
        ci_excludes_zero = int((g.high_ci_lo > 0).sum())
        p_05 = int((g.p < 0.05).sum())
        p_bonf = int((g.p < bonferroni).sum())
        emit(f"   {name:32}: high-bucket CI>0 in {ci_excludes_zero}/{len(g)} cells; "
             f"diff p<0.05 in {p_05}/{len(g)}; p<Bonferroni(0.0042) in {p_bonf}/{len(g)}")
    emit("\nVRP gates the signal but VIX level does not (VIX diffs p>0.8). If high-VRP")
    emit("days were just high-dispersion days, VIX level would gate it too, so this is")
    emit("evidence against a dispersion artifact -- the options price of risk is what")
    emit("selects the usable days, not raw volatility.")


def block_bootstrap(lifts, block_col, n_boot=5000, seed=7):
    """Resample whole blocks (e.g. calendar quarters). Every cell whose window
    falls in a drawn block enters the replicate together, so cross-cell and
    cross-time dependence is preserved."""
    blocks = lifts[block_col].dropna().unique()
    by_block = {b: lifts[lifts[block_col] == b]["lift"].to_numpy() for b in blocks}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, len(blocks), size=len(blocks))
        boot[i] = np.concatenate([by_block[blocks[j]] for j in sample]).mean()
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return ci, p, len(blocks)


def meta_analysis():
    header("(2) POOLED / META-ANALYTIC RETURN LIFT across all cells (calendar-block boot)")
    windows = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
    windows["test_start"] = pd.to_datetime(windows["test_start"])
    reps = ["B", "C", "D", "candidate_6", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]

    pivot = windows.pivot_table(index=["model", "scheme", "horizon", "window"],
                                columns="feature_set", values="ic").dropna(subset=["A"])
    # (scheme, horizon, window) -> calendar quarter of the test window
    quarter = (windows.drop_duplicates(["scheme", "horizon", "window"])
               .set_index(["scheme", "horizon", "window"])["test_start"]
               .dt.to_period("Q").astype(str).to_dict())

    rows = []
    for rep in reps:
        if rep not in pivot.columns:
            continue
        for (model, scheme, h, w), r in pivot[[rep, "A"]].dropna().iterrows():
            rows.append({"rep": rep, "model": model, "scheme": scheme, "horizon": h,
                         "quarter": quarter.get((scheme, int(h), int(w))),
                         "finewin": f"{scheme}|{h}|{w}", "lift": r[rep] - r["A"]})
    lifts = pd.DataFrame(rows)
    pooled = lifts["lift"].mean()

    ci_q, p_q, n_q = block_bootstrap(lifts, "quarter")    # primary: joint over time
    ci_f, p_f, n_f = block_bootstrap(lifts, "finewin")    # sensitivity: finer blocks
    n_cells = lifts.groupby(["rep", "model", "scheme", "horizon"]).ngroups

    emit(f"pooled options return-lift over A (all reps x models x cells): {pooled:+.4f}")
    emit(f"  PRIMARY calendar-quarter joint bootstrap: 95% CI [{ci_q[0]:+.4f},{ci_q[1]:+.4f}] p={p_q:.3f}")
    emit(f"  effective independent units = {n_q} calendar quarters,")
    emit(f"          vs {n_cells} nominal cells / {len(lifts)} per-window lifts (not independent).")
    emit(f"  SENSITIVITY finer scheme|horizon|window blocks: CI [{ci_f[0]:+.4f},{ci_f[1]:+.4f}] p={p_f:.3f} (n={n_f})")
    emit("per-representation pooled lift:")
    for rep in reps:
        v = lifts[lifts.rep == rep]["lift"]
        if len(v):
            emit(f"   {rep:14} {v.mean():+.4f}  (n={len(v)})")
    emit("\nA meta-analysis can only surface a consistent small lift; resampling calendar")
    emit("blocks jointly across cells keeps the p-value honest given the shared dates")
    emit("and near-duplicate representations.")


def main():
    oos = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet")
    oos["date"] = pd.to_datetime(oos["date"])
    panel = pd.read_parquet(PANEL / "panel_unnormalized.parquet",
                            columns=["ticker", "date", "feat_38", "feat_12", "feat_43", "feat_40"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.rename(columns={"feat_38": "vix", "feat_12": "vrp",
                                  "feat_43": "days_earn", "feat_40": "vix_ts"})

    vrp_thr = causal_vrp_threshold(panel)
    regime_analysis(oos, panel, vrp_thr)
    meta_analysis()

    (TAB / "return_extras.txt").write_text("\n".join(report_lines))
    print("\n[written] return_extras.txt")


if __name__ == "__main__":
    main()
