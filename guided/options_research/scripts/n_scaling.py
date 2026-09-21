"""Empirical test of the 1/sqrt(N) scaling assumed by the power calculator.

The power section extrapolates the detection threshold across universe sizes with
DT(N) = t* * SE_ref * sqrt(N_ref / N).  That rests on an assumption -- that the
HAC standard error of the mean per-window rank IC scales as 1/sqrt(N) -- which the
60-name study never tests on its own data.  This script tests it.

Method.  Nothing is retrained.  We take the saved out-of-sample predictions, draw
seeded random ticker subsets of size N, recompute the *same* per-window pooled rank
IC and the *same* Newey-West HAC standard error of its mean on each subset, and
compare the realised SE(N) against the 1/sqrt(N) prediction anchored at the full
cross-section.

What this does and does not isolate.  The models were fit once on the full 60-name
panel, so a subset draw is not a simulation of a study that only ever had N names
(that study would also have less training data).  It isolates the evaluation-side
cross-sectional sampling noise, which is exactly the quantity the power calculator
extrapolates, holding the fitted signal fixed.  That is the assumption under test.

Targets are re-demeaned within date over the subset, because a researcher holding
only N names would demean over those N names.  --no-redemean reruns with the
full-universe demeaning held fixed as a sensitivity.

Outputs: results/tables/n_scaling.csv, results/tables/n_scaling.txt,
         results/figures/fig_n_scaling.pdf (+ .png)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

ETFS = ("SPY", "QQQ", "IWM")
N_GRID = (20, 30, 40, 50)          # full cross-section is appended automatically
CELLS = [                          # (feature_set, scheme, horizon)
    ("A", "expanding", 1), ("A", "expanding", 3), ("A", "expanding", 5),
    ("A", "rolling_9m_3m", 1), ("A", "rolling_9m_3m", 3), ("A", "rolling_9m_3m", 5),
    ("D", "expanding", 1), ("D", "expanding", 3), ("D", "expanding", 5),
    ("D", "rolling_9m_3m", 1), ("D", "rolling_9m_3m", 3), ("D", "rolling_9m_3m", 5),
    ("repr_grid", "expanding", 1), ("repr_grid", "expanding", 3),
    ("repr_grid", "expanding", 5), ("repr_grid", "rolling_9m_3m", 1),
    ("repr_grid", "rolling_9m_3m", 3), ("repr_grid", "rolling_9m_3m", 5),
]


def nw_se(x: np.ndarray) -> tuple[float, float, float]:
    """Mean, Newey-West HAC SE of the mean, t.  Identical to posthoc_stats._nw_tstat."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(np.mean(x)) if n else np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    e = x - m
    var = float(np.mean(e * e))
    L = max(1, int(n ** (1.0 / 3.0)))
    for k in range(1, L + 1):
        var += 2.0 * (1.0 - k / (L + 1.0)) * float(np.mean(e[k:] * e[:-k]))
    se = np.sqrt(max(var, 0.0) / n)
    return m, se, (m / se if se > 0 else np.nan)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of tie-averaged ranks == Spearman rho."""
    if len(a) < 11:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def _demean_by(vals: np.ndarray, grp: np.ndarray, n_grp: int) -> np.ndarray:
    """Subtract the group mean of ``vals`` within each integer group code."""
    s = np.bincount(grp, weights=vals, minlength=n_grp)
    c = np.bincount(grp, minlength=n_grp)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return vals - mu[grp]


def window_ics(pred: np.ndarray, y: np.ndarray, wid: np.ndarray, n_win: int,
               date_code: np.ndarray, n_date: int, redemean: bool) -> np.ndarray:
    """Per-window pooled rank IC, the same object the walk-forward tables store."""
    if redemean:
        pred = _demean_by(pred, date_code, n_date)
        y = _demean_by(y, date_code, n_date)
    out = np.full(n_win, np.nan)
    order = np.argsort(wid, kind="stable")
    wid_s, pred_s, y_s = wid[order], pred[order], y[order]
    bounds = np.searchsorted(wid_s, np.arange(n_win + 1))
    for w in range(n_win):
        lo, hi = bounds[w], bounds[w + 1]
        if hi - lo > 10:
            out[w] = _spearman(y_s[lo:hi], pred_s[lo:hi])
    return out


def load_cell(oos: pd.DataFrame, wins: pd.DataFrame, fs: str, sc: str, h: int):
    """Arrays for one cell: pred, y, window id, date code, ticker code."""
    o = oos[(oos.feature_set == fs) & (oos.scheme == sc) & (oos.horizon == h)]
    w = (wins[(wins.feature_set == fs) & (wins.scheme == sc) & (wins.horizon == h)]
         .sort_values("window").reset_index(drop=True))
    if o.empty or w.empty:
        return None
    # map each row to its walk-forward window by test-block date range
    edges = w["test_start"].to_numpy("datetime64[ns]")
    ends = w["test_end"].to_numpy("datetime64[ns]")
    d = o["date"].to_numpy("datetime64[ns]")
    wid = np.searchsorted(edges, d, side="right") - 1
    ok = (wid >= 0) & (wid < len(w))
    ok[ok] &= d[ok] <= ends[wid[ok]]
    o, wid = o[ok], wid[ok]
    tick, tnames = pd.factorize(o["ticker"], sort=True)
    dcode, dnames = pd.factorize(o["date"], sort=True)
    return dict(
        pred=o["pred"].to_numpy(np.float64), y=o["y_demeaned"].to_numpy(np.float64),
        wid=wid, n_win=len(w), tick=tick, tickers=list(tnames),
        dcode=dcode, n_date=len(dnames), saved_ic=w["ic"].to_numpy(np.float64),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40, help="subset draws per (cell, N)")
    ap.add_argument("--no-redemean", action="store_true",
                    help="keep full-universe demeaning instead of re-demeaning on the subset")
    ap.add_argument("--drop-etfs", action="store_true",
                    help="restrict the sampling pool to the 57 single names")
    ap.add_argument("--placebo", choices=["none", "noise", "shuffle"], default="none",
                    help="none: real predictions. noise: replace predictions with i.i.d. "
                         "N(0,1) (true IC = 0, no time-varying signal, so the HAC SE of "
                         "the mean IC must scale as 1/sqrt(N) if the subsampling is "
                         "correct). shuffle: permute predictions within each date.")
    args = ap.parse_args()
    redemean = not args.no_redemean

    oos = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet")
    oos["date"] = pd.to_datetime(oos["date"])
    if args.placebo == "noise":
        oos["pred"] = np.random.default_rng(20260916).standard_normal(len(oos))
    elif args.placebo == "shuffle":
        rng = np.random.default_rng(20260916)
        oos["pred"] = (oos.groupby(["feature_set", "scheme", "horizon", "date"])["pred"]
                       .transform(lambda v: rng.permutation(v.to_numpy())))
    wins = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")
    wins = wins[wins.model == "XGBoost"].copy()
    for c in ("test_start", "test_end"):
        wins[c] = pd.to_datetime(wins[c])

    rows, lines = [], []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("EMPIRICAL vs 1/sqrt(N) SCALING OF THE HAC SE OF THE MEAN RANK IC")
    emit("=" * 78)
    emit(f"seeds per (cell,N) = {args.seeds} | subset re-demeaning = {redemean} | "
         f"sampling pool = {'57 single names' if args.drop_etfs else 'all 60 tickers'} | "
         f"placebo = {args.placebo}")
    emit("Models are NOT retrained: subsets are drawn from saved OOS predictions, so")
    emit("this isolates evaluation-side cross-sectional sampling noise -- the exact")
    emit("quantity DT(N) = t* * SE_ref * sqrt(N_ref/N) extrapolates.")
    emit("")

    for fs, sc, h in CELLS:
        cell = load_cell(oos, wins, fs, sc, h)
        if cell is None:
            emit(f"[skip] {fs}/{sc}/h{h}: no saved predictions")
            continue
        pool = [i for i, t in enumerate(cell["tickers"])
                if not (args.drop_etfs and t in ETFS)]
        n_full = len(pool)

        # full-cross-section reference, and a check against the released table
        base_mask = np.isin(cell["tick"], pool)
        ic_full = window_ics(cell["pred"][base_mask], cell["y"][base_mask],
                             cell["wid"][base_mask], cell["n_win"],
                             cell["dcode"][base_mask], cell["n_date"], redemean)
        m_full, se_full, t_full = nw_se(ic_full)
        recon_err = (np.nanmax(np.abs(ic_full - cell["saved_ic"]))
                     if (n_full == len(cell["tickers"]) and not redemean) else np.nan)

        for N in list(N_GRID) + [n_full]:
            if N > n_full:
                continue
            ses, means = [], []
            for s in range(1 if N < n_full else 1, (args.seeds if N < n_full else 1) + 1):
                rng = np.random.default_rng(1000 * N + s)
                sel = pool if N == n_full else rng.choice(pool, size=N, replace=False)
                mask = np.isin(cell["tick"], sel)
                ic = window_ics(cell["pred"][mask], cell["y"][mask], cell["wid"][mask],
                                cell["n_win"], cell["dcode"][mask], cell["n_date"], redemean)
                m, se, _ = nw_se(ic)
                if np.isfinite(se):
                    ses.append(se)
                    means.append(m)
            if not ses:
                continue
            ses = np.asarray(ses)
            rows.append(dict(
                feature_set=fs, scheme=sc, horizon=h, N=N, n_draws=len(ses),
                se_mean=ses.mean(), se_median=float(np.median(ses)),
                se_p05=float(np.percentile(ses, 5)), se_p95=float(np.percentile(ses, 95)),
                se_pred_sqrtN=se_full * np.sqrt(n_full / N),
                ic_mean=float(np.mean(means)), ic_sd_across_draws=float(np.std(means, ddof=1))
                if len(means) > 1 else np.nan,
                se_full_ref=se_full, n_full=n_full, mean_ic_full=m_full, t_full=t_full,
                recon_max_abs_err=recon_err,
            ))

    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "n_scaling.csv", index=False)

    # ---- aggregate across cells: the headline scaling test ----
    emit("-" * 78)
    emit("Per-N aggregate over %d cells (median across cells of the per-cell median SE)"
         % df.groupby(["feature_set", "scheme", "horizon"]).ngroups)
    emit("-" * 78)
    emit("   N    SE_empirical    SE_pred 1/sqrtN    ratio emp/pred    implied DT(N)=3*SE")
    agg = []
    for N, g in df.groupby("N"):
        emp = float(np.median(g.se_median))
        pred = float(np.median(g.se_pred_sqrtN))
        agg.append(dict(N=N, se_emp=emp, se_pred=pred, ratio=emp / pred, dt_emp=3 * emp,
                        dt_pred=3 * pred))
        emit("  %3d      %.5f          %.5f             %.3f            %.4f"
             % (N, emp, pred, emp / pred, 3 * emp))
    A = pd.DataFrame(agg)
    A.to_csv(TAB / "n_scaling_aggregate.csv", index=False)

    # log-log slope: SE ~ N^beta, 1/sqrt(N) implies beta = -0.5
    sub = A[A.N > 0]
    beta, intercept = np.polyfit(np.log(sub.N.to_numpy(float)),
                                 np.log(sub.se_emp.to_numpy(float)), 1)
    emit("")
    emit("log-log fit of SE on N over the aggregate curve: SE ~ N^(%.3f)" % beta)
    emit("   1/sqrt(N) scaling predicts an exponent of -0.500.")
    # per-cell slopes
    slopes = []
    for k, g in df.groupby(["feature_set", "scheme", "horizon"]):
        if len(g) >= 3:
            b, _ = np.polyfit(np.log(g.N.to_numpy(float)),
                              np.log(g.se_median.to_numpy(float)), 1)
            slopes.append(b)
    slopes = np.asarray(slopes)
    emit("per-cell exponents: n=%d  mean=%.3f  median=%.3f  sd=%.3f  range=[%.3f,%.3f]"
         % (len(slopes), slopes.mean(), np.median(slopes), slopes.std(ddof=1),
            slopes.min(), slopes.max()))
    # variance decomposition: Var(mean IC) = [s2_signal + s2_noise(N)] / n_win
    # so SE(N)^2 = a + b/N, with sqrt(a) the floor no cross-section size removes.
    xx = 1.0 / A.N.to_numpy(float)
    yy = A.se_emp.to_numpy(float) ** 2
    b_var, a_var = np.polyfit(xx, yy, 1)
    r2 = 1.0 - ((yy - (a_var + b_var * xx)) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    floor = np.sqrt(max(a_var, 0.0))
    emit("")
    emit("variance decomposition  SE(N)^2 = a + b/N   (R^2 = %.4f)" % r2)
    emit("   a = %.3e  ->  irreducible SE floor sqrt(a) = %.5f" % (a_var, floor))
    emit("   b = %.3e  ->  N-dependent sampling term" % b_var)
    emit("   implied floor on the t*=3 detection threshold: %.4f" % (3 * floor))
    emit("   (this floor is set by genuine time-variation in the cross-sectional IC")
    emit("    at this design's window count; it falls with MORE CALENDAR TIME, not")
    emit("    with more names.)")
    pd.DataFrame([dict(exponent_loglog=beta, a_var=a_var, b_var=b_var, r2=r2,
                       se_floor=floor, dt_floor=3 * floor,
                       exp_percell_mean=slopes.mean(),
                       exp_percell_median=float(np.median(slopes)),
                       exp_percell_sd=slopes.std(ddof=1),
                       placebo=args.placebo, redemean=redemean,
                       seeds=args.seeds)]).to_csv(TAB / "n_scaling_fit.csv", index=False)
    emit("")
    worst = float(np.max(np.abs(A.ratio - 1.0)))
    emit("largest deviation of empirical from predicted SE across the N grid: %.1f%%"
         % (100 * worst))
    if np.isfinite(df.recon_max_abs_err).any():
        emit("reconstruction check at full N (no re-demeaning): max |rebuilt-saved IC| = %.2e"
             % np.nanmax(df.recon_max_abs_err))
    # required N to resolve a given lift, under each scaling law, at BOTH the
    # t*=3 significance threshold and the 80%-power effect. Conflating these two
    # is the most common error in applied power statements, so both are reported.
    se_ref = float(A.loc[A.N.idxmax(), "se_emp"])
    n_ref = float(A.N.max())
    Z80 = 0.8416212335729143          # Phi^{-1}(0.80)
    MULTS = (("t*=3 detection threshold", 3.0),
             ("80% power at t*=3", 3.0 + Z80))

    # the two floors: no N brings the resolvable effect below mult*sqrt(a)
    emit("-" * 78)
    emit("RESOLUTION FLOORS (from SE^2 = a + b/N; no cross-section size removes them)")
    emit("-" * 78)
    for label, mult in MULTS:
        emit("  floor on the %-26s = %.4f" % (label, mult * floor))

    req_rows = []
    for label, mult in MULTS:
        emit("")
        emit("-" * 78)
        emit("REQUIRED CROSS-SECTION to place the %s at a given effect" % label)
        emit("-" * 78)
        emit("anchored at N=%d, SE=%.5f  (target SE = effect / %.4f)"
             % (n_ref, se_ref, mult))
        emit("  effect  target SE   N under 1/sqrt(N)   N under measured N^%.3f   "
             "SE^2=a+b/N" % beta)
        for lift in (0.002, 0.003, 0.004, 0.015, 0.019, 0.025, 0.036):
            tgt = lift / mult
            n_sqrt = n_ref * (se_ref / tgt) ** 2
            n_emp = n_ref * (se_ref / tgt) ** (1.0 / abs(beta))
            n_dec = (np.nan if tgt ** 2 <= a_var
                     else b_var / (tgt ** 2 - a_var))
            fmt = lambda v: (">1e7" if (np.isfinite(v) and v >= 1e7)
                             else ("unreachable" if not np.isfinite(v) else "%.0f" % v))
            emit("  %.4f  %.5f     %12s       %18s   %12s"
                 % (lift, tgt, fmt(n_sqrt), fmt(n_emp), fmt(n_dec)))
            req_rows.append(dict(criterion=label, t_multiplier=mult, effect=lift,
                                 target_se=tgt, n_sqrtN=n_sqrt, n_measured=n_emp,
                                 n_decomposition=n_dec,
                                 floor_on_effect=mult * floor))
    pd.DataFrame(req_rows).to_csv(TAB / "n_scaling_required_n.csv", index=False)
    ratio_eff = (3 + Z80) / 3.0
    pen_sqrt = ratio_eff ** 2
    pen_meas = ratio_eff ** (1.0 / abs(beta))
    emit("")
    emit("Note. Moving from bare significance to 80% power multiplies the required")
    emit("effect by (3+0.84)/3 = %.2f. That costs a factor %.2f in N under 1/sqrt(N)"
         % (ratio_eff, pen_sqrt))
    emit("but a factor %.2f under the measured N^%.3f law, because a flatter exponent"
         % (pen_meas, beta))
    emit("means each unit of extra resolution buys far fewer names. Reliable detection")
    emit("is therefore substantially more expensive than the conventional rule implies.")
    emit("")
    emit("Reading. A pure-noise placebo (--placebo noise) recovers an exponent of about")
    emit("-0.5 with no variance floor, so the subsampling machinery is sound. On the")
    emit("real predictions the exponent is materially flatter than -0.5 and the")
    emit("SE^2 = a + b/N fit carries a positive intercept. The reason is that the HAC SE")
    emit("of the MEAN per-window IC combines two terms: cross-sectional sampling noise,")
    emit("which does shrink with N, and genuine time-variation in the cross-sectional IC,")
    emit("which does not. The second term puts a floor under the standard error that no")
    emit("universe size removes at a fixed number of evaluation windows. The plain")
    emit("1/sqrt(N) calculator is therefore OPTIMISTIC: it understates the cross-section")
    emit("needed, and for effects below 3*sqrt(a) it reports a finite N where the honest")
    emit("answer is that more names alone will not do it -- more calendar time, or a")
    emit("panel-level estimator with a smaller floor, is required.")

    (TAB / "n_scaling.txt").write_text("\n".join(lines) + "\n")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAVY, GOLD = "#1f3864", "#c8951a"
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.1))

    Ns = A.N.to_numpy(float)
    ax[0].plot(Ns, A.se_pred, "--", color=GOLD, lw=2, label=r"predicted $\propto 1/\sqrt{N}$")
    ax[0].plot(Ns, A.se_emp, "o-", color=NAVY, lw=1.8, ms=6, label="empirical (subset draws)")
    for _, g in df.groupby(["feature_set", "scheme", "horizon"]):
        g = g.sort_values("N")
        ax[0].plot(g.N, g.se_median, color=NAVY, alpha=0.13, lw=0.9, zorder=1)
    ax[0].axhline(floor, color="#7a7a7a", ls=":", lw=1.4,
                  label=r"floor $\sqrt{a}=%.4f$ from SE$^2\!=a+b/N$" % floor)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("cross-section size $N$")
    ax[0].set_ylabel("HAC SE of mean per-window rank IC")
    ax[0].set_title(r"SE scaling: measured $N^{%.2f}$ vs assumed $N^{-0.50}$" % beta,
                    fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(alpha=0.25, which="both", lw=0.4)

    # right panel: extrapolate all three laws past the sampled range
    Nx = np.logspace(np.log10(20), np.log10(3000), 120)
    dt_sqrt = 3 * se_ref * np.sqrt(n_ref / Nx)
    dt_emp = 3 * se_ref * (Nx / n_ref) ** beta
    dt_dec = 3 * np.sqrt(np.maximum(a_var, 0.0) + b_var / Nx)
    ax[1].plot(Nx, dt_sqrt, "--", color=GOLD, lw=2, label=r"$1/\sqrt{N}$ rule (as published)")
    ax[1].plot(Nx, dt_emp, "-", color=NAVY, lw=2, label=r"measured $N^{%.2f}$" % beta)
    ax[1].plot(Nx, dt_dec, "-.", color="#3b7a57", lw=1.8, label=r"SE$^2\!=a+b/N$")
    ax[1].axhline(3 * floor, color="#7a7a7a", ls=":", lw=1.4,
                  label=r"floor $3\sqrt{a}=%.4f$" % (3 * floor))
    ax[1].plot(Ns, 3 * A.se_emp, "o", color=NAVY, ms=5, zorder=5)
    ax[1].axhspan(0.002, 0.004, color=GOLD, alpha=0.22, lw=0,
                  label="observed option return lift")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("cross-section size $N$")
    ax[1].set_ylabel(r"detection threshold at $t^\star=3$")
    ax[1].set_title("Extrapolated threshold under each scaling law", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    ax[1].grid(alpha=0.25, which="both", lw=0.4)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_n_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_n_scaling.png", dpi=200, bbox_inches="tight")
    print(f"\n[written] {TAB/'n_scaling.csv'}, {TAB/'n_scaling.txt'}, "
          f"{FIG/'fig_n_scaling.pdf'}")


if __name__ == "__main__":
    main()
