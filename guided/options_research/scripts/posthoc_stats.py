"""Post-hoc statistical rigor on the walk-forward grid outputs.

Runs on the CSVs written by run_full_matrix.py (no retraining):

  1. Newey-West/HAC t-statistics on each cell's per-window IC series, and the
     Harvey-Liu-Zhu (2016) multiple-testing hurdle (|t| > 3.0): how many of the
     many model x feature-set x horizon x scheme cells survive.
  2. A cross-trial "deflated Sharpe" benchmark (Bailey & Lopez de Prado 2014,
     expected-maximum form): the highest Sharpe expected from N zero-skill
     trials given the observed cross-trial Sharpe dispersion. Long/short cells
     are flagged as surviving if their Sharpe exceeds that benchmark.
  3. The classifier base rate (fraction of up moves) so directional accuracy
     can be read against the right null rather than 50%.

Outputs: results/tables/posthoc_ic_tstats.csv,
         results/tables/posthoc_deflated_sharpe.csv,
         results/tables/posthoc_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.io_helpers import RESULTS_DIR, FEATURES_DIR  # noqa: E402

TABLES = RESULTS_DIR / "tables"
EULER = 0.5772156649015329
CELL_KEYS = ["model", "feature_set", "scheme", "horizon"]


def _nw_tstat(x: np.ndarray) -> tuple[float, float, float]:
    """Mean, Newey-West HAC standard error of the mean, and t-stat for series x."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(np.mean(x)) if n else np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    e = x - m
    g0 = float(np.mean(e * e))
    L = max(1, int(n ** (1.0 / 3.0)))
    var = g0
    for k in range(1, L + 1):
        gk = float(np.mean(e[k:] * e[:-k]))
        var += 2.0 * (1.0 - k / (L + 1.0)) * gk
    se = np.sqrt(max(var, 0.0) / n)
    t = m / se if se > 0 else np.nan
    return m, se, t


def ic_tstats() -> pd.DataFrame:
    path = TABLES / "full_matrix_walkforward_windows.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for key, grp in df.groupby(CELL_KEYS):
        m, se, t = _nw_tstat(grp["ic"].to_numpy())
        rows.append({
            **dict(zip(CELL_KEYS, key)),
            "n_windows": int(len(grp)),
            "mean_ic": m, "nw_se": se, "nw_tstat": t,
            "survives_5pct": bool(abs(t) > 1.96) if np.isfinite(t) else False,
            "survives_hlz_t3": bool(abs(t) > 3.0) if np.isfinite(t) else False,
        })
    out = pd.DataFrame(rows).sort_values("nw_tstat", ascending=False)
    out.to_csv(TABLES / "posthoc_ic_tstats.csv", index=False)
    return out


def deflated_sharpe() -> pd.DataFrame:
    path = TABLES / "full_matrix_strategy_summary.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    ls = df[df["strategy"] == "long_short_decile"].copy()
    if ls.empty:
        return pd.DataFrame()
    # One Sharpe per trial; use the 15%-spread cost scheme as the canonical trial.
    if "cost_scheme" in ls.columns:
        trials = ls[ls["cost_scheme"].astype(str).str.contains("15")]
        if trials.empty:
            trials = ls
    else:
        trials = ls
    sr = trials["sharpe"].to_numpy(dtype=np.float64)
    sr = sr[np.isfinite(sr)]
    N = len(sr)
    V = float(np.var(sr, ddof=1)) if N > 1 else 0.0
    # Expected maximum Sharpe from N zero-skill trials (Bailey-Lopez de Prado).
    if N > 1 and V > 0:
        sr0 = np.sqrt(V) * (
            (1 - EULER) * norm.ppf(1 - 1.0 / N)
            + EULER * norm.ppf(1 - 1.0 / (N * np.e))
        )
    else:
        sr0 = np.nan
    trials = trials.copy()
    trials["deflation_benchmark_sr0"] = sr0
    trials["excess_over_benchmark"] = trials["sharpe"] - sr0
    trials["survives_deflation"] = trials["sharpe"] > sr0
    keep = CELL_KEYS + ["sharpe", "annualized_return", "max_drawdown",
                        "deflation_benchmark_sr0", "excess_over_benchmark",
                        "survives_deflation"]
    keep = [c for c in keep if c in trials.columns]
    out = trials[keep].sort_values("sharpe", ascending=False)
    out.to_csv(TABLES / "posthoc_deflated_sharpe.csv", index=False)
    return out, sr0, N


def pbo_cscv(scheme: str = "expanding", horizon: int = 3, S: int = 8) -> dict:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2014) on the
    long/short daily P&L. Trials = all (model, feature_set) within one
    (scheme, horizon) group, aligned on common rebalance dates."""
    path = TABLES / "full_matrix_ls_daily.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df = df[(df["scheme"] == scheme) & (df["horizon"] == horizon)]
    if df.empty:
        return {}
    df["trial"] = df["model"] + "/" + df["feature_set"]
    mat = df.pivot_table(index="date", columns="trial", values="portfolio_return").dropna()
    if mat.shape[1] < 3 or mat.shape[0] < 2 * S:
        return {"pbo": np.nan, "n_trials": int(mat.shape[1]), "note": "too few obs/trials"}
    R = mat.to_numpy()
    T, Ntr = R.shape
    blocks = np.array_split(np.arange(T), S)
    from itertools import combinations
    logits = []
    for is_idx in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_idx])
        def sr(rows):
            x = R[rows]; mu = x.mean(0); sd = x.std(0, ddof=1)
            return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
        sr_is, sr_oos = sr(is_rows), sr(oos_rows)
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() / Ntr  # OOS rank of IS-best
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())  # IS-best lands below OOS median
    return {"pbo": pbo, "n_trials": int(Ntr), "n_obs": int(T),
            "n_combinations": int(len(logits)), "scheme": scheme, "horizon": horizon}


def regime_ic(model: str = "XGBoost", feature_set: str = "D",
              scheme: str = "expanding", horizon: int = 3) -> pd.DataFrame:
    """Cross-sectional rank-IC by market regime for the HEADLINE walk-forward
    cell (not the demoted single split). Regimes from the raw panel."""
    path = TABLES / "full_matrix_oos_xgb.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df[(df["model"] == model) & (df["feature_set"] == feature_set)
            & (df["scheme"] == scheme) & (df["horizon"] == horizon)]
    if df.empty:
        return pd.DataFrame()
    try:
        panel = pd.read_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
        panel["date"] = pd.to_datetime(panel["date"])
        reg = panel[["ticker", "date", "feat_38", "feat_43", "feat_21"]].rename(
            columns={"feat_38": "vix", "feat_43": "days_earn", "feat_21": "gex"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(reg, on=["ticker", "date"], how="left")
    except Exception as e:
        print(f"[regime_ic] merge failed: {e}")
        return pd.DataFrame()

    def ic(sub):
        if len(sub) < 30:
            return np.nan
        return float(pd.Series(sub["pred"]).corr(pd.Series(sub["y_demeaned"]), method="spearman"))

    rows = [
        ("VIX", "low (<=15)", ic(df[df.vix <= 15])),
        ("VIX", "normal (15-25)", ic(df[(df.vix > 15) & (df.vix < 25)])),
        ("VIX", "high (>=25)", ic(df[df.vix >= 25])),
        ("Earnings", "near (<=7d)", ic(df[df.days_earn <= 7])),
        ("Earnings", "far (>7d)", ic(df[df.days_earn > 7])),
        ("GEX", "negative", ic(df[df.gex < 0])),
        ("GEX", "positive", ic(df[df.gex >= 0])),
    ]
    out = pd.DataFrame(rows, columns=["regime", "state", "rank_ic"])
    out["model"], out["feature_set"], out["horizon"] = model, feature_set, horizon
    out.to_csv(TABLES / "posthoc_regime_ic.csv", index=False)
    return out


def signal_decay(model: str = "XGBoost", feature_set: str = "D",
                 scheme: str = "expanding") -> pd.DataFrame:
    """Pooled rank-IC and L/S Sharpe by horizon for the headline cell."""
    pp = TABLES / "full_matrix_walkforward_pooled.csv"
    sp = TABLES / "full_matrix_strategy_summary.csv"
    if not pp.exists():
        return pd.DataFrame()
    pool = pd.read_csv(pp)
    pool = pool[(pool.model == model) & (pool.feature_set == feature_set) & (pool.scheme == scheme)]
    rows = []
    strat = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    for h in sorted(pool.horizon.unique()):
        r = pool[pool.horizon == h]
        sh = np.nan
        if not strat.empty:
            s = strat[(strat.model == model) & (strat.feature_set == feature_set)
                      & (strat.scheme == scheme) & (strat.horizon == h)
                      & (strat.strategy == "long_short_decile")]
            if "cost_scheme" in s.columns:
                s = s[s.cost_scheme.astype(str).str.contains("15")]
            sh = float(s["sharpe"].iloc[0]) if len(s) else np.nan
        rows.append({"horizon": int(h), "pooled_ic": float(r.pooled_ic.iloc[0]),
                     "pooled_r2": float(r.pooled_r2.iloc[0]), "ls_sharpe": sh})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "posthoc_signal_decay.csv", index=False)
    return out


def svi_fit_quality() -> dict:
    """SVI calibration diagnostics from the panel (fit error, success rate)."""
    try:
        p = pd.read_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    except Exception:
        return {}
    out = {}
    if "svi_a" in p.columns:
        out["svi_fit_success_rate"] = float(p["svi_a"].notna().mean())
    if "svi_fit_error" in p.columns:
        e = p["svi_fit_error"].dropna()
        if len(e):
            out["svi_fit_rmse_mean"] = float(e.mean())
            out["svi_fit_rmse_median"] = float(e.median())
            out["svi_fit_rmse_p90"] = float(e.quantile(0.90))
    return out


def base_rates() -> dict:
    """Fraction of up-moves per horizon, so accuracy is read against the right null."""
    rates = {}
    try:
        tgt = pd.read_parquet(FEATURES_DIR / "targets.parquet")
        for h in (1, 3, 5):
            col = f"ret_{h}d"
            if col in tgt.columns:
                v = tgt[col].dropna()
                rates[h] = float((v > 0).mean())
    except Exception as e:
        print(f"[base_rates] {e}")
    return rates


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    ic = ic_tstats()
    ds = deflated_sharpe()
    rates = base_rates()
    pbo = pbo_cscv()
    reg = regime_ic()
    decay = signal_decay()
    svi = svi_fit_quality()

    lines = ["Post-hoc rigor summary", "=" * 50, ""]
    if not ic.empty:
        n = len(ic)
        n5 = int(ic["survives_5pct"].sum())
        n3 = int(ic["survives_hlz_t3"].sum())
        lines += [
            f"IC multiple-testing ({n} cells):",
            f"  survive |t|>1.96 (5%): {n5}",
            f"  survive |t|>3.0 (Harvey-Liu-Zhu hurdle): {n3}",
            "  top 5 cells by NW t-stat:",
        ]
        for _, r in ic.head(5).iterrows():
            lines.append(
                f"    {r['model']}/{r['feature_set']}/h{r['horizon']}/{r['scheme']}: "
                f"IC={r['mean_ic']:+.4f} t={r['nw_tstat']:+.2f} (n={r['n_windows']})"
            )
        lines.append("")
    if isinstance(ds, tuple):
        out, sr0, N = ds
        n_surv = int(out["survives_deflation"].sum()) if not out.empty else 0
        lines += [
            f"Deflated Sharpe (N={N} long/short trials):",
            f"  expected-max Sharpe under null SR0 = {sr0:.3f}",
            f"  long/short cells with Sharpe > SR0: {n_surv}",
            "",
        ]
    if rates:
        lines.append("Direction base rate (fraction up):")
        for h, p in rates.items():
            lines.append(f"  {h}-day: {p:.4f}  (accuracy must beat this, not 0.50)")
        lines.append("")
    if pbo:
        lines += [
            "Probability of Backtest Overfitting (CSCV, expanding/3-day L/S):",
            f"  PBO = {pbo.get('pbo', float('nan')):.3f} over {pbo.get('n_trials','?')} trials, "
            f"{pbo.get('n_combinations','?')} splits  (lower is better; >0.5 = overfit)",
            "",
        ]
    if isinstance(reg, pd.DataFrame) and not reg.empty:
        lines.append("Regime rank-IC (headline cell: XGBoost/D/expanding/3-day):")
        for _, r in reg.iterrows():
            lines.append(f"  {r['regime']:9s} {r['state']:16s}: IC={r['rank_ic']:+.4f}")
        lines.append("")
    if isinstance(decay, pd.DataFrame) and not decay.empty:
        lines.append("Signal decay by horizon (XGBoost/D/expanding):")
        for _, r in decay.iterrows():
            lines.append(f"  {int(r['horizon'])}-day: pooled IC={r['pooled_ic']:+.4f}  "
                         f"L/S Sharpe={r['ls_sharpe']:+.2f}")
        lines.append("")
    if svi:
        lines.append("SVI calibration quality:")
        for k, v in svi.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    txt = "\n".join(lines)
    (TABLES / "posthoc_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
