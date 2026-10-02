"""Task 0b.6 — corrected economics on the corrected return predictions.

Search family: the 162 return configurations {OLS, ElasticNet, XGBoost} x 9 feature sets
x 2 schemes x 3 horizons, each traded with the book below. FFNN/TFT are not rerun and are
not part of the family.

Book (each configuration)
  * tradeable universe: the 56 stocks; SPY, QQQ and IWM are excluded (manuscript design).
    The all-names book is also computed, for comparison with the BIR numbers only.
  * every h-th test date (non-overlapping h-day holds, as in BIR): rank by prediction, long
    the top 6 and short the bottom 6 at +-1/6 each (dollar-neutral, gross exposure 2).
  * ties: names tied at a cut-off share the remaining slots equally, so no name is chosen by
    ticker or row order. If all predictions are tied, or the long and short tie groups overlap,
    the book is flat for that period.
  * period P&L = sum w * (h-day raw return) - spread cost - borrow;
    spread cost = sum |dw| x 5 bp x 0.15; borrow = short gross x 50 bp/yr / (252/h) (BIR values).
  * Sharpe = mean / sd(ddof=1) x sqrt(252/h), the arithmetic Sharpe.
Corrections
  * SR0: expected maximum Sharpe of N = 162 zero-skill trials (Bailey & Lopez de Prado 2014),
    from the cross-trial variance of the annualised Sharpes.
  * DSR: PSR(SR0) = Phi((SR - SR0) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2)), per-period
    units, with g3/g4 the skewness/kurtosis of the configuration's own period returns.
  * PBO: CSCV with S = 8 (70 splits), relative rank omega = rank/(N+1) (rank 1 = worst OOS),
    PBO = share of splits with logit(omega) <= 0. (a) family: all 162 configurations on
    calendar-month returns over the months common to all; (b) each scheme x horizon group
    (27 configurations) on its common rebalance dates.

Run after run_grid.py:  python revision/T0b/economics.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis, norm, skew  # noqa: E402

from common import OUT, ETFS  # noqa: E402

TOP, BOTTOM, SPREAD_BPS, EFF, BORROW_BPS = 6, 6, 5.0, 0.15, 50.0
EUL = 0.5772156649015329
ECON = OUT / "economics"
ECON.mkdir(parents=True, exist_ok=True)
KEYS = ["model", "feature_set", "scheme", "horizon"]


def book_weights(pred: np.ndarray) -> tuple[np.ndarray, str]:
    n = len(pred)
    w = np.zeros(n)
    if n < TOP + BOTTOM or np.ptp(pred) <= 1e-14 * max(1.0, np.abs(pred).max()):
        return w, "flat_all_tied"
    s = np.sort(pred)
    v_long, v_short = s[-TOP], s[BOTTOM - 1]
    if v_long <= v_short:
        return w, "flat_overlap"
    above, tie_l = pred > v_long, pred == v_long
    w[above] = 1 / TOP
    w[tie_l] = (TOP - above.sum()) / tie_l.sum() / TOP
    below, tie_s = pred < v_short, pred == v_short
    w[below] = -1 / BOTTOM
    w[tie_s] = -(BOTTOM - below.sum()) / tie_s.sum() / BOTTOM
    return w, "ok"


def run_book(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """df: date, ticker, pred, y_raw for one configuration (already restricted to the universe)."""
    dates = np.sort(df.date.unique())[::h]
    df = df[df.date.isin(dates)]
    prev: dict[str, float] = {}
    rows = []
    for dt, g in df.groupby("date", sort=True):
        w, status = book_weights(g.pred.to_numpy(float))
        cur = dict(zip(g.ticker, w))
        names = set(prev) | set(cur)
        turnover = sum(abs(cur.get(t, 0.0) - prev.get(t, 0.0)) for t in names)
        r = g.y_raw.to_numpy(float)
        long_r, short_r = float(np.sum(np.where(w > 0, w * r, 0))), float(np.sum(np.where(w < 0, -w * r, 0)))
        cost = turnover * SPREAD_BPS / 1e4 * EFF
        borrow = float(-w[w < 0].sum()) * BORROW_BPS / 1e4 / (252 / h)
        rows.append({"date": dt, "ret": long_r - short_r - cost - borrow, "long_leg": long_r, "short_leg": short_r,
                     "turnover": turnover, "spread_cost": cost, "borrow_cost": borrow, "status": status})
        prev = {t: v for t, v in cur.items() if v != 0}
    return pd.DataFrame(rows)


def summarise(per: pd.DataFrame, h: int) -> dict:
    r = per.ret.to_numpy(float)
    ppy = 252 / h
    sd = r.std(ddof=1)
    return {"n_periods": len(r), "sharpe": r.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan,
            "mean_period": r.mean(), "ann_return_arith": r.mean() * ppy, "ann_vol": sd * np.sqrt(ppy),
            "final_dollars": 1000 * np.prod(1 + r), "share_flat": float((per.status != "ok").mean()),
            "avg_turnover": per.turnover.mean(), "long_leg_mean": per.long_leg.mean(),
            "short_leg_mean": per.short_leg.mean(), "skew": skew(r), "kurt": kurtosis(r, fisher=False),
            "never_trades": bool(sd == 0 and np.all(r == 0))}


def sr0(sharpes: np.ndarray) -> float:
    n = len(sharpes)
    return float(np.sqrt(np.var(sharpes, ddof=1)) * ((1 - EUL) * norm.ppf(1 - 1 / n) + EUL * norm.ppf(1 - 1 / (n * np.e))))


def dsr(row, sr0_ann: float) -> float:
    ppy = 252 / row.horizon
    s, s0, T = row.sharpe / np.sqrt(ppy), sr0_ann / np.sqrt(ppy), row.n_periods
    den = 1 - row.skew * s + (row.kurt - 1) / 4 * s ** 2
    return float(norm.cdf((s - s0) * np.sqrt(T - 1) / np.sqrt(den))) if den > 0 else np.nan


def pbo_cscv(R: np.ndarray, S: int = 8) -> dict:
    T, N = R.shape
    blocks = np.array_split(np.arange(T), S)
    logits, oos_ranks = [], []
    for is_b in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in is_b])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_b])

        def sr(rows):
            x = R[rows]
            m, sd = x.mean(0), x.std(0, ddof=1)
            return np.divide(m, sd, out=np.zeros_like(m), where=sd > 0)
        s_is, s_oos = sr(is_rows), sr(oos_rows)
        best = int(np.argmax(s_is))
        rank = 1 + int((s_oos < s_oos[best]).sum())          # 1 = worst out of sample
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        oos_ranks.append(rank)
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_trials": N, "n_obs": T, "n_splits": len(logits),
            "median_oos_rank_of_is_best": float(np.median(oos_ranks))}


def main() -> None:
    files = sorted((OUT / "predictions" / "return").glob("h*/*.parquet"))
    summ, periods = [], []
    for f in files:                                   # one (horizon, feature set) file at a time: low memory
        preds = pd.read_parquet(f, columns=KEYS + ["date", "ticker", "pred", "y_raw"])
        preds["date"] = pd.to_datetime(preds["date"])
        for key, g in preds.groupby(KEYS, sort=True):
            h = int(key[3])
            for universe, sub in (("stocks", g[~g.ticker.isin(ETFS)]), ("all_names", g)):
                per = run_book(sub, h)
                summ.append(dict(zip(KEYS, key)) | {"universe": universe} | summarise(per, h))
                if universe == "stocks":
                    periods.append(per.assign(**dict(zip(KEYS, key))))
        print(f"  {f.parent.name}/{f.name}: done", flush=True)
    S = pd.DataFrame(summ)
    P = pd.concat(periods, ignore_index=True)
    P.to_parquet(ECON / "period_returns_stocks.parquet", index=False)

    fam = S[S.universe == "stocks"].copy()
    assert len(fam) == 162, len(fam)
    # A configuration whose book never trades has 0/0 Sharpe. It is a searched configuration with
    # zero skill, so it enters the family at Sharpe 0 (same convention as IC = 0 for constant forecasts).
    fam["sharpe_family"] = np.where(fam.never_trades, 0.0, fam.sharpe)
    assert fam.sharpe_family.notna().all()
    b = sr0(fam.sharpe_family.to_numpy())
    b_excl = sr0(fam.sharpe.dropna().to_numpy())
    fam["sr0_family"] = b
    fam["dsr"] = [np.nan if r.never_trades else dsr(r, b) for r in fam.itertuples()]
    fam["clears_sr0"] = fam.sharpe_family > b
    # sensitivity: SR0 from the 157 defined Sharpes only (never-trading configurations excluded)
    fam["dsr_sensitivity"] = [np.nan if r.never_trades else dsr(r, b_excl) for r in fam.itertuples()]
    fam["clears_sr0_sensitivity"] = fam.sharpe > b_excl
    S = S.merge(fam[KEYS + ["universe", "sharpe_family", "sr0_family", "dsr", "clears_sr0", "dsr_sensitivity",
                            "clears_sr0_sensitivity"]], on=KEYS + ["universe"], how="left")
    S.to_csv(ECON / "strategy_summary.csv", index=False)

    rows = []
    P["month"] = P.date.dt.to_period("M")
    M = (P.groupby(KEYS + ["month"]).ret.apply(lambda x: np.prod(1 + x) - 1).unstack(KEYS).dropna())
    rows.append({"family": "all 162 (calendar-month returns, common months)"} | pbo_cscv(M.to_numpy()))
    for (s, h), g in P.groupby(["scheme", "horizon"]):
        R = g.pivot_table(index="date", columns=["model", "feature_set"], values="ret").dropna()
        rows.append({"family": f"{s} h={h} (27 configurations)"} | pbo_cscv(R.to_numpy()))
    pbo = pd.DataFrame(rows)
    pbo.to_csv(ECON / "pbo.csv", index=False)

    hl = S[(S.model == "ElasticNet") & (S.feature_set == "D") & (S.scheme == "rolling_9m_3m") & (S.horizon == 5)]
    lines = ["Corrected economics (stocks-only book unless stated; arithmetic Sharpe)",
             f"search family N = {len(fam)} ({int(fam.never_trades.sum())} never trade: Sharpe set to 0); SR0 = {b:.4f}; "
             f"configurations above SR0: {int(fam.clears_sr0.sum())}; DSR > 0.95: {int((fam.dsr > 0.95).sum())}",
             f"sensitivity: SR0 excluding the never-trading configurations (N = {int(fam.sharpe.notna().sum())}) = {b_excl:.4f}; "
             f"configurations above it {int(fam.clears_sr0_sensitivity.sum())}; DSR > 0.95: "
             f"{int((fam.dsr_sensitivity > 0.95).sum())}; headline DSR "
             f"{fam.loc[(fam.model == 'ElasticNet') & (fam.feature_set == 'D') & (fam.scheme == 'rolling_9m_3m') & (fam.horizon == 5), 'dsr_sensitivity'].iloc[0]:.3f}; "
             f"best-configuration DSR {fam.loc[fam.sharpe_family.idxmax(), 'dsr_sensitivity']:.3f}",
             f"best configuration: {fam.loc[fam.sharpe_family.idxmax(), KEYS].to_dict()} Sharpe "
             f"{fam.sharpe_family.max():.4f} (DSR {fam.loc[fam.sharpe_family.idxmax(), 'dsr']:.3f})",
             f"Sharpe distribution over the family: median {fam.sharpe_family.median():.3f}, "
             f"share > 0 {(fam.sharpe_family > 0).mean():.2f}",
             "BIR headline cell ElasticNet / D / rolling_9m_3m / h=5:"]
    for r in hl.itertuples():
        lines.append(f"  {r.universe:9s}: Sharpe {r.sharpe:.4f}, final $ {r.final_dollars:,.0f}, flat periods "
                     f"{r.share_flat:.1%}, DSR {r.dsr if r.universe == 'stocks' else float('nan'):.3f}")
    for r in pbo.itertuples():
        lines.append(f"PBO {r.family}: {r.pbo:.3f} (N={r.n_trials}, T={r.n_obs}, splits={r.n_splits}, "
                     f"median OOS rank of IS-best {r.median_oos_rank_of_is_best:.0f})")
    (ECON / "economics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
