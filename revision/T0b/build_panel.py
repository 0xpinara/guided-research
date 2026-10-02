"""Task 0b.1 — corrected data layer, built only from real project data.

Read-only on data/; writes only to revision/out/T0b/.

Inputs
  data/features/resolution_1_scalar/features_panel.parquet  raw (pre-preprocessing) features + return targets
  data/interim/stock_clean/stock_daily.parquet              CRSP prices/returns + Yahoo VIX/T-bill/sector ETF
  data/interim/merged/<TICKER>.parquet                       n_contracts per ticker-day (after option filters)
  data/interim/options_clean/<TICKER>_options.parquet        cleaned contracts (for feat_25/26/50 recomputation)
  data/features/resolution_2_surface/surface_features_all.parquet   raw-quote IV grid (repr_grid_raw)
  data/raw/wrds/compustat/earnings_dates.parquet             Compustat report dates (rdq)
  revision/out/T0b/raw/yahoo_earnings_dates.csv              Yahoo snapshot (fetch_yahoo_earnings.py)

Outputs (revision/out/T0b/)
  panel/panel_primary.parquet     corrected panel: META excluded, ticker-days with < 50 contracts excluded
  panel/panel_unfiltered.parquet  same corrections, thin option days kept (robustness comparison only)
  data_audit.csv                  every rule with before/after row counts and changed values
  data_checks/*.csv               identifier, earnings-source and BKM validation tables

Corrections (Task 0 finding IDs in brackets)
  R1 [D2]  META removed: the CRSP pull maps "META" to two securities and the Facebook-era
           history of PERMNO 13407 is not on disk; WRDS could not be reached to re-pull it.
  R2 [D3]  feat_28/29/33/34/35/36 recomputed on CRSP prices adjusted only at corporate-action
           events (splits, reverse splits, spin-offs; factor from CRSP prc and ret), so these
           events no longer create jumps. Formulas unchanged; event-free windows keep BIR values.
  R3 [D6]  GOOG earnings dates from Yahoo (GOOGL); feat_43/44 recomputed with the BIR rule and
           feat_51 recomputed with the unchanged BIR function.
  R4 [D4]  feat_25/26 recomputed with the textbook BKM (2003) formulas (see bkm_moments).
  R5 [D5]  feat_50 recomputed with contracts keyed on (expiration, strike, type).
  R6 [D12] realised-volatility targets built on the full CRSP calendar before any row is dropped.
  R7 [D11] no join with the legacy 60/20/20 split file (its 10 buffer dates are kept).
  R8 [D7]  primary panel drops ticker-days with fewer than 50 cleaned contracts.
  R9 [new] feat_43/44/45/51 removed from the canonical panels: they need the next realised
           earnings or ex-dividend date, and no point-in-time announcement record exists.
           Kept only in panel/event_features_expost.parquet for ex-post diagnostics.
           (R3 therefore affects only that diagnostic file.)
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.aggregated_features import _implied_earnings_move  # noqa: E402  (BIR function, unchanged)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXCLUDED_FUTURE_EVENT, write_feature_sets_yaml  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "revision" / "out" / "T0b"
PANEL_DIR, CHECK_DIR = OUT / "panel", OUT / "data_checks"
for d in (PANEL_DIR, CHECK_DIR):
    d.mkdir(parents=True, exist_ok=True)
MIN_CONTRACTS = 50
AUDIT: list[dict] = []


def audit(rule: str, finding: str, description: str, rows_before: int, rows_after: int,
          values_changed: int | None = None, note: str = "") -> None:
    AUDIT.append({"rule": rule, "task0_finding": finding, "description": description,
                  "rows_before": rows_before, "rows_after": rows_after,
                  "values_changed": values_changed, "note": note})
    print(f"[{rule}] {description}: rows {rows_before:,} -> {rows_after:,}"
          + (f", values changed {values_changed:,}" if values_changed is not None else "") + (f" ({note})" if note else ""))


# ---------------------------------------------------------------------------- BKM
def bkm_moments(chain: pd.DataFrame, spot: float, rf: float, tau: float) -> tuple[float, float]:
    """Bakshi, Kapadia & Madan (2003) risk-neutral skewness and kurtosis of the log return.

    One expiry. OTM calls (K > S) and OTM puts (K < S); x = ln(K/S) for both sides:
      V = sum 2(1 - x) / K^2 * Q(K) dK
      W = sum (6x - 3x^2) / K^2 * Q(K) dK
      X = sum (12x^2 - 4x^3) / K^2 * Q(K) dK
      mu   = e^{r tau} - 1 - e^{r tau} V / 2 - e^{r tau} W / 6 - e^{r tau} X / 24
      SKEW = (e^{r tau} W - 3 mu e^{r tau} V + 2 mu^3) / (e^{r tau} V - mu^2)^{3/2}
      KURT = (e^{r tau} X - 4 mu e^{r tau} W + 6 mu^2 e^{r tau} V - 3 mu^4) / (e^{r tau} V - mu^2)^2
    Trapezoidal strike spacing as in the BIR code; the BIR minimum of 5 OTM options is kept.
    """
    puts = chain[(chain.is_call == 0) & (chain.strike < spot)]
    calls = chain[(chain.is_call == 1) & (chain.strike > spot)]
    otm = pd.concat([puts, calls]).sort_values("strike")
    if len(otm) < 5:
        return np.nan, np.nan
    K = otm.strike.to_numpy(float)
    Q = otm.mid_price.to_numpy(float)
    dk = np.empty_like(K)
    dk[0], dk[-1] = K[1] - K[0], K[-1] - K[-2]
    dk[1:-1] = (K[2:] - K[:-2]) / 2
    x = np.log(K / spot)
    base = Q * dk / K ** 2
    V, W, X = (2 * (1 - x) * base).sum(), ((6 * x - 3 * x ** 2) * base).sum(), ((12 * x ** 2 - 4 * x ** 3) * base).sum()
    e = np.exp(rf * tau)
    mu = e - 1 - e * V / 2 - e * W / 6 - e * X / 24
    var = e * V - mu ** 2
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    skew = (e * W - 3 * mu * e * V + 2 * mu ** 3) / var ** 1.5
    kurt = (e * X - 4 * mu * e * W + 6 * mu ** 2 * e * V - 3 * mu ** 4) / var ** 2
    return float(skew), float(kurt)


def bkm_self_test() -> dict:
    """Black-Scholes prices on a wide strike grid: log-return skew ~ 0 and kurtosis ~ 3."""
    from scipy.stats import norm
    S, r, sig, tau = 100.0, 0.02, 0.25, 30 / 365
    K = np.arange(40.0, 200.0, 0.5)
    d1 = (np.log(S / K) + (r + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
    d2 = d1 - sig * np.sqrt(tau)
    call = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    put = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    ch = pd.DataFrame({"strike": np.r_[K, K], "is_call": np.r_[np.ones_like(K), np.zeros_like(K)].astype(int),
                       "mid_price": np.r_[call, put]})
    s, k = bkm_moments(ch, S, r, tau)
    return {"bs_skew": s, "bs_kurt": k}


# ---------------------------------------------------------------------------- options features
def option_features(ticker: str, rf_by_date: pd.Series) -> pd.DataFrame:
    """feat_25/26 (BKM fixed) and feat_50 (fixed contract key) for one ticker."""
    cols = ["date", "expiration", "strike", "is_call", "mid_price", "underlying_price", "dte",
            "open_interest", "volume"]
    o = pd.read_parquet(DATA / "interim" / "options_clean" / f"{ticker}.parquet", columns=cols)
    o["date"] = pd.to_datetime(o["date"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    o = o.sort_values(["date", "expiration", "strike", "is_call"])
    out = []
    prev = None
    for dt, day in o.groupby("date", sort=True):
        # BKM on the single expiry nearest 30 DTE (BIR pooled expiries within +-5 days)
        dte = day.dte.to_numpy()
        best = dte[np.argmin(np.abs(dte - 30))]
        chain = day[day.dte == best]
        spot = float(chain.underlying_price.iloc[0])
        rf = float(rf_by_date.get(dt, 0.0))
        s25, k26 = bkm_moments(chain, spot, rf, best / 365.0) if spot > 0 else (np.nan, np.nan)
        # feat_50: opening volume where OI rose versus the previous date, contracts keyed on expiry date
        key = day[["expiration", "strike", "is_call", "open_interest", "volume"]]
        if prev is None:
            f50 = np.nan
        else:
            m = key.merge(prev, on=["expiration", "strike", "is_call"], how="left", suffixes=("", "_prev"))
            opening = m.open_interest > m.open_interest_prev.fillna(0)
            cv = m.volume[opening & (m.is_call == 1)].sum()
            pv = m.volume[opening & (m.is_call == 0)].sum()
            f50 = pv / cv if cv > 0 else np.nan
        prev = key[["expiration", "strike", "is_call", "open_interest"]]
        out.append((dt, s25, k26, f50))
    df = pd.DataFrame(out, columns=["date", "feat_25", "feat_26", "feat_50"])
    df["ticker"] = ticker
    return df


def implied_earnings_move(ticker: str, days_to_earn: pd.Series) -> pd.DataFrame:
    """feat_51 with the unchanged BIR function, for one ticker."""
    o = pd.read_parquet(DATA / "interim" / "options_clean" / f"{ticker}.parquet",
                        columns=["date", "underlying_price", "moneyness", "dte", "is_call", "mid_price"])
    o["date"] = pd.to_datetime(o["date"])
    rows = []
    for dt, day in o.groupby("date", sort=True):
        rows.append((dt, _implied_earnings_move(day, float(days_to_earn.get(dt, np.nan)))))
    return pd.DataFrame(rows, columns=["date", "feat_51"]).assign(ticker=ticker)


def _opt_job(args):
    return option_features(*args)


# ---------------------------------------------------------------------------- price features
EVENT_THRESHOLD = 1.2   # |log factor| > log(1.2): splits, reverse splits, spin-offs (same rule as Task 0 S11)


def price_features(grp: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """BIR formulas for feat_28/29/33/34/35/36 on a corporate-action-adjusted CRSP price.

    On each date the factor f_t = |prc_{t-1}| (1 + ret_t) / |prc_t| is 1 plus the day's
    dividend yield on ordinary days and the split / distribution ratio on event days.
    Only event days (|log f_t| > log 1.2) are adjusted: prices before an event are divided
    by its factor, so the price is continuous across it. Every window that contains no
    event keeps exactly the BIR value (raw price, dividends not added back).
    """
    grp = grp.sort_values("date")
    p = grp.prc.abs().astype(float).to_numpy()
    r = grp.ret.astype(float).to_numpy()
    f = np.ones(len(p))
    with np.errstate(divide="ignore", invalid="ignore"):
        f[1:] = p[:-1] * (1 + r[1:]) / p[1:]
    is_event = np.isfinite(f) & (np.abs(np.log(np.where(f > 0, f, 1.0))) > np.log(EVENT_THRESHOLD))
    f = np.where(is_event, f, 1.0)
    later = np.r_[np.cumprod(f[::-1])[::-1][1:], 1.0]       # product of the factors of events after t
    price = pd.Series(p / later, index=grp.index)
    events = [{"ticker": grp.ticker.iloc[0], "date": grp.date.iloc[i], "factor": float(f[i]),
               "prc_prev": float(p[i - 1]), "prc": float(p[i]), "ret": float(r[i])}
              for i in np.flatnonzero(is_event)]
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()    # noqa: E731
    out = pd.DataFrame({"ticker": grp.ticker.values, "date": grp.date.values})
    out["feat_28"] = price.pct_change(5).values
    out["feat_29"] = price.pct_change(10).values
    macd = (ema(price, 12) - ema(price, 26)) / price.replace(0, np.nan)
    out["feat_33"] = macd.values
    out["feat_34"] = ema(macd, 9).values
    sma20, std20 = price.rolling(20, min_periods=20).mean(), price.rolling(20, min_periods=20).std()
    out["feat_35"] = np.where(std20 > 0, (price - sma20) / (2 * std20), 0)
    sma50 = price.rolling(50, min_periods=50).mean()
    out["feat_36"] = ((price - sma50) / sma50.replace(0, np.nan)).values
    return out, events


def days_to_next(dates: pd.Series, events: np.ndarray) -> np.ndarray:
    """BIR rule (clean_stocks._compute_earnings_proximity): days to the next event on or after the date."""
    events = np.sort(pd.to_datetime(events).values)
    i = np.searchsorted(events, dates.values)
    out = np.full(len(dates), np.nan)
    ok = i < len(events)
    out[ok] = (events[i[ok]] - dates.values[ok]) / np.timedelta64(1, "D")
    return out


def rv_target(stock: pd.DataFrame, h: int) -> pd.Series:
    """Forward realised volatility over [t+1, t+h] on the full CRSP calendar of each ticker."""
    out = pd.Series(np.nan, index=stock.index)
    for _, g in stock.groupby("ticker"):
        g = g.sort_values("date")
        sq = np.log1p(g.ret.astype(float)) ** 2
        fwd = sq.shift(-1)[::-1].rolling(h, min_periods=h).sum()[::-1]   # sum of r_{t+1}^2 .. r_{t+h}^2
        out.loc[g.index] = np.sqrt(fwd.values)
    return out


def main() -> None:
    t0 = time.time()
    fp = pd.read_parquet(DATA / "features/resolution_1_scalar/features_panel.parquet")
    fp = fp.drop(columns=[c for c in fp.columns if c.startswith("__")])
    fp["date"] = pd.to_datetime(fp["date"])
    st = pd.read_parquet(DATA / "interim/stock_clean/stock_daily.parquet")
    st = st.drop(columns=[c for c in st.columns if c.startswith("__")])
    st["date"] = pd.to_datetime(st["date"])
    n0 = len(fp)
    audit("R0", "", "start: BIR raw feature panel (features_panel.parquet)", n0, n0)
    assert len(st) == n0 and fp[["ticker", "date"]].merge(st[["ticker", "date"]]).shape[0] == n0

    # ---------------- identifier checks (all tickers) ----------------
    crsp_ids = st.groupby("ticker").permno.agg(["nunique", "min", "max"]).reset_index()
    crsp_ids.to_csv(CHECK_DIR / "crsp_permno_per_ticker.csv", index=False)
    under = []
    for t in sorted(st.ticker.unique()):
        o = pd.read_parquet(DATA / "interim/options_clean" / f"{t}.parquet", columns=["date", "underlying_price"])
        o["date"] = pd.to_datetime(o["date"])
        u = o.groupby("date").underlying_price.median().rename("opt_underlying").reset_index()
        m = st.loc[st.ticker == t, ["date", "prc"]].merge(u, on="date")
        m["ratio"] = m.opt_underlying / m.prc.abs()
        under.append({"ticker": t, "days_compared": len(m), "median_ratio": m.ratio.median(),
                      "days_off_by_5pct": int((np.abs(m.ratio - 1) > 0.05).sum()),
                      "first_off": m.loc[np.abs(m.ratio - 1) > 0.05, "date"].min(),
                      "last_off": m.loc[np.abs(m.ratio - 1) > 0.05, "date"].max()})
    under = pd.DataFrame(under)
    under.to_csv(CHECK_DIR / "option_underlying_vs_crsp_price.csv", index=False)
    earn = pd.read_parquet(DATA / "raw/wrds/compustat/earnings_dates.parquet")
    earn["rdq"] = pd.to_datetime(earn["rdq"])
    g = earn.groupby("ticker").agg(gvkeys=("gvkey", "nunique"), n_rdq=("rdq", "count"),
                                   max_gap_days=("rdq", lambda s: s.sort_values().diff().dt.days.max()))
    g.reset_index().to_csv(CHECK_DIR / "compustat_earnings_per_ticker.csv", index=False)
    print("identifier checks: tickers with >1 PERMNO:", crsp_ids[crsp_ids["nunique"] > 1].ticker.tolist(),
          "| option-vs-CRSP price mismatches >5%:", under[under.days_off_by_5pct > 0][["ticker", "days_off_by_5pct"]].values.tolist(),
          "| Compustat tickers with >1 GVKEY:", g[g.gvkeys > 1].index.tolist())

    # ---------------- R1: META ----------------
    n = len(fp)
    fp, st = fp[fp.ticker != "META"].copy(), st[st.ticker != "META"].copy()
    audit("R1", "D2", "drop META (CRSP 'META' = PERMNO 21413 then 13407; FB-era 13407 not on disk; WRDS unreachable)",
          n, len(fp), note="the 148 PERMNO-21413 rows (2021-06-30..2022-01-28) carry another security's returns and "
                           "stock features and have no option data (n_contracts = 0); Meta itself starts 2022-06-09")

    # ---------------- R2: corporate-action-adjusted price features ----------------
    parts = [price_features(gg) for _, gg in st.groupby("ticker")]
    pf = pd.concat([p_ for p_, _ in parts], ignore_index=True)
    events = pd.DataFrame([e for _, ev in parts for e in ev])
    events.to_csv(CHECK_DIR / "corporate_action_events.csv", index=False)
    print(f"  corporate-action events adjusted: {len(events)} -> "
          + ", ".join(f"{r.ticker} {pd.Timestamp(r.date).date()} x{r.factor:.3f}" for r in events.itertuples()))
    merged = fp[["ticker", "date"] + list(pf.columns[2:])].merge(pf, on=["ticker", "date"], suffixes=("_old", ""))
    for c in pf.columns[2:]:
        old, new = merged[f"{c}_old"], merged[c]
        changed = int((~np.isclose(old, new, rtol=1e-9, atol=1e-12, equal_nan=True)).sum())
        large = int((np.abs(old - new) > 0.01).sum())
        audit("R2", "D3", f"recompute {c} on the corporate-action-adjusted price", len(fp), len(fp), changed,
              note=f"{large:,} of the {changed:,} changed rows move by more than 0.01; unchanged rows are identical to BIR")
    fp = fp.drop(columns=list(pf.columns[2:])).merge(pf, on=["ticker", "date"], how="left")

    # ---------------- R3: GOOG earnings ----------------
    y = pd.read_csv(OUT / "raw" / "yahoo_earnings_dates.csv", parse_dates=["announce_date"])
    val = []
    for sym in sorted(set(y.symbol) - {"GOOGL"}):
        ys = set(y.loc[(y.symbol == sym) & y.announce_date.between("2016-01-01", "2024-12-31"), "announce_date"])
        cs = set(earn.loc[(earn.ticker == sym) & earn.rdq.between("2016-01-01", "2024-12-31"), "rdq"])
        exact = len(ys & cs)
        within1 = sum(any(abs((a - b).days) <= 1 for b in ys) for a in cs)
        val.append({"ticker": sym, "compustat_dates": len(cs), "yahoo_dates": len(ys),
                    "exact_matches": exact, "compustat_dates_within_1_day_of_yahoo": within1})
    val = pd.DataFrame(val)
    val.to_csv(CHECK_DIR / "earnings_source_validation.csv", index=False)
    print("  Yahoo vs Compustat report dates 2016-2024:", int(val.exact_matches.sum()), "exact /",
          int(val.compustat_dates_within_1_day_of_yahoo.sum()), "within 1 day /", int(val.compustat_dates.sum()), "Compustat dates")
    gd = y.loc[y.symbol == "GOOGL", "announce_date"].drop_duplicates().sort_values().values
    m_goog = fp.ticker == "GOOG"
    days = days_to_next(fp.loc[m_goog, "date"], gd)
    before43 = fp.loc[m_goog, "feat_43"].nunique()
    fp.loc[m_goog, "feat_43"] = days
    fp.loc[m_goog, "feat_44"] = (days <= 7).astype(int)
    audit("R3", "D6", "GOOG feat_43/44 from Yahoo GOOGL announcement dates (BIR rule)", len(fp), len(fp),
          int(m_goog.sum()), note=f"distinct feat_43 values {before43} -> {pd.Series(days).nunique()}")
    f51 = implied_earnings_move("GOOG", pd.Series(days, index=fp.loc[m_goog, "date"].values))
    old51 = fp.loc[m_goog, "feat_51"].notna().sum()
    fp = fp.merge(f51.rename(columns={"feat_51": "f51_goog"}), on=["ticker", "date"], how="left")
    fp.loc[fp.ticker == "GOOG", "feat_51"] = fp.loc[fp.ticker == "GOOG", "f51_goog"]
    fp = fp.drop(columns="f51_goog")
    audit("R3", "D6", "GOOG feat_51 recomputed with the BIR function and the Yahoo dates", len(fp), len(fp),
          int(fp.loc[fp.ticker == "GOOG", "feat_51"].notna().sum()), note=f"non-missing before {old51}")

    # ---------------- R4/R5: options features ----------------
    bt = bkm_self_test()
    print(f"  BKM self-test on Black-Scholes prices: skew {bt['bs_skew']:+.4f} (expect ~0), kurt {bt['bs_kurt']:.4f} (expect ~3)")
    pd.DataFrame([bt]).to_csv(CHECK_DIR / "bkm_self_test.csv", index=False)
    rf = st.drop_duplicates("date").set_index("date").tbill_rate.fillna(0.0) / 100.0
    tickers = sorted(fp.ticker.unique())
    with Pool(7) as pool:
        of = pd.concat(pool.map(_opt_job, [(t, rf) for t in tickers]), ignore_index=True)
    cmp_ = fp[["ticker", "date", "feat_25", "feat_26", "feat_50", "feat_01"]].merge(of, on=["ticker", "date"], how="left",
                                                                                      suffixes=("_old", ""))
    for c, fid in (("feat_25", "D4"), ("feat_26", "D4"), ("feat_50", "D5")):
        changed = int((~np.isclose(cmp_[f"{c}_old"], cmp_[c], equal_nan=True)).sum())
        corr = cmp_[[f"{c}_old", c]].corr(method="spearman").iloc[0, 1]
        audit("R4" if fid == "D4" else "R5", fid, f"recompute {c}", len(fp), len(fp), changed,
              note=f"rank corr old vs new {corr:.3f}")
    stocks = ~cmp_.ticker.isin(["SPY", "QQQ", "IWM"])
    eq_old = np.isclose(cmp_.feat_50_old[stocks], cmp_.feat_01[stocks]).mean()
    eq_new = np.isclose(cmp_.feat_50[stocks], cmp_.feat_01[stocks]).mean()
    print(f"  feat_50 == feat_01 on stock rows: before {eq_old:.4f}, after {eq_new:.4f}")
    cmp_.to_parquet(CHECK_DIR / "option_feature_old_vs_new.parquet", index=False)
    fp = fp.drop(columns=["feat_25", "feat_26", "feat_50"]).merge(of, on=["ticker", "date"], how="left")

    # ---------------- R6: targets ----------------
    st = st.sort_values(["ticker", "date"])
    for h in (1, 3, 5):
        st[f"rv_{h}d"] = rv_target(st, h)
        comp = st.groupby("ticker").ret.transform(lambda s: (1 + s).shift(-1)[::-1].rolling(h).apply(np.prod, raw=True)[::-1] - 1)
        chk = fp[["ticker", "date", f"ret_{h}d"]].merge(st[["ticker", "date"]].assign(c=comp.values), on=["ticker", "date"])
        print(f"  check: ret_{h}d equals compounded CRSP returns over t+1..t+h: max |diff| "
              f"{np.nanmax(np.abs(chk[f'ret_{h}d'] - chk.c)):.2e}")
    fp = fp.merge(st[["ticker", "date", "rv_1d", "rv_3d", "rv_5d"]], on=["ticker", "date"], how="left")
    audit("R6", "D12", "realised-vol targets rv_1d/3d/5d built on the full CRSP calendar", len(fp), len(fp),
          note="BIR built RV after the split join (480 spliced 5-day windows)")
    audit("R7", "D11", "no join with the legacy split file (its 10 buffer dates retained)", len(fp), len(fp))

    # ---------------- raw grid ----------------
    rg = pd.read_parquet(DATA / "features/resolution_2_surface/surface_features_all.parquet")
    rg["date"] = pd.to_datetime(rg["date"])
    rg_cols = [c for c in rg.columns if c.startswith(("iv_surf_", "surface_"))]
    rg = rg[["ticker", "date"] + rg_cols].drop_duplicates(["ticker", "date"])
    fp = fp.merge(rg, on=["ticker", "date"], how="left")

    # ---------------- R9: remove future-event features ----------------
    ev_cols = sorted(EXCLUDED_FUTURE_EVENT)
    fp[["ticker", "date"] + ev_cols].to_parquet(PANEL_DIR / "event_features_expost.parquet", index=False)
    fp = fp.drop(columns=ev_cols)
    audit("R9", "new (2026-10-01)", "remove future-event features " + ", ".join(ev_cols) + " from the canonical panels",
          len(fp), len(fp), note="they use realised next earnings / ex-dividend dates with no point-in-time "
                                 "announcement record; kept only in panel/event_features_expost.parquet "
                                 "(ex-post descriptive diagnostics), no replacement features added")
    write_feature_sets_yaml(OUT / "feature_sets_canonical.yaml")

    # ---------------- R8: 50-contract rule ----------------
    nc = []
    for t in tickers:
        m = pd.read_parquet(DATA / "interim/merged" / f"{t}.parquet", columns=["date", "n_contracts"])
        m["date"] = pd.to_datetime(m["date"])
        nc.append(m.assign(ticker=t))
    fp = fp.merge(pd.concat(nc), on=["ticker", "date"], how="left")
    fp["n_contracts"] = fp.n_contracts.fillna(0).astype(int)
    fp = fp.sort_values(["ticker", "date"]).reset_index(drop=True)
    fp.to_parquet(PANEL_DIR / "panel_unfiltered.parquet", index=False)
    thin = fp.n_contracts < MIN_CONTRACTS
    thin_by = fp[thin].groupby("ticker").size().rename("thin_days").reset_index()
    thin_by.to_csv(CHECK_DIR / "thin_days_by_ticker.csv", index=False)
    prim = fp[~thin].reset_index(drop=True)
    audit("R8", "D7", f"primary panel drops ticker-days with fewer than {MIN_CONTRACTS} cleaned contracts",
          len(fp), len(prim), note=f"{int(thin.sum()):,} ticker-days, {int((thin_by.thin_days > 0).sum())} tickers; unfiltered panel kept for robustness")
    prim.to_parquet(PANEL_DIR / "panel_primary.parquet", index=False)
    audit("END", "", f"panel_primary: {prim.ticker.nunique()} tickers, {prim.date.nunique()} dates, "
          f"{prim.date.min().date()}..{prim.date.max().date()}", len(prim), len(prim))
    pd.DataFrame(AUDIT).to_csv(OUT / "data_audit.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
