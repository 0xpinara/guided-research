#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(days=14)


def days_to_next_event(dates: pd.Series, event_dates: pd.DatetimeIndex) -> pd.Series:
    event_dates = event_dates.sort_values()
    vals: list[float] = []
    for d in dates:
        i = event_dates.searchsorted(d, side="left")
        if i >= len(event_dates):
            vals.append(np.nan)
        else:
            vals.append(float((event_dates[i] - d).days))
    return pd.Series(vals, index=dates.index)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def get_close(symbol: str, start: str = "2008-01-01") -> pd.Series:
    df = yf.download(symbol, start=start, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")["close"]


def get_underlying_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "dividend_amount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    tk = yf.Ticker(symbol)
    div = tk.dividends.reset_index().rename(columns={"Date": "date", "Dividends": "dividend_amount"})
    if len(div) > 0:
        div["date"] = pd.to_datetime(div["date"], utc=True, errors="coerce")
    else:
        div = pd.DataFrame({"date": [], "dividend_amount": []})
    out = df.merge(div[["date", "dividend_amount"]], on="date", how="left")
    out["dividend_amount"] = out["dividend_amount"].fillna(0.0)
    return out[["date", "open", "high", "low", "close", "volume", "dividend_amount"]]


def bkm_style_moments(day_df: pd.DataFrame, rf: float, q: float) -> tuple[float, float]:
    x = day_df[(day_df["bid"] > 0) & (day_df["ask"] > 0) & (day_df["dte"] >= 7) & (day_df["strike"] > 0)].copy()
    if x.empty:
        return np.nan, np.nan
    # near 30D slice
    exp = (
        x.groupby("expiration", as_index=False)["dte"]
        .median()
        .assign(dist=lambda d: (d["dte"] - 30).abs())
        .sort_values("dist")
    )
    if exp.empty:
        return np.nan, np.nan
    x = x[x["expiration"] == exp.iloc[0]["expiration"]].copy()
    if x.empty:
        return np.nan, np.nan
    s = float(x["under_close"].iloc[0])
    t = max(float(x["dte"].median()) / 365.25, 1e-6)
    fwd = s * math.exp((rf - q) * t)
    x["mid"] = ((x["bid"] + x["ask"]) / 2.0).astype(float)
    calls = x[x["type"] == "call"][["strike", "mid"]].groupby("strike", as_index=False)["mid"].mean()
    puts = x[x["type"] == "put"][["strike", "mid"]].groupby("strike", as_index=False)["mid"].mean()
    if calls.empty or puts.empty:
        return np.nan, np.nan
    kset = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
    if len(kset) < 6:
        return np.nan, np.nan
    c_map = dict(zip(calls["strike"], calls["mid"]))
    p_map = dict(zip(puts["strike"], puts["mid"]))
    rows: list[tuple[float, float]] = []
    for k in kset:
        qk = p_map[k] if k < fwd else c_map[k]
        if np.isfinite(qk) and qk > 0:
            rows.append((float(k), float(qk)))
    if len(rows) < 6:
        return np.nan, np.nan
    arr = np.array(rows)
    k = arr[:, 0]
    qk = arr[:, 1]
    ord_idx = np.argsort(k)
    k, qk = k[ord_idx], qk[ord_idx]
    w = qk / (k**2)
    den = np.trapezoid(w, k)
    if not np.isfinite(den) or den <= 0:
        return np.nan, np.nan
    w = w / den
    xlog = np.log(k / fwd)
    mu = np.trapezoid(w * xlog, k)
    c2 = np.trapezoid(w * (xlog - mu) ** 2, k)
    c3 = np.trapezoid(w * (xlog - mu) ** 3, k)
    c4 = np.trapezoid(w * (xlog - mu) ** 4, k)
    if c2 <= 0:
        return np.nan, np.nan
    return float(c3 / (c2**1.5)), float(c4 / (c2**2))


def sector_etf_for_ticker(ticker: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "AVGO", "AMD", "PLTR"}
    comm = {"GOOGL", "META", "NFLX"}
    cons = {"AMZN", "TSLA"}
    fin = {"JPM"}
    broad = {"SPY", "QQQ"}
    if ticker in tech:
        return "XLK"
    if ticker in comm:
        return "XLC"
    if ticker in cons:
        return "XLY"
    if ticker in fin:
        return "XLF"
    if ticker in broad:
        return ticker
    return "SPY"


def build_ticker_features(
    ticker: str,
    base_dir: Path,
    market_series: dict[str, pd.Series],
    earnings_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    opt = pd.read_parquet(base_dir / ticker.lower() / "options.parquet")
    und = pd.read_parquet(base_dir / ticker.lower() / "underlying.parquet")
    opt["date"] = pd.to_datetime(opt["date"], utc=True)
    opt["expiration"] = pd.to_datetime(opt["expiration"], utc=True)
    if und.empty or und["date"].isna().all():
        start = str(opt["date"].min().date()) if len(opt) else "2008-01-01"
        end = str((opt["date"].max() + pd.Timedelta(days=5)).date()) if len(opt) else None
        und = get_underlying_ohlcv(ticker, start=start, end=end)
    und["date"] = pd.to_datetime(und["date"], utc=True, errors="coerce")
    if "dividend_amount" not in und.columns:
        und["dividend_amount"] = 0.0
    if "volume" not in und.columns:
        und["volume"] = np.nan
    if "high" not in und.columns:
        und["high"] = np.nan
    if "low" not in und.columns:
        und["low"] = np.nan
    und = und.sort_values("date").drop_duplicates("date").set_index("date")

    opt = opt.merge(
        und.reset_index()[["date", "close", "volume"]],
        on="date",
        how="left",
        suffixes=("", "_under"),
    )
    if "close_under" in opt.columns:
        opt = opt.rename(columns={"close_under": "under_close"})
    elif "close" in opt.columns:
        opt = opt.rename(columns={"close": "under_close"})
    if "volume_under" in opt.columns:
        opt = opt.rename(columns={"volume_under": "under_volume"})

    opt["dte"] = (opt["expiration"] - opt["date"]).dt.days
    opt["s_over_k"] = opt["under_close"] / opt["strike"].replace(0, np.nan)
    opt["k_over_s"] = opt["strike"] / opt["under_close"].replace(0, np.nan)
    opt["mid"] = ((opt["bid"].fillna(0.0) + opt["ask"].fillna(0.0)) / 2.0).replace(0.0, np.nan)
    opt["spread_pct"] = (opt["ask"] - opt["bid"]) / opt["mid"]
    last_in_range = (opt["last"] >= (opt["bid"] * 0.9)) & (opt["last"] <= (opt["ask"] * 1.1))
    filt = (
        (opt["open_interest"] > 10)
        & (opt["bid"] > 0)
        & (opt["dte"] >= 7)
        & (opt["s_over_k"].between(0.8, 1.2))
        & last_in_range
    )
    f = opt[filt].copy()

    d = pd.DataFrame(index=und.index.copy())
    d["ticker"] = ticker
    call = f[f["type"] == "call"]
    put = f[f["type"] == "put"]

    d["feat_01_put_call_volume_ratio"] = ratio(put.groupby("date")["volume"].sum(), call.groupby("date")["volume"].sum())
    d["feat_02_put_call_open_interest_ratio"] = ratio(put.groupby("date")["open_interest"].sum(), call.groupby("date")["open_interest"].sum())
    d["feat_03_options_to_stock_volume_ratio"] = f.groupby("date")["volume"].sum() * 100.0 / und["volume"].replace(0, np.nan)
    d["feat_04_net_delta_exposure"] = (
        call.assign(x=call["delta"] * call["open_interest"]).groupby("date")["x"].sum()
        - put.assign(x=put["delta"].abs() * put["open_interest"]).groupby("date")["x"].sum()
    )
    d["feat_05_volume_weighted_avg_iv"] = (
        f.assign(x=f["volume"] * f["implied_volatility"]).groupby("date")["x"].sum()
        / f.groupby("date")["volume"].sum().replace(0, np.nan)
    )
    atm = f[f["dte"].between(20, 40)].copy()
    atm["atm_dist"] = (atm["s_over_k"] - 1.0).abs()
    atm = atm.sort_values(["date", "atm_dist"]).groupby("date").head(5)
    d["feat_06_atm_iv"] = (
        atm.assign(x=atm["volume"] * atm["implied_volatility"]).groupby("date")["x"].sum()
        / atm.groupby("date")["volume"].sum().replace(0, np.nan)
    )
    mn = d["feat_06_atm_iv"].rolling(252, min_periods=20).min()
    mx = d["feat_06_atm_iv"].rolling(252, min_periods=20).max()
    d["feat_07_iv_rank_52w_percentile"] = ratio(d["feat_06_atm_iv"] - mn, mx - mn)

    c25 = call.assign(dd=(call["delta"] - 0.25).abs()).sort_values(["date", "dd"]).groupby("date").head(1)
    p25 = put.assign(dd=(put["delta"] + 0.25).abs()).sort_values(["date", "dd"]).groupby("date").head(1)
    d["feat_08_iv_skew_25delta"] = p25.set_index("date")["implied_volatility"] - c25.set_index("date")["implied_volatility"]

    atm30 = f[f["dte"].between(20, 40)].copy()
    atm90 = f[f["dte"].between(75, 105)].copy()
    atm30["dd"] = (atm30["s_over_k"] - 1.0).abs()
    atm90["dd"] = (atm90["s_over_k"] - 1.0).abs()
    atm30 = atm30.sort_values(["date", "dd"]).groupby("date").head(1).set_index("date")
    atm90 = atm90.sort_values(["date", "dd"]).groupby("date").head(1).set_index("date")
    d["feat_09_iv_term_structure_slope"] = atm30["implied_volatility"] - atm90["implied_volatility"]
    d["feat_10_atm_iv_change_1d"] = d["feat_06_atm_iv"].diff(1)
    d["feat_11_atm_iv_change_5d"] = d["feat_06_atm_iv"].diff(5)
    logret = np.log(und["close"] / und["close"].shift(1))
    hv30 = logret.rolling(30, min_periods=20).std() * math.sqrt(252)
    d["feat_12_iv_minus_realized_vol_30d"] = d["feat_06_atm_iv"] - hv30

    total_vol = f.groupby("date")["volume"].sum()
    total_oi = f.groupby("date")["open_interest"].sum()
    d["feat_13_volume_spike_indicator"] = total_vol / total_vol.rolling(20, min_periods=5).mean()
    d["feat_14_open_interest_change_1d"] = total_oi.diff(1)
    otm_call = f[(f["type"] == "call") & (f["s_over_k"] < 0.95)]
    otm_put = f[(f["type"] == "put") & (f["k_over_s"] < 0.95)]
    atm_call = f[(f["type"] == "call") & ((f["s_over_k"] - 1.0).abs() <= 0.02)]
    atm_put = f[(f["type"] == "put") & ((f["s_over_k"] - 1.0).abs() <= 0.02)]
    d["feat_15_otm_volume_concentration"] = ratio(
        otm_call.groupby("date")["volume"].sum().add(otm_put.groupby("date")["volume"].sum(), fill_value=0.0),
        total_vol,
    )
    d["feat_16_call_volume_skew"] = ratio(otm_call.groupby("date")["volume"].sum(), atm_call.groupby("date")["volume"].sum())
    d["feat_17_put_volume_skew"] = ratio(otm_put.groupby("date")["volume"].sum(), atm_put.groupby("date")["volume"].sum())
    d["feat_18_volume_weighted_avg_spread"] = (
        f.assign(x=f["volume"] * f["spread_pct"]).groupby("date")["x"].sum() / total_vol.replace(0, np.nan)
    )
    d["feat_19_spread_change_1d"] = d["feat_18_volume_weighted_avg_spread"].diff(1)
    d["feat_20_volume_weighted_moneyness"] = (
        f.assign(x=f["volume"] * f["s_over_k"]).groupby("date")["x"].sum() / total_vol.replace(0, np.nan)
    )
    d["feat_21_estimated_dealer_gamma_gex"] = (
        f.assign(x=f["gamma"] * f["open_interest"] * 100.0 * f["under_close"]).groupby("date")["x"].sum()
    )
    shares = np.nan
    try:
        shares = float(yf.Ticker(ticker).fast_info.get("shares"))
    except Exception:
        pass
    mc = und["close"] * shares if np.isfinite(shares) else np.nan
    d["feat_22_gex_pct_market_cap"] = d["feat_21_estimated_dealer_gamma_gex"] / mc

    c = f.copy()
    tau = (c["dte"].clip(lower=1) / 365.25).to_numpy(dtype=float)
    sigma = c["implied_volatility"].clip(lower=1e-6).to_numpy(dtype=float)
    s = c["under_close"].clip(lower=1e-6).to_numpy(dtype=float)
    k = c["strike"].clip(lower=1e-6).to_numpy(dtype=float)
    r = (market_series["irx"] / 100.0).reindex(c["date"]).ffill().bfill().to_numpy(dtype=float)
    q_series = (und["dividend_amount"].fillna(0.0).rolling(252, min_periods=1).sum() / und["close"].replace(0, np.nan)).fillna(0.0)
    q = q_series.reindex(c["date"]).ffill().fillna(0.0).to_numpy(dtype=float)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    phi = norm_pdf(d1)
    charm = -(phi * (2 * (r - q) * tau - d2 * sigma * np.sqrt(tau))) / (2 * tau * sigma * np.sqrt(tau))
    c["type_sign"] = np.where(c["type"] == "call", 1.0, -1.0)
    c["charm"] = charm
    d["feat_23_charm_exposure"] = (
        c.assign(x=c["type_sign"] * c["charm"] * c["open_interest"] * 100.0 * c["under_close"]).groupby("date")["x"].sum()
    )
    full_daily = opt.groupby("date")["volume"].sum()
    z = opt[(opt["dte"] >= 0) & (opt["dte"] <= 1)].groupby("date")["volume"].sum()
    d["feat_24_0dte_volume_share"] = ratio(z.reindex(d.index).fillna(0.0), full_daily.reindex(d.index))

    rf_series = (market_series["irx"] / 100.0).reindex(d.index).ffill().bfill()
    q_daily = q_series.reindex(d.index).ffill().fillna(0.0)
    bsrc = opt[(opt["dte"] >= 7) & (opt["bid"] > 0) & (opt["ask"] > 0)].copy()
    sk = {}
    ku = {}
    for dt, g in bsrc.groupby("date"):
        s3, k4 = bkm_style_moments(g, rf=float(rf_series.get(dt, 0.0)), q=float(q_daily.get(dt, 0.0)))
        sk[dt] = s3
        ku[dt] = k4
    d["feat_25_options_implied_skewness"] = pd.Series(sk).reindex(d.index)
    d["feat_26_options_implied_kurtosis"] = pd.Series(ku).reindex(d.index)

    close = und["close"]
    d["feat_27_stock_1d_return"] = close.pct_change(1)
    d["feat_28_stock_5d_return"] = close.pct_change(5)
    d["feat_29_stock_10d_return"] = close.pct_change(10)
    d["feat_30_hv_10d"] = logret.rolling(10, min_periods=10).std() * math.sqrt(252)
    d["feat_31_hv_30d"] = hv30
    d["feat_32_rsi_14d"] = rsi(close, 14)
    macd_line, macd_sig = macd(close)
    d["feat_33_macd"] = macd_line
    d["feat_34_macd_signal"] = macd_sig
    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    d["feat_35_bollinger_position"] = (close - (ma20 - 2 * sd20)) / ((ma20 + 2 * sd20) - (ma20 - 2 * sd20)).replace(0, np.nan)
    sma50 = close.rolling(50, min_periods=30).mean()
    d["feat_36_distance_from_50sma_pct"] = (close / sma50 - 1.0) * 100.0
    d["feat_37_atr_14d"] = atr14(und["high"], und["low"], close)

    d["feat_38_vix_close"] = market_series["vix"].reindex(d.index)
    d["feat_39_vix_5d_change"] = d["feat_38_vix_close"].diff(5)
    d["feat_40_vix_term_structure_slope"] = d["feat_38_vix_close"] - market_series["vix3m"].reindex(d.index)
    d["feat_41_risk_free_rate_3m"] = market_series["irx"].reindex(d.index) / 100.0
    etf = sector_etf_for_ticker(ticker)
    d["feat_42_sector_etf_daily_return"] = market_series[etf].reindex(d.index).pct_change(1)

    dates = pd.Series(d.index, index=d.index)
    d["feat_43_days_to_next_earnings"] = days_to_next_event(dates, earnings_idx) if len(earnings_idx) else np.nan
    d["feat_44_earnings_within_7d_flag"] = ((d["feat_43_days_to_next_earnings"] >= 0) & (d["feat_43_days_to_next_earnings"] <= 7)).astype(int)
    div_idx = pd.DatetimeIndex(d.index[und["dividend_amount"].reindex(d.index).fillna(0.0) > 0])
    d["feat_45_days_to_ex_dividend"] = days_to_next_event(dates, div_idx) if len(div_idx) else np.nan
    d["feat_46_day_of_week"] = d.index.weekday
    opex = pd.DatetimeIndex([third_friday(x.year, x.month) for x in pd.date_range(d.index.min(), d.index.max(), freq="MS", tz="UTC")])
    d["feat_47_days_to_monthly_opex"] = days_to_next_event(dates, opex)
    d["feat_48_quarter_end_flag"] = d.index.is_quarter_end.astype(int)
    d["feat_49_news_sentiment_score"] = np.nan

    d["target_r_t_plus_1"] = close.shift(-1) / close - 1.0
    d["target_direction_t_plus_1"] = (d["target_r_t_plus_1"] > 0).astype("Int64")
    return d.reset_index(names="date")


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    base_dir = root / "data/raw/options_data_repo"
    out_dir = root / "data/processed/panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA",
        "AVGO",
        "AMD",
        "NFLX",
        "PLTR",
        "SPY",
        "QQQ",
        "JPM",
    ]

    # Pull market controls once.
    market_series = {
        "vix": get_close("^VIX", start="2008-01-01"),
        "vix3m": get_close("^VIX3M", start="2008-01-01"),
        "irx": get_close("^IRX", start="2008-01-01"),
        "XLK": get_close("XLK", start="2008-01-01"),
        "XLC": get_close("XLC", start="2008-01-01"),
        "XLY": get_close("XLY", start="2008-01-01"),
        "XLF": get_close("XLF", start="2008-01-01"),
        "SPY": get_close("SPY", start="2008-01-01"),
        "QQQ": get_close("QQQ", start="2008-01-01"),
    }

    parts: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    for t in tickers:
        print(f"Building features for {t} ...")
        tk = yf.Ticker(t)
        try:
            earnings = tk.get_earnings_dates(limit=100)
            if earnings is not None and len(earnings) > 0:
                earnings = earnings.reset_index().rename(columns={"Earnings Date": "earnings_date"})
                eidx = pd.to_datetime(earnings["earnings_date"], utc=True, errors="coerce").dropna().dt.floor("D")
                eidx = pd.DatetimeIndex(sorted(eidx.unique()))
            else:
                eidx = pd.DatetimeIndex([])
        except Exception:
            eidx = pd.DatetimeIndex([])
        df = build_ticker_features(t, base_dir, market_series, eidx)
        parts.append(df)
        df.to_parquet(out_dir / f"{t.lower()}_features.parquet", index=False)
        coverage_rows.append(
            {
                "ticker": t,
                "rows": len(df),
                "date_min": str(df["date"].min()) if len(df) else None,
                "date_max": str(df["date"].max()) if len(df) else None,
                "non_null_target_rows": int(df["target_r_t_plus_1"].notna().sum()) if len(df) else 0,
            }
        )

    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    panel.to_parquet(out_dir / "panel_features_all_tickers.parquet", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "panel_ticker_coverage.csv", index=False)
    print(f"Saved panel rows={len(panel)} to {out_dir / 'panel_features_all_tickers.parquet'}")


if __name__ == "__main__":
    main()
