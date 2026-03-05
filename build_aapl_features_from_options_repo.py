#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AAPL features from free options-data repo.")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-16", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--base-dir",
        default="data/raw/options_data_repo/aapl",
        help="Directory containing options.parquet and underlying.parquet",
    )
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    return parser.parse_args()


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
        idx = event_dates.searchsorted(d, side="left")
        if idx >= len(event_dates):
            vals.append(np.nan)
        else:
            vals.append(float((event_dates[idx] - d).days))
    return pd.Series(vals, index=dates.index)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def get_shares_series(
    ticker: yf.Ticker,
    start: pd.Timestamp,
    end: pd.Timestamp,
    idx: pd.DatetimeIndex,
) -> pd.Series:
    # Prefer historical shares outstanding if available, fallback to constant proxy.
    try:
        s = ticker.get_shares_full(
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        )
        if isinstance(s, pd.Series) and len(s) > 0:
            s.index = pd.to_datetime(s.index, utc=True).floor("D")
            s = s[~s.index.duplicated(keep="last")].sort_index()
            return s.reindex(idx).ffill().bfill()
    except Exception:
        pass
    shares = np.nan
    try:
        shares = float(ticker.fast_info.get("shares"))
    except Exception:
        shares = np.nan
    if not np.isfinite(shares):
        return pd.Series(np.nan, index=idx)
    return pd.Series(shares, index=idx)


def bkm_style_moments(
    day_df: pd.DataFrame,
    rf: float,
    q: float,
    target_dte: int = 30,
) -> tuple[float, float]:
    # Practical model-free approximation using OTM option prices over a chosen expiry.
    # Returns skewness and kurtosis of log-moneyness distribution weighted by Q(K)/K^2.
    if day_df.empty:
        return np.nan, np.nan
    day_df = day_df[(day_df["bid"] > 0) & (day_df["ask"] > 0) & (day_df["strike"] > 0) & (day_df["dte"] >= 7)]
    if day_df.empty:
        return np.nan, np.nan

    expiries = (
        day_df.groupby("expiration", as_index=False)["dte"]
        .median()
        .assign(dist=lambda x: (x["dte"] - target_dte).abs())
        .sort_values("dist")
    )
    if expiries.empty:
        return np.nan, np.nan
    chosen_exp = expiries.iloc[0]["expiration"]
    x = day_df[day_df["expiration"] == chosen_exp].copy()
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
        if k <= 0:
            continue
        qk = p_map[k] if k < fwd else c_map[k]
        if not np.isfinite(qk) or qk <= 0:
            continue
        rows.append((float(k), float(qk)))
    if len(rows) < 6:
        return np.nan, np.nan

    arr = np.array(rows, dtype=float)
    k = arr[:, 0]
    qk = arr[:, 1]
    order = np.argsort(k)
    k = k[order]
    qk = qk[order]

    # Model-free variance (VIX-style) and strike-weighted density proxy.
    integrand = qk / (k**2)
    model_free_var = (2.0 * math.exp(rf * t) / t) * np.trapezoid(integrand, k)
    if not np.isfinite(model_free_var) or model_free_var <= 0:
        return np.nan, np.nan

    w = integrand / np.trapezoid(integrand, k)
    if np.any(~np.isfinite(w)):
        return np.nan, np.nan
    xlog = np.log(k / fwd)
    mu = np.trapezoid(w * xlog, k)
    cen2 = np.trapezoid(w * (xlog - mu) ** 2, k)
    cen3 = np.trapezoid(w * (xlog - mu) ** 3, k)
    cen4 = np.trapezoid(w * (xlog - mu) ** 4, k)
    if cen2 <= 0:
        return np.nan, np.nan
    skew = cen3 / (cen2 ** 1.5)
    kurt = cen4 / (cen2**2)
    return float(skew), float(kurt)


def build() -> None:
    args = parse_args()
    root = Path("/Users/pa/Desktop/guided")
    base = root / args.base_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    opt = pd.read_parquet(base / "options.parquet")
    und = pd.read_parquet(base / "underlying.parquet")

    opt["date"] = pd.to_datetime(opt["date"], utc=True)
    opt["expiration"] = pd.to_datetime(opt["expiration"], utc=True)
    und["date"] = pd.to_datetime(und["date"], utc=True)

    opt = opt[(opt["date"] >= start) & (opt["date"] <= end)].copy()
    und = und[(und["date"] >= start) & (und["date"] <= end)].copy()
    und = und.sort_values("date").drop_duplicates("date").set_index("date")

    # Market controls from Yahoo
    def dl(symbol: str) -> pd.Series:
        s = yf.download(symbol, start=args.start, end=(end + pd.Timedelta(days=3)).strftime("%Y-%m-%d"), interval="1d", progress=False)
        if isinstance(s.columns, pd.MultiIndex):
            s.columns = [c[0] for c in s.columns]
        s = s.reset_index().rename(columns={"Date": "date", "Close": "close"})
        s["date"] = pd.to_datetime(s["date"], utc=True)
        return s.set_index("date")["close"]

    vix = dl("^VIX")
    vix3m = dl("^VIX3M")
    irx = dl("^IRX")
    xlk = dl("XLK")

    # Earnings/dividends
    t = yf.Ticker("AAPL")
    earnings = t.get_earnings_dates(limit=64)
    if earnings is not None and len(earnings) > 0:
        earnings = earnings.reset_index().rename(columns={"Earnings Date": "earnings_date"})
        eidx = pd.to_datetime(earnings["earnings_date"], utc=True, errors="coerce").dropna().dt.floor("D")
        eidx = pd.DatetimeIndex(sorted(eidx.unique()))
    else:
        eidx = pd.DatetimeIndex([])
    divs = t.dividends
    didx = pd.DatetimeIndex(pd.to_datetime(divs.index, utc=True).floor("D")) if len(divs) else pd.DatetimeIndex([])

    # Join underlying close into option rows
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

    # Core daily frame
    daily = pd.DataFrame(index=und.index.copy())
    daily["symbol"] = "AAPL"

    # 1,2,3
    call = f[f["type"] == "call"]
    put = f[f["type"] == "put"]
    daily["feat_01_put_call_volume_ratio"] = ratio(
        put.groupby("date")["volume"].sum(),
        call.groupby("date")["volume"].sum(),
    )
    daily["feat_02_put_call_open_interest_ratio"] = ratio(
        put.groupby("date")["open_interest"].sum(),
        call.groupby("date")["open_interest"].sum(),
    )
    daily["feat_03_options_to_stock_volume_ratio"] = (
        f.groupby("date")["volume"].sum() * 100.0 / und["volume"].replace(0, np.nan)
    )

    # 4,5,6
    daily["feat_04_net_delta_exposure"] = (
        call.assign(x=call["delta"] * call["open_interest"]).groupby("date")["x"].sum()
        - put.assign(x=put["delta"].abs() * put["open_interest"]).groupby("date")["x"].sum()
    )
    daily["feat_05_volume_weighted_avg_iv"] = (
        f.assign(x=f["volume"] * f["implied_volatility"]).groupby("date")["x"].sum()
        / f.groupby("date")["volume"].sum().replace(0, np.nan)
    )
    atm = f[(f["dte"].between(20, 40))].copy()
    if len(atm):
        atm["atm_dist"] = (atm["s_over_k"] - 1.0).abs()
        atm = atm.sort_values(["date", "atm_dist"]).groupby("date").head(5)
        daily["feat_06_atm_iv"] = (
            atm.assign(x=atm["volume"] * atm["implied_volatility"]).groupby("date")["x"].sum()
            / atm.groupby("date")["volume"].sum().replace(0, np.nan)
        )
    else:
        daily["feat_06_atm_iv"] = np.nan

    # 7,8,9,10,11,12
    roll_min = daily["feat_06_atm_iv"].rolling(252, min_periods=20).min()
    roll_max = daily["feat_06_atm_iv"].rolling(252, min_periods=20).max()
    daily["feat_07_iv_rank_52w_percentile"] = ratio(daily["feat_06_atm_iv"] - roll_min, roll_max - roll_min)

    c25 = call.assign(d=(call["delta"] - 0.25).abs()).sort_values(["date", "d"]).groupby("date").head(1)
    p25 = put.assign(d=(put["delta"] + 0.25).abs()).sort_values(["date", "d"]).groupby("date").head(1)
    daily["feat_08_iv_skew_25delta"] = (
        p25.set_index("date")["implied_volatility"] - c25.set_index("date")["implied_volatility"]
    )

    # Base 30D-90D ATM IV slope, with interpolation fallback by daily ATM term curve.
    atm30 = f[f["dte"].between(20, 40)].copy()
    atm90 = f[f["dte"].between(75, 105)].copy()
    atm30["d"] = (atm30["s_over_k"] - 1.0).abs()
    atm90["d"] = (atm90["s_over_k"] - 1.0).abs()
    atm30 = atm30.sort_values(["date", "d"]).groupby("date").head(1).set_index("date")
    atm90 = atm90.sort_values(["date", "d"]).groupby("date").head(1).set_index("date")
    base_slope = atm30["implied_volatility"] - atm90["implied_volatility"]

    term_src = f.copy()
    term_src["atm_dist"] = (term_src["s_over_k"] - 1.0).abs()
    term_src = term_src.sort_values(["date", "expiration", "atm_dist"]).groupby(["date", "expiration"]).head(1)
    term_src["term_dte"] = term_src["dte"].astype(float)
    interp_slope: dict[pd.Timestamp, float] = {}
    for d, gdf in term_src.groupby("date"):
        x = gdf["term_dte"].to_numpy(dtype=float)
        y = gdf["implied_volatility"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if ok.sum() < 3:
            interp_slope[d] = np.nan
            continue
        x = x[ok]
        y = y[ok]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # Clamp to the support if 30/90 are slightly out of observed range.
        iv30 = np.interp(30.0, x, y, left=y[0], right=y[-1])
        iv90 = np.interp(90.0, x, y, left=y[0], right=y[-1])
        interp_slope[d] = float(iv30 - iv90)
    daily["feat_09_iv_term_structure_slope"] = base_slope.combine_first(pd.Series(interp_slope))

    daily["feat_10_atm_iv_change_1d"] = daily["feat_06_atm_iv"].diff(1)
    daily["feat_11_atm_iv_change_5d"] = daily["feat_06_atm_iv"].diff(5)

    logret = np.log(und["close"] / und["close"].shift(1))
    hv30 = logret.rolling(30, min_periods=20).std() * math.sqrt(252)
    daily["feat_12_iv_minus_realized_vol_30d"] = daily["feat_06_atm_iv"] - hv30

    # 13,14,15,16,17
    total_vol = f.groupby("date")["volume"].sum()
    total_oi = f.groupby("date")["open_interest"].sum()
    daily["feat_13_volume_spike_indicator"] = total_vol / total_vol.rolling(20, min_periods=5).mean()
    daily["feat_14_open_interest_change_1d"] = total_oi.diff(1)

    otm_call = f[(f["type"] == "call") & (f["s_over_k"] < 0.95)]
    otm_put = f[(f["type"] == "put") & (f["k_over_s"] < 0.95)]
    atm_call = f[(f["type"] == "call") & ((f["s_over_k"] - 1.0).abs() <= 0.02)]
    atm_put = f[(f["type"] == "put") & ((f["s_over_k"] - 1.0).abs() <= 0.02)]
    daily["feat_15_otm_volume_concentration"] = ratio(
        otm_call.groupby("date")["volume"].sum().add(otm_put.groupby("date")["volume"].sum(), fill_value=0.0),
        total_vol,
    )
    daily["feat_16_call_volume_skew"] = ratio(
        otm_call.groupby("date")["volume"].sum(),
        atm_call.groupby("date")["volume"].sum(),
    )
    daily["feat_17_put_volume_skew"] = ratio(
        otm_put.groupby("date")["volume"].sum(),
        atm_put.groupby("date")["volume"].sum(),
    )

    # 18,19,20
    daily["feat_18_volume_weighted_avg_spread"] = (
        f.assign(x=f["volume"] * f["spread_pct"]).groupby("date")["x"].sum()
        / total_vol.replace(0, np.nan)
    )
    daily["feat_19_spread_change_1d"] = daily["feat_18_volume_weighted_avg_spread"].diff(1)
    daily["feat_20_volume_weighted_moneyness"] = (
        f.assign(x=f["volume"] * f["s_over_k"]).groupby("date")["x"].sum()
        / total_vol.replace(0, np.nan)
    )

    # 21,22,23,24
    daily["feat_21_estimated_dealer_gamma_gex"] = (
        f.assign(x=f["gamma"] * f["open_interest"] * 100.0 * f["under_close"]).groupby("date")["x"].sum()
    )
    shares_series = get_shares_series(t, start, end, und.index)
    market_cap = und["close"] * shares_series
    daily["feat_22_gex_pct_market_cap"] = daily["feat_21_estimated_dealer_gamma_gex"] / market_cap

    # Charm approximation from BS with observed IV.
    c = f.copy()
    tau = (c["dte"].clip(lower=1) / 365.25).to_numpy(dtype=float)
    sigma = c["implied_volatility"].clip(lower=1e-6).to_numpy(dtype=float)
    s = c["under_close"].clip(lower=1e-6).to_numpy(dtype=float)
    k = c["strike"].clip(lower=1e-6).to_numpy(dtype=float)
    r_daily = (irx / 100.0).reindex(c["date"]).ffill().bfill()
    r = r_daily.to_numpy(dtype=float)
    # Dividend yield proxy from trailing 252-day cash dividends / spot.
    div_roll = und["dividend_amount"].fillna(0.0).rolling(252, min_periods=1).sum()
    q_series = (div_roll / und["close"].replace(0, np.nan)).fillna(0.0)
    q = q_series.reindex(c["date"]).ffill().fillna(0.0).to_numpy(dtype=float)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    phi = norm_pdf(d1)
    charm = -(phi * (2 * (r - q) * tau - d2 * sigma * np.sqrt(tau))) / (2 * tau * sigma * np.sqrt(tau))
    c["charm"] = charm
    c["type_sign"] = np.where(c["type"] == "call", 1.0, -1.0)
    daily["feat_23_charm_exposure"] = (
        c.assign(x=c["type_sign"] * c["charm"] * c["open_interest"] * 100.0 * c["under_close"])
        .groupby("date")["x"]
        .sum()
    )

    full_daily = opt.groupby("date")["volume"].sum()
    z = opt[(opt["dte"] >= 0) & (opt["dte"] <= 1)].groupby("date")["volume"].sum()
    daily["feat_24_0dte_volume_share"] = ratio(z.reindex(daily.index).fillna(0.0), full_daily.reindex(daily.index))

    # 25,26: model-free (BKM-style) daily implied moments using OTM option prices.
    rf_series = (irx / 100.0).reindex(daily.index).ffill().bfill()
    q_daily = (und["dividend_amount"].fillna(0.0).rolling(252, min_periods=1).sum() / und["close"].replace(0, np.nan)).fillna(0.0)
    bkm_src = opt[(opt["dte"] >= 7) & (opt["bid"] > 0) & (opt["ask"] > 0)].copy()
    skew_map: dict[pd.Timestamp, float] = {}
    kurt_map: dict[pd.Timestamp, float] = {}
    for d, gdf in bkm_src.groupby("date"):
        rf = float(rf_series.get(d, np.nan))
        qd = float(q_daily.get(d, 0.0))
        s3, k4 = bkm_style_moments(gdf, rf=rf if np.isfinite(rf) else 0.0, q=qd)
        skew_map[d] = s3
        kurt_map[d] = k4
    daily["feat_25_options_implied_skewness"] = pd.Series(skew_map).reindex(daily.index)
    daily["feat_26_options_implied_kurtosis"] = pd.Series(kurt_map).reindex(daily.index)

    # 27..37 controls
    close = und["close"]
    daily["feat_27_stock_1d_return"] = close.pct_change(1)
    daily["feat_28_stock_5d_return"] = close.pct_change(5)
    daily["feat_29_stock_10d_return"] = close.pct_change(10)
    daily["feat_30_hv_10d"] = logret.rolling(10, min_periods=10).std() * math.sqrt(252)
    daily["feat_31_hv_30d"] = hv30
    daily["feat_32_rsi_14d"] = rsi(close, 14)
    macd_line, macd_signal = macd(close)
    daily["feat_33_macd"] = macd_line
    daily["feat_34_macd_signal"] = macd_signal
    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    upper = ma20 + 2.0 * sd20
    lower = ma20 - 2.0 * sd20
    daily["feat_35_bollinger_position"] = (close - lower) / (upper - lower).replace(0, np.nan)
    sma50 = close.rolling(50, min_periods=30).mean()
    daily["feat_36_distance_from_50sma_pct"] = (close / sma50 - 1.0) * 100.0
    daily["feat_37_atr_14d"] = atr14(und["high"], und["low"], close)

    # 38..42 market controls
    daily["feat_38_vix_close"] = vix.reindex(daily.index)
    daily["feat_39_vix_5d_change"] = daily["feat_38_vix_close"].diff(5)
    daily["feat_40_vix_term_structure_slope"] = daily["feat_38_vix_close"] - vix3m.reindex(daily.index)
    daily["feat_41_risk_free_rate_3m"] = irx.reindex(daily.index) / 100.0
    daily["feat_42_sector_etf_daily_return"] = xlk.reindex(daily.index).pct_change(1)

    # 43..48 event and temporal
    dates = pd.Series(daily.index, index=daily.index)
    daily["feat_43_days_to_next_earnings"] = days_to_next_event(dates, eidx) if len(eidx) else np.nan
    daily["feat_44_earnings_within_7d_flag"] = (
        (daily["feat_43_days_to_next_earnings"] >= 0) & (daily["feat_43_days_to_next_earnings"] <= 7)
    ).astype(int)
    daily["feat_45_days_to_ex_dividend"] = days_to_next_event(dates, didx) if len(didx) else np.nan
    daily["feat_46_day_of_week"] = daily.index.weekday
    opex = pd.DatetimeIndex([third_friday(d.year, d.month) for d in pd.date_range(daily.index.min(), daily.index.max(), freq="MS", tz="UTC")])
    daily["feat_47_days_to_monthly_opex"] = days_to_next_event(dates, opex)
    daily["feat_48_quarter_end_flag"] = daily.index.is_quarter_end.astype(int)

    # 49 optional sentiment not yet included.
    daily["feat_49_news_sentiment_score"] = np.nan

    # Targets
    daily["target_r_t_plus_1"] = close.shift(-1) / close - 1.0
    daily["target_direction_t_plus_1"] = (daily["target_r_t_plus_1"] > 0).astype("Int64")

    # Output
    out_features = out_dir / "aapl_features_daily_from_options_repo.parquet"
    daily.reset_index(names="date").to_parquet(out_features, index=False)
    coverage = pd.DataFrame(
        {
            "feature": [c for c in daily.columns if c.startswith("feat_")],
            "non_null_count": [int(daily[c].notna().sum()) for c in daily.columns if c.startswith("feat_")],
            "coverage_ratio": [float(daily[c].notna().mean()) for c in daily.columns if c.startswith("feat_")],
        }
    )
    out_coverage = out_dir / "aapl_features_coverage_from_options_repo.csv"
    coverage.to_csv(out_coverage, index=False)
    out_meta = out_dir / "aapl_features_notes_from_options_repo.txt"
    out_meta.write_text(
        "Notes:\\n"
        "- Features 25/26 use model-free BKM-style approximation from OTM option prices and strike integration.\\n"
        "- Feature 22 uses historical shares outstanding when Yahoo provides it; otherwise constant fallback.\\n"
        "- Feature 23 uses Black-Scholes charm with call/put directional sign aggregation.\\n"
        "- Feature 49 is intentionally left NaN (sentiment pipeline not added yet).\\n"
    )
    print(f"Saved features: {out_features} rows={len(daily)} cols={len(daily.columns)}")
    print(f"Saved coverage: {out_coverage}")
    print(f"Saved notes: {out_meta}")


if __name__ == "__main__":
    build()
