#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


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
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal


def atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(days=14)


def days_to_next_event(dates: pd.Series, event_dates: pd.DatetimeIndex) -> pd.Series:
    event_dates = event_dates.sort_values()
    out = []
    for d in dates:
        idx = event_dates.searchsorted(d, side="left")
        if idx >= len(event_dates):
            out.append(np.nan)
        else:
            out.append((event_dates[idx] - d).days)
    return pd.Series(out, index=dates.index, dtype=float)


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    db_dir = root / "data" / "raw" / "databento"
    free_dir = root / "data" / "raw" / "free_sources"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    stock = pd.read_parquet(db_dir / "aapl_stock_ohlcv_1d.parquet")[
        ["open", "high", "low", "close", "volume", "symbol"]
    ].copy()
    stock["date"] = stock.index.tz_convert("UTC").floor("D")
    stock = stock.set_index("date").sort_index()
    stock = stock[~stock.index.duplicated(keep="last")]

    opt = pd.read_parquet(db_dir / "aapl_options_ohlcv_1d.parquet")[
        ["instrument_id", "close", "volume", "symbol"]
    ].copy()
    opt["date"] = opt.index.tz_convert("UTC").floor("D")

    # Consolidate by contract-day across venues/publishers.
    opt = (
        opt.groupby(["date", "instrument_id", "symbol"], as_index=False)
        .agg(volume=("volume", "sum"), close=("close", "last"))
        .sort_values(["date", "instrument_id"])
    )

    defs = pd.read_parquet(db_dir / "aapl_options_definition.parquet")[
        ["instrument_id", "instrument_class", "expiration", "strike_price"]
    ].copy()
    defs = defs.sort_index().groupby("instrument_id", as_index=False).last()
    defs["expiration"] = pd.to_datetime(defs["expiration"], utc=True).dt.floor("D")

    opt = opt.merge(
        defs[["instrument_id", "instrument_class", "expiration", "strike_price"]],
        on="instrument_id",
        how="left",
    )
    opt = opt[opt["instrument_class"].isin(["C", "P"])].copy()
    opt["date"] = pd.to_datetime(opt["date"], utc=True)

    # Join underlying close for moneyness and DTE.
    under_px = stock[["close"]].rename(columns={"close": "under_close"}).reset_index()
    opt = opt.merge(under_px, on="date", how="left")
    opt["dte"] = (opt["expiration"] - opt["date"]).dt.days
    opt["s_over_k"] = opt["under_close"] / opt["strike_price"].replace(0.0, np.nan)
    opt["k_over_s"] = opt["strike_price"] / opt["under_close"].replace(0.0, np.nan)

    # Apply report-like filter subset we can support from current fields.
    opt_f = opt[(opt["dte"] >= 7) & (opt["s_over_k"] >= 0.80) & (opt["s_over_k"] <= 1.20)].copy()

    def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.replace(0.0, np.nan)

    g = opt_f.groupby("date")
    daily = pd.DataFrame(index=stock.index.copy())
    daily["total_options_volume_contracts"] = g["volume"].sum()
    daily["call_volume_contracts"] = opt_f[opt_f["instrument_class"] == "C"].groupby("date")["volume"].sum()
    daily["put_volume_contracts"] = opt_f[opt_f["instrument_class"] == "P"].groupby("date")["volume"].sum()
    daily["feat_01_put_call_volume_ratio"] = ratio(daily["put_volume_contracts"], daily["call_volume_contracts"])

    daily["feat_03_options_to_stock_volume_ratio"] = (
        (daily["total_options_volume_contracts"] * 100.0) / stock["volume"].replace(0.0, np.nan)
    )
    daily["feat_13_volume_spike_indicator"] = (
        daily["total_options_volume_contracts"]
        / daily["total_options_volume_contracts"].rolling(20, min_periods=5).mean()
    )

    otm_call = opt_f[(opt_f["instrument_class"] == "C") & (opt_f["s_over_k"] < 0.95)]
    otm_put = opt_f[(opt_f["instrument_class"] == "P") & (opt_f["k_over_s"] < 0.95)]
    atm_call = opt_f[(opt_f["instrument_class"] == "C") & (opt_f["s_over_k"].sub(1.0).abs() <= 0.02)]
    atm_put = opt_f[(opt_f["instrument_class"] == "P") & (opt_f["s_over_k"].sub(1.0).abs() <= 0.02)]

    daily["feat_15_otm_volume_concentration"] = ratio(
        otm_call.groupby("date")["volume"].sum().add(otm_put.groupby("date")["volume"].sum(), fill_value=0.0),
        daily["total_options_volume_contracts"],
    )
    daily["feat_16_call_volume_skew"] = ratio(
        otm_call.groupby("date")["volume"].sum(),
        atm_call.groupby("date")["volume"].sum(),
    )
    daily["feat_17_put_volume_skew"] = ratio(
        otm_put.groupby("date")["volume"].sum(),
        atm_put.groupby("date")["volume"].sum(),
    )
    daily["feat_20_volume_weighted_moneyness"] = (
        opt_f.assign(vw=opt_f["volume"] * opt_f["s_over_k"]).groupby("date")["vw"].sum()
        / daily["total_options_volume_contracts"].replace(0.0, np.nan)
    )
    daily["feat_24_0dte_volume_share"] = ratio(
        opt[(opt["dte"] <= 1) & (opt["dte"] >= 0)].groupby("date")["volume"].sum(),
        opt.groupby("date")["volume"].sum(),
    )

    # Stats-based features (available only where statistics was pulled).
    stats_path = db_dir / "aapl_options_statistics.parquet"
    if stats_path.exists():
        stats = pd.read_parquet(stats_path)[
            ["ts_event", "instrument_id", "stat_type", "price", "quantity"]
        ].copy()
        stats["date"] = pd.to_datetime(stats["ts_event"], utc=True).dt.floor("D")

        oi = stats[stats["stat_type"] == 9].copy()
        oi = oi.groupby(["date", "instrument_id"], as_index=False)["quantity"].max()
        oi_daily = oi.groupby("date", as_index=True)["quantity"].sum()
        daily["feat_02_put_call_open_interest_ratio"] = np.nan
        oi_with_type = oi.merge(
            defs[["instrument_id", "instrument_class"]],
            on="instrument_id",
            how="left",
        )
        oi_put = oi_with_type[oi_with_type["instrument_class"] == "P"].groupby("date")["quantity"].sum()
        oi_call = oi_with_type[oi_with_type["instrument_class"] == "C"].groupby("date")["quantity"].sum()
        daily.loc[oi_daily.index, "feat_02_put_call_open_interest_ratio"] = ratio(oi_put, oi_call)
        daily["feat_14_open_interest_change_1d"] = oi_daily.diff()
    else:
        daily["feat_02_put_call_open_interest_ratio"] = np.nan
        daily["feat_14_open_interest_change_1d"] = np.nan

    # Stock/control features.
    close = stock["close"]
    logret = np.log(close / close.shift(1))
    daily["feat_27_stock_1d_return"] = close.pct_change(1)
    daily["feat_28_stock_5d_return"] = close.pct_change(5)
    daily["feat_29_stock_10d_return"] = close.pct_change(10)
    daily["feat_30_hv_10d"] = logret.rolling(10, min_periods=10).std() * math.sqrt(252)
    daily["feat_31_hv_30d"] = logret.rolling(30, min_periods=20).std() * math.sqrt(252)
    daily["feat_32_rsi_14d"] = rsi(close, 14)
    macd_line, signal_line = macd(close)
    daily["feat_33_macd"] = macd_line
    daily["feat_34_macd_signal"] = signal_line
    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    upper = ma20 + 2.0 * sd20
    lower = ma20 - 2.0 * sd20
    daily["feat_35_bollinger_position"] = (close - lower) / (upper - lower).replace(0.0, np.nan)
    sma50 = close.rolling(50, min_periods=30).mean()
    daily["feat_36_distance_from_50sma_pct"] = (close / sma50 - 1.0) * 100.0
    daily["feat_37_atr_14d"] = atr14(stock["high"], stock["low"], close)

    # Merge free-source market/event controls.
    def load_series(path: Path, name: str) -> pd.Series:
        if not path.exists():
            return pd.Series(dtype=float, name=name)
        df = pd.read_parquet(path)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
        return df.set_index("date")["close"].rename(name)

    vix = load_series(free_dir / "vix_ohlcv_1d_yahoo.parquet", "vix_close")
    vix3m = load_series(free_dir / "vix3m_ohlcv_1d_yahoo.parquet", "vix3m_close")
    irx = load_series(free_dir / "irx_ohlcv_1d_yahoo.parquet", "irx_close")
    xlk = load_series(free_dir / "xlk_ohlcv_1d_yahoo.parquet", "xlk_close")

    daily["feat_38_vix_close"] = vix.reindex(daily.index)
    daily["feat_39_vix_5d_change"] = daily["feat_38_vix_close"].diff(5)
    daily["feat_40_vix_term_structure_slope"] = (
        daily["feat_38_vix_close"] - vix3m.reindex(daily.index)
    )
    daily["feat_41_risk_free_rate_3m"] = irx.reindex(daily.index) / 100.0
    daily["feat_42_sector_etf_daily_return"] = xlk.reindex(daily.index).pct_change(1)

    earnings_path = free_dir / "aapl_earnings_dates_yahoo.parquet"
    if earnings_path.exists():
        earnings = pd.read_parquet(earnings_path)
        e_idx = pd.to_datetime(earnings["earnings_date"], utc=True, errors="coerce").dropna().dt.floor("D")
        e_idx = pd.DatetimeIndex(sorted(e_idx.unique()))
    else:
        e_idx = pd.DatetimeIndex([])

    div_path = free_dir / "aapl_dividends_yahoo.parquet"
    if div_path.exists():
        divs = pd.read_parquet(div_path)
        d_idx = pd.to_datetime(divs["date"], utc=True, errors="coerce").dropna().dt.floor("D")
        d_idx = pd.DatetimeIndex(sorted(d_idx.unique()))
    else:
        d_idx = pd.DatetimeIndex([])

    dates = pd.Series(daily.index, index=daily.index)
    daily["feat_43_days_to_next_earnings"] = (
        days_to_next_event(dates, e_idx) if len(e_idx) else np.nan
    )
    daily["feat_44_earnings_within_7d_flag"] = (
        (daily["feat_43_days_to_next_earnings"] >= 0)
        & (daily["feat_43_days_to_next_earnings"] <= 7)
    ).astype(int)
    daily["feat_45_days_to_ex_dividend"] = (
        days_to_next_event(dates, d_idx) if len(d_idx) else np.nan
    )
    daily["feat_46_day_of_week"] = pd.Index(daily.index).weekday

    opex_dates = [
        third_friday(d.year, d.month)
        for d in pd.date_range(daily.index.min(), daily.index.max(), freq="MS", tz="UTC")
    ]
    opex_idx = pd.DatetimeIndex(sorted(opex_dates))
    daily["feat_47_days_to_monthly_opex"] = days_to_next_event(dates, opex_idx)
    daily["feat_48_quarter_end_flag"] = pd.Index(daily.index).is_quarter_end.astype(int)

    # Keep report numbering for missing features as explicit null columns.
    missing_cols = {
        "feat_04_net_delta_exposure": np.nan,
        "feat_05_volume_weighted_avg_iv": np.nan,
        "feat_06_atm_iv": np.nan,
        "feat_07_iv_rank_52w_percentile": np.nan,
        "feat_08_iv_skew_25delta": np.nan,
        "feat_09_iv_term_structure_slope": np.nan,
        "feat_10_atm_iv_change_1d": np.nan,
        "feat_11_atm_iv_change_5d": np.nan,
        "feat_12_iv_minus_realized_vol_30d": np.nan,
        "feat_18_volume_weighted_avg_spread": np.nan,
        "feat_19_spread_change_1d": np.nan,
        "feat_21_estimated_dealer_gamma_gex": np.nan,
        "feat_22_gex_pct_market_cap": np.nan,
        "feat_23_charm_exposure": np.nan,
        "feat_25_options_implied_skewness": np.nan,
        "feat_26_options_implied_kurtosis": np.nan,
        "feat_49_news_sentiment_score": np.nan,
    }
    for col, val in missing_cols.items():
        if col not in daily.columns:
            daily[col] = val

    # Targets from report.
    daily["target_r_t_plus_1"] = close.shift(-1) / close - 1.0
    daily["target_direction_t_plus_1"] = (daily["target_r_t_plus_1"] > 0).astype("Int64")

    # Reorder roughly by feature number.
    ordered = [f"feat_{i:02d}_" for i in range(1, 50)]
    feat_cols: list[str] = []
    for prefix in ordered:
        feat_cols.extend([c for c in daily.columns if c.startswith(prefix)])
    other_cols = [c for c in daily.columns if c not in feat_cols]
    daily = daily[feat_cols + other_cols]

    out_features = out_dir / "aapl_features_daily_partial.parquet"
    daily.reset_index(names="date").to_parquet(out_features, index=False)

    coverage = pd.DataFrame(
        {
            "feature": [c for c in daily.columns if c.startswith("feat_")],
            "non_null_count": [int(daily[c].notna().sum()) for c in daily.columns if c.startswith("feat_")],
            "coverage_ratio": [float(daily[c].notna().mean()) for c in daily.columns if c.startswith("feat_")],
        }
    ).sort_values("feature")
    out_cov = out_dir / "aapl_features_coverage_partial.csv"
    coverage.to_csv(out_cov, index=False)

    print(f"Saved features: {out_features} rows={len(daily)} cols={len(daily.columns)}")
    print(f"Saved coverage: {out_cov}")
    print("Date range:", daily.index.min(), "->", daily.index.max())
    top_ready = coverage.sort_values("coverage_ratio", ascending=False).head(15)
    print("\\nTop covered features:")
    print(top_ready.to_string(index=False))


if __name__ == "__main__":
    main()
