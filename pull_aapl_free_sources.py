#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull AAPL data from free/public sources (Yahoo Finance)."
    )
    parser.add_argument("--start", default="2023-03-28", help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/free_sources",
        help="Output directory for parquet/csv artifacts.",
    )
    return parser.parse_args()


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x * x)


def compute_bs_greeks(
    spot: float,
    strike: pd.Series,
    ttm_years: pd.Series,
    rf_rate: float,
    iv: pd.Series,
    option_type: pd.Series,
) -> pd.DataFrame:
    # Vectorized Black-Scholes Greeks from IV, used to fill missing greek fields in Yahoo data.
    eps = 1e-12
    k = strike.to_numpy(dtype=float)
    t = ttm_years.to_numpy(dtype=float)
    sigma = iv.to_numpy(dtype=float)
    typ = option_type.to_numpy(dtype=str)

    valid = (k > 0) & (t > 0) & (sigma > 0) & np.isfinite(k) & np.isfinite(t) & np.isfinite(sigma)
    delta = np.full_like(k, np.nan, dtype=float)
    gamma = np.full_like(k, np.nan, dtype=float)
    theta = np.full_like(k, np.nan, dtype=float)
    vega = np.full_like(k, np.nan, dtype=float)

    if not np.any(valid):
        return pd.DataFrame({"delta": delta, "gamma": gamma, "theta": theta, "vega": vega})

    kv = np.clip(k[valid], eps, None)
    tv = np.clip(t[valid], eps, None)
    sv = np.clip(sigma[valid], eps, None)

    d1 = (np.log(spot / kv) + (rf_rate + 0.5 * sv**2) * tv) / (sv * np.sqrt(tv))
    d2 = d1 - sv * np.sqrt(tv)
    nd1 = normal_cdf(d1)
    npd1 = normal_pdf(d1)
    nd2 = normal_cdf(d2)

    is_call = typ[valid] == "C"
    delta_v = np.where(is_call, nd1, nd1 - 1.0)
    gamma_v = npd1 / (spot * sv * np.sqrt(tv))
    theta_call = (
        -(spot * npd1 * sv) / (2.0 * np.sqrt(tv))
        - rf_rate * kv * np.exp(-rf_rate * tv) * nd2
    )
    theta_put = (
        -(spot * npd1 * sv) / (2.0 * np.sqrt(tv))
        + rf_rate * kv * np.exp(-rf_rate * tv) * normal_cdf(-d2)
    )
    theta_v = np.where(is_call, theta_call, theta_put) / 365.0
    vega_v = spot * npd1 * np.sqrt(tv) / 100.0

    delta[valid] = delta_v
    gamma[valid] = gamma_v
    theta[valid] = theta_v
    vega[valid] = vega_v
    return pd.DataFrame({"delta": delta, "gamma": gamma, "theta": theta, "vega": vega})


def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )


def download_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"])
    out = flatten_ohlcv(df)
    out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out


def download_options_snapshot(ticker: yf.Ticker) -> tuple[pd.DataFrame, list[str]]:
    snapshot_ts = pd.Timestamp.now(tz="UTC")
    expiries = list(ticker.options)
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for exp in expiries:
        try:
            chain = ticker.option_chain(exp)
            for side, code in [("calls", "C"), ("puts", "P")]:
                side_df = getattr(chain, side).copy()
                if side_df.empty:
                    continue
                side_df["option_type"] = code
                side_df["expiry"] = pd.to_datetime(exp, utc=True)
                side_df["snapshot_ts"] = snapshot_ts
                frames.append(side_df)
        except Exception:
            failed.append(exp)

    if not frames:
        return pd.DataFrame(), failed
    out = pd.concat(frames, ignore_index=True)
    out["lastTradeDate"] = pd.to_datetime(out["lastTradeDate"], utc=True, errors="coerce")
    out["mid"] = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0)) / 2.0
    return out, failed


def feature_coverage_rows() -> Iterable[dict[str, str]]:
    return [
        {"source": "Yahoo stock OHLCV", "covers_report_features": "27-37 (mostly)"},
        {"source": "Yahoo options snapshot", "covers_report_features": "1-5,6,8,13-20,24 (snapshot-only)"},
        {"source": "Derived BS greeks", "covers_report_features": "4,21,23 (approximate)"},
        {"source": "Yahoo VIX/VIX3M", "covers_report_features": "38-40"},
        {"source": "Yahoo ^IRX", "covers_report_features": "41"},
        {"source": "Yahoo earnings/dividends", "covers_report_features": "43,44,45"},
        {"source": "Yahoo XLK", "covers_report_features": "42 (AAPL proxy)"},
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker = yf.Ticker("AAPL")
    spot_hist = download_ohlcv("AAPL", args.start, args.end)
    vix = download_ohlcv("^VIX", args.start, args.end)
    vix3m = download_ohlcv("^VIX3M", args.start, args.end)
    irx = download_ohlcv("^IRX", args.start, args.end)
    xlk = download_ohlcv("XLK", args.start, args.end)

    options_snapshot, failed_expiries = download_options_snapshot(ticker)
    rf_rate = float(irx["close"].dropna().iloc[-1] / 100.0) if not irx.empty else 0.04

    if not options_snapshot.empty and not spot_hist.empty:
        spot = float(spot_hist["close"].dropna().iloc[-1])
        ttm = ((options_snapshot["expiry"] - options_snapshot["snapshot_ts"]).dt.total_seconds() / (365.25 * 24 * 3600)).clip(lower=0.0)
        greeks = compute_bs_greeks(
            spot=spot,
            strike=options_snapshot["strike"],
            ttm_years=ttm,
            rf_rate=rf_rate,
            iv=options_snapshot["impliedVolatility"],
            option_type=options_snapshot["option_type"],
        )
        options_snapshot = pd.concat([options_snapshot, greeks], axis=1)
        options_snapshot["ttm_years"] = ttm
        options_snapshot["underlying_spot"] = spot
        options_snapshot["moneyness_s_over_k"] = spot / options_snapshot["strike"].replace(0, np.nan)

    earnings = pd.DataFrame()
    try:
        earnings = ticker.get_earnings_dates(limit=48).reset_index().rename(columns={"Earnings Date": "earnings_date"})
        if "earnings_date" in earnings.columns:
            earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"], utc=True, errors="coerce")
    except Exception:
        pass

    dividends = ticker.dividends.reset_index().rename(columns={"Date": "date", "Dividends": "dividend"})
    if not dividends.empty:
        dividends["date"] = pd.to_datetime(dividends["date"], utc=True)

    spot_hist.to_parquet(out_dir / "aapl_stock_ohlcv_1d_yahoo.parquet", index=False)
    vix.to_parquet(out_dir / "vix_ohlcv_1d_yahoo.parquet", index=False)
    vix3m.to_parquet(out_dir / "vix3m_ohlcv_1d_yahoo.parquet", index=False)
    irx.to_parquet(out_dir / "irx_ohlcv_1d_yahoo.parquet", index=False)
    xlk.to_parquet(out_dir / "xlk_ohlcv_1d_yahoo.parquet", index=False)
    dividends.to_parquet(out_dir / "aapl_dividends_yahoo.parquet", index=False)
    earnings.to_parquet(out_dir / "aapl_earnings_dates_yahoo.parquet", index=False)
    options_snapshot.to_parquet(out_dir / "aapl_options_snapshot_yahoo.parquet", index=False)

    coverage = pd.DataFrame(feature_coverage_rows())
    coverage.to_csv(out_dir / "aapl_feature_coverage_free_sources.csv", index=False)

    summary = {
        "date_generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {
            "aapl_stock_ohlcv_1d_yahoo": int(len(spot_hist)),
            "aapl_options_snapshot_yahoo": int(len(options_snapshot)),
            "aapl_dividends_yahoo": int(len(dividends)),
            "aapl_earnings_dates_yahoo": int(len(earnings)),
            "vix_ohlcv_1d_yahoo": int(len(vix)),
            "vix3m_ohlcv_1d_yahoo": int(len(vix3m)),
            "irx_ohlcv_1d_yahoo": int(len(irx)),
            "xlk_ohlcv_1d_yahoo": int(len(xlk)),
        },
        "failed_option_expiries": failed_expiries,
        "critical_gap": (
            "Free Yahoo endpoints do not provide a full historical daily options chain with historical IV/OI/bid/ask by contract. "
            "You now have a current-chain snapshot, not a full backfilled history."
        ),
    }
    (out_dir / "aapl_free_data_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
