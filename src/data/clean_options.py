"""Filter and clean raw options data to produce per-ticker cleaned parquets.

Supports two data sources:
  - Cboe/WRDS (from wrds_download.py)
  - philippdubach/options-data GitHub dataset (preferred, 2008-2025)

Auto-detects which source is available per ticker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, RAW_DIR, INTERIM_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

OPTIONS_CLEAN_DIR = INTERIM_DIR / "options_clean"
GITHUB_DIR = RAW_DIR / "github_options"
HANWECK_DIR = RAW_DIR / "wrds" / "hanweck"


def _load_github_options(ticker: str) -> pd.DataFrame:
    """Load options data from philippdubach/options-data format."""
    path = GITHUB_DIR / f"{ticker.lower()}_options.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Rename columns to our standard format
    col_map = {
        "symbol": "ticker",
        "expiration": "expiration",
        "strike": "strike",
        "type": "call_put",       # "call" / "put"
        "bid": "best_bid",
        "ask": "best_offer",
        "volume": "volume",
        "open_interest": "open_interest",
        "implied_volatility": "impl_volatility",
        "delta": "delta",
        "gamma": "gamma",
        "vega": "vega",
        "theta": "theta",
        "mark": "mid_price",
        "date": "date",
    }
    df = df.rename(columns=col_map)

    # Normalize call_put to C/P
    df["call_put"] = df["call_put"].str.strip().str.upper()
    df["call_put"] = df["call_put"].replace({"CALL": "C", "PUT": "P"})

    # Ensure ticker is uppercase
    df["ticker"] = ticker.upper()

    return df


def _load_cboe_options(ticker: str) -> pd.DataFrame:
    """Load options data from Cboe/WRDS format (fallback)."""
    files = sorted(HANWECK_DIR.glob(f"{ticker}*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _load_underlying_prices(ticker: str) -> pd.Series:
    """Load underlying prices from github dataset or CRSP."""
    # Try github underlying first
    github_path = GITHUB_DIR / f"{ticker.lower()}_underlying.parquet"
    if github_path.exists():
        df = pd.read_parquet(github_path)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["close"]

    # Fallback to CRSP
    crsp_path = RAW_DIR / "wrds" / "crsp" / "stock_daily.parquet"
    if crsp_path.exists():
        df = pd.read_parquet(crsp_path)
        df = df[df["ticker"] == ticker].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["prc"] = df["prc"].abs()
        return df.set_index("date")["prc"]

    return pd.Series(dtype=float)


def clean_ticker(ticker: str, cfg) -> pd.DataFrame | None:
    """Clean options data for a single ticker.

    Tries github data first, falls back to Cboe/WRDS.
    """
    # Load from best available source
    df = _load_github_options(ticker)
    source = "github"
    if df.empty:
        df = _load_cboe_options(ticker)
        source = "cboe"
    if df.empty:
        log.warning("No options data for %s from any source", ticker)
        return None

    n_raw = len(df)
    log.info("Loading %s from %s: %d raw rows", ticker, source, n_raw)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Get underlying prices if not already present
    if "underlying_price" not in df.columns:
        prices = _load_underlying_prices(ticker)
        if not prices.empty:
            df["underlying_price"] = df["date"].map(prices)
        else:
            log.error("No underlying price for %s", ticker)
            return None

    # Drop rows without underlying price
    df = df.dropna(subset=["underlying_price"])
    df = df[df["underlying_price"] > 0]

    # Filter to config date range
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    if df.empty:
        log.warning("No data for %s within date range %s to %s", ticker, start.date(), end.date())
        return None

    # Derived fields
    df["dte"] = (df["expiration"] - df["date"]).dt.days
    df["moneyness"] = df["strike"] / df["underlying_price"]
    if "mid_price" not in df.columns:
        df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2
    df["spread_pct"] = np.where(
        df["mid_price"] > 0,
        (df["best_offer"] - df["best_bid"]) / df["mid_price"],
        np.nan,
    )
    df["is_call"] = (df["call_put"] == "C").astype(int)
    df["is_otm"] = (
        ((df["is_call"] == 1) & (df["moneyness"] > 1))
        | ((df["is_call"] == 0) & (df["moneyness"] < 1))
    ).astype(int)

    filt = cfg.options_filters

    # Compute zero_dte_volume_share BEFORE filtering short-dte contracts
    total_vol_pre = df.groupby("date")["volume"].sum()
    zero_dte_vol = df[df["dte"] <= 1].groupby("date")["volume"].sum()
    zero_dte_share = (zero_dte_vol / total_vol_pre).rename("zero_dte_volume_share")

    # Apply filters
    mask = (
        (df["open_interest"] >= filt.min_open_interest)
        & (df["best_bid"] >= filt.min_bid)
        & (df["moneyness"] >= filt.moneyness_range[0])
        & (df["moneyness"] <= filt.moneyness_range[1])
        & (df["dte"] >= filt.dte_range[0])
        & (df["dte"] <= filt.dte_range[1])
        & (df["impl_volatility"].between(0.01, 5.0))
        & (df["volume"] >= 0)
    )
    df = df[mask].copy()

    # Deduplicate
    df = df.sort_values("volume", ascending=False)
    df = df.drop_duplicates(subset=["date", "strike", "expiration", "call_put"], keep="first")

    # Attach zero_dte_volume_share
    df = df.merge(zero_dte_share.reset_index(), on="date", how="left")
    df["zero_dte_volume_share"] = df["zero_dte_volume_share"].fillna(0)

    n_clean = len(df)
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    avg_per_day = n_clean / df["date"].nunique() if df["date"].nunique() > 0 else 0

    log.info(
        "%s: %d raw -> %d clean (%.1f%%), dates %s, avg %.0f contracts/day",
        ticker, n_raw, n_clean, 100 * n_clean / max(n_raw, 1),
        date_range, avg_per_day,
    )

    return df


def run(cfg=None):
    """Clean options data for all tickers."""
    if cfg is None:
        cfg = load_config()

    OPTIONS_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in all_tickers(cfg):
        out_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"
        if out_path.exists():
            log.info("Skipping %s (already cleaned)", ticker)
            continue

        df = clean_ticker(ticker, cfg)
        if df is not None and not df.empty:
            save_parquet(df, out_path)

    log.info("=== Options Cleaning Complete ===")


if __name__ == "__main__":
    run()
