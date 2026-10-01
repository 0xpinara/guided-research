"""Merge cleaned options and stock data into per-ticker master files."""

from __future__ import annotations

import pandas as pd

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, INTERIM_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

MERGED_DIR = INTERIM_DIR / "merged"
OPTIONS_CLEAN_DIR = INTERIM_DIR / "options_clean"
STOCK_CLEAN_DIR = INTERIM_DIR / "stock_clean"


def merge_ticker(ticker: str, min_daily_contracts: int) -> pd.DataFrame | None:
    """Merge options and stock data for a single ticker.

    Returns a stock-day-level DataFrame with contract counts and a
    thin_options_day flag.  The actual contract-level data stays in the
    options_clean directory and is loaded on the fly during feature
    engineering.
    """
    stock_path = STOCK_CLEAN_DIR / "stock_daily.parquet"
    opts_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"

    if not stock_path.exists():
        log.error("Stock data not found")
        return None

    # Load stock data for this ticker
    stock = pd.read_parquet(stock_path)
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock[stock["ticker"] == ticker].copy()

    if stock.empty:
        log.warning("No stock data for %s", ticker)
        return None

    # Count qualifying options contracts per day
    if opts_path.exists():
        opts = pd.read_parquet(opts_path)
        opts["date"] = pd.to_datetime(opts["date"])
        daily_counts = opts.groupby("date").size().rename("n_contracts")
        stock = stock.merge(daily_counts.reset_index(), on="date", how="left")
        stock["n_contracts"] = stock["n_contracts"].fillna(0).astype(int)
    else:
        log.warning("No cleaned options for %s", ticker)
        stock["n_contracts"] = 0

    stock["thin_options_day"] = (stock["n_contracts"] < min_daily_contracts).astype(int)

    thin_pct = stock["thin_options_day"].mean() * 100
    log.info(
        "%s: %d trading days, %d with options, %.1f%% thin",
        ticker, len(stock), (stock["n_contracts"] > 0).sum(), thin_pct,
    )

    return stock


def run(cfg=None):
    """Merge all tickers."""
    if cfg is None:
        cfg = load_config()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    min_contracts = cfg.options_filters.min_daily_contracts

    for ticker in all_tickers(cfg):
        out_path = MERGED_DIR / f"{ticker}.parquet"
        if out_path.exists():
            log.info("Skipping %s (already merged)", ticker)
            continue

        df = merge_ticker(ticker, min_contracts)
        if df is not None and not df.empty:
            save_parquet(df, out_path)

    log.info("=== Merge Complete ===")


if __name__ == "__main__":
    run()
