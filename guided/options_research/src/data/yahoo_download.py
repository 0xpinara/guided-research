"""Download supplementary data from Yahoo Finance (VIX, sector ETFs, dividends)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, RAW_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

YAHOO_DIR = RAW_DIR / "yahoo"


def _download_price_series(ticker: str, start: str, end: str, name: str) -> pd.DataFrame | None:
    """Download daily OHLCV for a single ticker from Yahoo Finance."""
    out_path = YAHOO_DIR / f"{name}.parquet"
    if out_path.exists():
        log.info("Already have %s, skipping.", name)
        return pd.read_parquet(out_path)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            log.warning("No Yahoo data for %s", ticker)
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "date"
        save_parquet(df.reset_index(), out_path)
        log.info("Downloaded Yahoo %s: %d rows", name, len(df))
        return df
    except Exception as e:
        log.error("Yahoo download failed for %s: %s", ticker, e)
        return None


def download_vix(start: str, end: str) -> None:
    """Download VIX daily close."""
    _download_price_series("^VIX", start, end, "vix_daily")
    time.sleep(0.5)


def download_vix3m(start: str, end: str) -> None:
    """Download VIX3M daily close."""
    _download_price_series("^VIX3M", start, end, "vix3m_daily")
    time.sleep(0.5)


def download_tbill(start: str, end: str) -> None:
    """Download 13-week T-bill rate."""
    _download_price_series("^IRX", start, end, "tbill_daily")
    time.sleep(0.5)


def download_sector_etfs(etfs: list[str], start: str, end: str) -> None:
    """Download daily close and returns for sector ETFs."""
    for etf in etfs:
        name = f"sector_{etf.lower()}"
        df = _download_price_series(etf, start, end, name)
        time.sleep(0.5)


def download_dividends(tickers: list[str], start: str, end: str) -> None:
    """Download dividend ex-dates for each ticker."""
    out_path = YAHOO_DIR / "dividends.parquet"
    if out_path.exists():
        log.info("Dividends already downloaded, skipping.")
        return

    records = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            if divs is not None and len(divs) > 0:
                df_div = divs.reset_index()
                df_div.columns = ["ex_date", "dividend"]
                df_div["ticker"] = ticker
                # Filter to date range
                df_div["ex_date"] = pd.to_datetime(df_div["ex_date"]).dt.tz_localize(None)
                df_div = df_div[
                    (df_div["ex_date"] >= start) & (df_div["ex_date"] <= end)
                ]
                records.append(df_div)
                log.info("Got %d dividend records for %s", len(df_div), ticker)
        except Exception as e:
            log.warning("Failed to get dividends for %s: %s", ticker, e)
        time.sleep(0.5)

    if records:
        df_all = pd.concat(records, ignore_index=True)
        save_parquet(df_all, out_path)
        log.info("Saved %d total dividend records", len(df_all))
    else:
        log.warning("No dividend data collected")


def run(cfg=None):
    """Download all Yahoo Finance data."""
    if cfg is None:
        cfg = load_config()

    YAHOO_DIR.mkdir(parents=True, exist_ok=True)

    tickers = all_tickers(cfg)
    start, end = cfg.start_date, cfg.end_date
    etfs = cfg.yahoo_fallback.sector_etfs

    log.info("=== Downloading Yahoo Finance Data ===")

    download_vix(start, end)
    download_vix3m(start, end)
    download_tbill(start, end)
    download_sector_etfs(etfs, start, end)
    download_dividends(tickers, start, end)

    log.info("=== Yahoo Finance Download Complete ===")


if __name__ == "__main__":
    run()
