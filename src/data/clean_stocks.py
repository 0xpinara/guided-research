"""Clean CRSP stock data and merge with VIX, FF factors, earnings, dividends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import load_config, all_tickers, ticker_to_sector, sector_to_etf
from src.utils.io_helpers import save_parquet, RAW_DIR, INTERIM_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

STOCK_CLEAN_DIR = INTERIM_DIR / "stock_clean"
YAHOO_DIR = RAW_DIR / "yahoo"


def _load_yahoo_series(name: str, col: str = "Close") -> pd.Series:
    """Load a Yahoo-downloaded series by name, returning date-indexed values."""
    path = YAHOO_DIR / f"{name}.parquet"
    if not path.exists():
        log.warning("Missing Yahoo file: %s", path)
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[col]


def _load_sector_etf_returns(etfs: list[str]) -> pd.DataFrame:
    """Load sector ETF daily returns, one column per ETF."""
    frames = {}
    for etf in etfs:
        path = YAHOO_DIR / f"sector_{etf.lower()}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        frames[etf] = df["Close"].pct_change()
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def _compute_earnings_proximity(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Add days_to_next_earnings and earnings_flag columns."""
    earn_path = RAW_DIR / "wrds" / "compustat" / "earnings_dates.parquet"
    if not earn_path.exists():
        log.warning("No earnings dates file; setting earnings features to NaN")
        stock_df["days_to_next_earnings"] = np.nan
        stock_df["earnings_flag"] = 0
        return stock_df

    earn = pd.read_parquet(earn_path)
    earn["rdq"] = pd.to_datetime(earn["rdq"])

    for ticker in stock_df["ticker"].unique():
        mask_t = stock_df["ticker"] == ticker
        dates_t = stock_df.loc[mask_t, "date"]

        earn_t = earn[earn["ticker"] == ticker]["rdq"].dropna().sort_values().unique()
        if len(earn_t) == 0:
            stock_df.loc[mask_t, "days_to_next_earnings"] = np.nan
            stock_df.loc[mask_t, "earnings_flag"] = 0
            continue

        earn_t = pd.to_datetime(earn_t)
        # For each trading date, find the next earnings date
        idx = np.searchsorted(earn_t, dates_t.values)
        days_to = np.full(len(dates_t), np.nan)
        for i, (dt, ei) in enumerate(zip(dates_t.values, idx)):
            if ei < len(earn_t):
                days_to[i] = (earn_t[ei] - pd.Timestamp(dt)).days

        stock_df.loc[mask_t, "days_to_next_earnings"] = days_to

    stock_df["earnings_flag"] = (stock_df["days_to_next_earnings"] <= 7).astype(int)
    return stock_df


def _compute_dividend_proximity(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Add days_to_ex_div column."""
    div_path = YAHOO_DIR / "dividends.parquet"
    if not div_path.exists():
        log.warning("No dividends file; setting dividend features to 999")
        stock_df["days_to_ex_div"] = 999
        return stock_df

    divs = pd.read_parquet(div_path)
    divs["ex_date"] = pd.to_datetime(divs["ex_date"])

    stock_df["days_to_ex_div"] = 999.0

    for ticker in stock_df["ticker"].unique():
        mask_t = stock_df["ticker"] == ticker
        dates_t = stock_df.loc[mask_t, "date"]

        ex_dates = divs[divs["ticker"] == ticker]["ex_date"].sort_values().values
        if len(ex_dates) == 0:
            continue

        ex_dates = pd.to_datetime(ex_dates)
        idx = np.searchsorted(ex_dates, dates_t.values)
        days_to = np.full(len(dates_t), 999.0)
        for i, (dt, ei) in enumerate(zip(dates_t.values, idx)):
            if ei < len(ex_dates):
                d = (ex_dates[ei] - pd.Timestamp(dt)).days
                if d <= 90:
                    days_to[i] = d

        stock_df.loc[mask_t, "days_to_ex_div"] = days_to

    return stock_df


def run(cfg=None):
    """Clean stock data and merge all supplementary sources."""
    if cfg is None:
        cfg = load_config()

    STOCK_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STOCK_CLEAN_DIR / "stock_daily.parquet"
    if out_path.exists():
        log.info("stock_daily.parquet already exists, skipping.")
        return

    # Load CRSP
    crsp_path = RAW_DIR / "wrds" / "crsp" / "stock_daily.parquet"
    if not crsp_path.exists():
        log.error("CRSP data not found at %s", crsp_path)
        return

    df = pd.read_parquet(crsp_path)
    df["date"] = pd.to_datetime(df["date"])
    df["prc"] = df["prc"].abs()
    df["mktcap"] = df["prc"] * df["shrout"] * 1000  # shrout in thousands

    # Flag extreme returns
    extreme = df["ret"].abs() > 0.5
    if extreme.any():
        log.warning("Found %d observations with |ret| > 0.5", extreme.sum())

    # Merge VIX
    vix = _load_yahoo_series("vix_daily", "Close")
    if not vix.empty:
        df = df.merge(vix.rename("vix_close").reset_index(), on="date", how="left")
    else:
        df["vix_close"] = np.nan

    # Merge VIX3M
    vix3m = _load_yahoo_series("vix3m_daily", "Close")
    if not vix3m.empty:
        df = df.merge(vix3m.rename("vix3m_close").reset_index(), on="date", how="left")
    else:
        df["vix3m_close"] = np.nan

    # Merge T-bill rate
    tbill = _load_yahoo_series("tbill_daily", "Close")
    if not tbill.empty:
        df = df.merge(tbill.rename("tbill_rate").reset_index(), on="date", how="left")
    else:
        df["tbill_rate"] = np.nan

    # Merge sector ETF returns
    t2s = ticker_to_sector(cfg)
    s2e = sector_to_etf(cfg)
    etf_rets = _load_sector_etf_returns(list(s2e.values()))
    if not etf_rets.empty:
        df["sector_etf"] = df["ticker"].map(lambda t: s2e.get(t2s.get(t, ""), ""))
        # Map each row to its sector ETF return
        etf_rets_long = etf_rets.stack().reset_index()
        etf_rets_long.columns = ["date", "sector_etf", "sector_etf_return"]
        df = df.merge(etf_rets_long, on=["date", "sector_etf"], how="left")
        df.drop(columns=["sector_etf"], inplace=True)
    else:
        df["sector_etf_return"] = np.nan

    # Merge Fama-French factors
    ff_path = RAW_DIR / "wrds" / "ff_factors" / "ff5_daily.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path)
        ff["date"] = pd.to_datetime(ff["date"])
        df = df.merge(ff, on="date", how="left")
    else:
        for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
            df[col] = np.nan

    # Earnings proximity
    df = _compute_earnings_proximity(df)

    # Dividend proximity
    df = _compute_dividend_proximity(df)

    # Sort
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    save_parquet(df, out_path)
    log.info("=== Stock Cleaning Complete: %d rows ===", len(df))


if __name__ == "__main__":
    run()
