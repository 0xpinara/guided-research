"""Download all required data from WRDS (Cboe options, CRSP, Compustat, FF).

Supports two modes:
  - Full Cboe access: per-year tables (cboe.optprice_2024, etc.)
  - Sample Cboe access: cboesamp.optprice (limited tickers/dates)

The script auto-detects which you have and adjusts accordingly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, RAW_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def _get_connection(username: str):
    """Open a WRDS connection."""
    import wrds
    return wrds.Connection(wrds_username=username)


# ---------------------------------------------------------------------------
# Cboe Options Data
# ---------------------------------------------------------------------------

def _detect_cboe_access(conn) -> str:
    """Detect whether we have full or sample Cboe access.

    Returns 'full', 'sample', or 'none'.
    """
    # Try full access
    try:
        conn.raw_sql("SELECT 1 FROM cboe.optprice_2024 LIMIT 1")
        log.info("Full Cboe access detected")
        return "full"
    except Exception:
        pass

    # Try sample
    try:
        conn.raw_sql("SELECT 1 FROM cboesamp.optprice LIMIT 1")
        log.info("Cboe sample access detected (limited tickers/dates)")
        return "sample"
    except Exception:
        pass

    log.warning("No Cboe options access detected")
    return "none"


def _download_cboe_sample(conn) -> None:
    """Download from cboesamp tables (limited data)."""
    out_dir = RAW_DIR / "wrds" / "hanweck"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get all options prices with contract info and IV/Greeks joined
    query = """
        SELECT
            e.ticker,
            o.date_ AS date,
            c.expdate AS expiration,
            c.strike,
            c.putcall AS call_put,
            o.bid AS best_bid,
            o.ask AS best_offer,
            o.volume,
            o.openint AS open_interest,
            iv.ivmid AS impl_volatility,
            iv.delta,
            iv.gamma,
            iv.vega,
            iv.theta,
            eq.close_ AS underlying_price
        FROM cboesamp.optprice o
        JOIN cboesamp.optcontract c ON o.optid = c.optid
        JOIN cboesamp.eqmaster e ON c.eqid = e.eqid
        LEFT JOIN cboesamp.ivlisted iv ON o.optid = iv.optid AND o.date_ = iv.date_
        LEFT JOIN cboesamp.eqprice eq ON c.eqid = eq.eqid AND o.date_ = eq.date_
    """

    log.info("Downloading Cboe sample options data...")
    df = conn.raw_sql(query)
    log.info("Downloaded %d rows from Cboe sample", len(df))

    # Normalize call_put column
    df["call_put"] = df["call_put"].str.strip().str.upper()
    # C/P format
    df["call_put"] = df["call_put"].replace({"CALL": "C", "PUT": "P"})

    # Save per-ticker
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker]
        out_path = out_dir / f"{ticker}_sample.parquet"
        save_parquet(ticker_df, out_path)
        log.info("  %s: %d rows, dates %s to %s",
                 ticker, len(ticker_df),
                 ticker_df["date"].min(), ticker_df["date"].max())


def _download_cboe_full(conn, tickers: list, start: str, end: str) -> None:
    """Download from full cboe per-year tables."""
    out_dir = RAW_DIR / "wrds" / "hanweck"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_year = int(start[:4])
    end_year = int(end[:4])

    # Get the ticker-to-eqid mapping
    ticker_str = "', '".join(tickers)
    eqmaster = conn.raw_sql(f"""
        SELECT eqid, ticker FROM cboe.eqmaster
        WHERE ticker IN ('{ticker_str}')
    """)
    ticker_eqid = dict(zip(eqmaster["ticker"], eqmaster["eqid"]))

    for ticker in tickers:
        eqid = ticker_eqid.get(ticker)
        if eqid is None:
            log.warning("No eqid found for %s", ticker)
            continue

        for year in range(start_year, end_year + 1):
            out_path = out_dir / f"{ticker}_{year}.parquet"
            if out_path.exists():
                log.info("Skipping %s %d (exists)", ticker, year)
                continue

            query = f"""
                SELECT
                    '{ticker}' AS ticker,
                    o.date_ AS date,
                    c.expdate AS expiration,
                    c.strike,
                    c.putcall AS call_put,
                    o.bid AS best_bid,
                    o.ask AS best_offer,
                    o.volume,
                    o.openint AS open_interest,
                    iv.ivmid AS impl_volatility,
                    iv.delta,
                    iv.gamma,
                    iv.vega,
                    iv.theta,
                    eq.close_ AS underlying_price
                FROM cboe.optprice_{year} o
                JOIN cboe.optcontract c ON o.optid = c.optid
                LEFT JOIN cboe.ivlisted_{year} iv
                    ON o.optid = iv.optid AND o.date_ = iv.date_
                LEFT JOIN cboe.eqprice eq
                    ON c.eqid = eq.eqid AND o.date_ = eq.date_
                WHERE c.eqid = {eqid}
            """
            try:
                df = conn.raw_sql(query)
                if df.empty:
                    log.warning("No data for %s %d", ticker, year)
                    continue
                df["call_put"] = df["call_put"].str.strip().str.upper()
                df["call_put"] = df["call_put"].replace({"CALL": "C", "PUT": "P"})
                save_parquet(df, out_path)
                log.info("%s %d: %d rows", ticker, year, len(df))
            except Exception as e:
                log.error("Failed %s %d: %s", ticker, year, str(e)[:100])

            time.sleep(0.3)


def download_options(conn, tickers: list, start: str, end: str) -> None:
    """Download options data, auto-detecting access level."""
    access = _detect_cboe_access(conn)
    if access == "full":
        _download_cboe_full(conn, tickers, start, end)
    elif access == "sample":
        _download_cboe_sample(conn)
    else:
        log.error("No options data access. Request Cboe access from your WRDS rep.")


# ---------------------------------------------------------------------------
# CRSP Stock Data
# ---------------------------------------------------------------------------

def download_crsp(conn, tickers: list, start: str, end: str) -> None:
    """Download CRSP daily stock data."""
    out_path = RAW_DIR / "wrds" / "crsp" / "stock_daily.parquet"
    if out_path.exists():
        log.info("CRSP already exists, skipping.")
        return

    (RAW_DIR / "wrds" / "crsp").mkdir(parents=True, exist_ok=True)

    ticker_str = "', '".join(tickers)
    query = f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.vol, a.shrout,
               b.ticker
        FROM crsp.dsf AS a
        JOIN crsp.dsenames AS b
          ON a.permno = b.permno
         AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE b.ticker IN ('{ticker_str}')
          AND a.date BETWEEN '{start}' AND '{end}'
    """
    df = conn.raw_sql(query)
    save_parquet(df, out_path)
    log.info("CRSP daily: %d rows", len(df))


# ---------------------------------------------------------------------------
# Compustat Earnings Dates
# ---------------------------------------------------------------------------

def download_earnings(conn, tickers: list, start: str, end: str) -> None:
    """Download quarterly earnings report dates."""
    out_path = RAW_DIR / "wrds" / "compustat" / "earnings_dates.parquet"
    if out_path.exists():
        log.info("Earnings dates already exist, skipping.")
        return

    (RAW_DIR / "wrds" / "compustat").mkdir(parents=True, exist_ok=True)

    ticker_str = "', '".join(tickers)

    # Earnings dates
    earn = conn.raw_sql(f"""
        SELECT gvkey, datadate, rdq, tic AS ticker
        FROM comp.fundq
        WHERE tic IN ('{ticker_str}')
          AND datadate BETWEEN '{start}' AND '{end}'
          AND rdq IS NOT NULL
    """)
    save_parquet(earn, out_path)
    log.info("Earnings dates: %d rows", len(earn))


# ---------------------------------------------------------------------------
# Fama-French Factors
# ---------------------------------------------------------------------------

def download_ff_factors(conn, start: str, end: str) -> None:
    """Download Fama-French 5 factors + momentum from WRDS."""
    out_path = RAW_DIR / "wrds" / "ff_factors" / "ff5_daily.parquet"
    if out_path.exists():
        log.info("FF factors already exist, skipping.")
        return

    (RAW_DIR / "wrds" / "ff_factors").mkdir(parents=True, exist_ok=True)

    df = conn.raw_sql(f"""
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff.fivefactors_daily
        WHERE date BETWEEN '{start}' AND '{end}'
    """)
    save_parquet(df, out_path)
    log.info("FF factors: %d rows", len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg=None):
    """Download all WRDS data."""
    if cfg is None:
        cfg = load_config()

    tickers = all_tickers(cfg)
    start, end = cfg.start_date, cfg.end_date

    conn = _get_connection(cfg.wrds.username)

    log.info("=== Downloading Options Data ===")
    download_options(conn, tickers, start, end)

    log.info("=== Downloading CRSP Stock Data ===")
    download_crsp(conn, tickers, start, end)

    log.info("=== Downloading Earnings Dates ===")
    download_earnings(conn, tickers, start, end)

    log.info("=== Downloading Fama-French Factors ===")
    download_ff_factors(conn, start, end)

    conn.close()
    log.info("=== WRDS Download Complete ===")


if __name__ == "__main__":
    run()
