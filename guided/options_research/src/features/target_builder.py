"""Build forward-return targets at 1, 3, 5-day horizons."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def build_targets(stock_df: pd.DataFrame, horizons: list[int] = (1, 3, 5)) -> pd.DataFrame:
    """Compute forward returns and direction labels per ticker.

    Parameters
    ----------
    stock_df : DataFrame
        Must contain columns: ticker, date, prc (absolute price).
    horizons : list of int
        Number of *trading days* ahead for each target.

    Returns
    -------
    DataFrame with columns: ticker, date, ret_{h}d, dir_{h}d for each h.
    """
    records = []

    for ticker, grp in stock_df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        daily_ret = pd.to_numeric(grp["ret"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        dates = grp["date"].values

        row_data = {"ticker": [ticker] * len(grp), "date": dates}

        for h in horizons:
            # Forward return using CRSP total returns (includes dividends).
            # Compound daily returns: ret_h = (1+r[t+1])*(1+r[t+2])*...*(1+r[t+h]) - 1
            ret = np.full(len(grp), np.nan)
            for i in range(len(grp) - h):
                window = daily_ret[i + 1: i + 1 + h]
                if np.any(np.isnan(window)):
                    ret[i] = np.nan
                else:
                    ret[i] = np.prod(1 + window) - 1

            nan_mask = np.isnan(ret)
            direction = np.where(nan_mask, np.nan, (ret > 0).astype(float))

            row_data[f"ret_{h}d"] = ret
            row_data[f"dir_{h}d"] = direction

        records.append(pd.DataFrame(row_data))

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def run(cfg=None):
    """Build and save targets."""
    if cfg is None:
        cfg = load_config()

    out_path = FEATURES_DIR / "targets.parquet"
    if out_path.exists():
        log.info("Targets already exist, skipping.")
        return

    stock_path = INTERIM_DIR / "stock_clean" / "stock_daily.parquet"
    if not stock_path.exists():
        log.error("Clean stock data not found")
        return

    stock = pd.read_parquet(stock_path)
    stock["date"] = pd.to_datetime(stock["date"])

    targets = build_targets(stock, horizons=cfg.targets.horizons)
    save_parquet(targets, out_path)
    log.info("Built targets: %d rows, horizons %s", len(targets), cfg.targets.horizons)


if __name__ == "__main__":
    run()
