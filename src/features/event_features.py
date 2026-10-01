"""Event and temporal features (feat_43 through feat_48)."""

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.io_helpers import save_parquet, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the 3rd Friday of the given month (monthly options expiration)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    # Find first Friday
    offset = (4 - first.weekday()) % 7  # 4 = Friday
    first_friday = first + pd.Timedelta(days=offset)
    return first_friday + pd.Timedelta(weeks=2)


def _days_to_monthly_opex(date: pd.Timestamp) -> int:
    """Calendar days to the next monthly options expiration (3rd Friday)."""
    opex = _third_friday(date.year, date.month)
    if date.date() > opex.date():
        # Next month
        if date.month == 12:
            opex = _third_friday(date.year + 1, 1)
        else:
            opex = _third_friday(date.year, date.month + 1)
    return (opex - date).days


def _quarter_end_flag(date: pd.Timestamp, n_trading_days: int = 5) -> int:
    """1 if within last n trading days of quarter."""
    quarter_end_months = {3, 6, 9, 12}
    month = date.month
    # Check if we're near the end of a quarter-end month
    if month in quarter_end_months:
        last_day = pd.Timestamp(year=date.year, month=month, day=1) + pd.offsets.MonthEnd(0)
        bdays_to_end = np.busday_count(
            date.date(), last_day.date()
        )
        return 1 if bdays_to_end <= n_trading_days else 0
    return 0


def compute_event_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features 43-48 for the full stock panel.

    Features 43-45 (earnings, dividends) are already in stock_df from clean_stocks.
    This adds features 46-48 (day_of_week, days_to_opex, quarter_end_flag).
    """
    df = stock_df[["ticker", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    # feat_43, 44, 45: already in stock_df
    for col, feat in [("days_to_next_earnings", "feat_43"),
                       ("earnings_flag", "feat_44"),
                       ("days_to_ex_div", "feat_45")]:
        if col in stock_df.columns:
            df[feat] = stock_df[col].values
        else:
            df[feat] = np.nan

    # feat_46: day of week
    df["feat_46"] = df["date"].dt.weekday

    # feat_47: days to monthly opex
    df["feat_47"] = df["date"].apply(_days_to_monthly_opex)

    # feat_48: quarter end flag
    df["feat_48"] = df["date"].apply(_quarter_end_flag)

    return df


def run(cfg=None):
    """Compute event features."""
    if cfg is None:
        cfg = load_config()

    stock_path = INTERIM_DIR / "stock_clean" / "stock_daily.parquet"
    if not stock_path.exists():
        log.error("Clean stock data not found")
        return

    stock = pd.read_parquet(stock_path)
    stock["date"] = pd.to_datetime(stock["date"])

    df = compute_event_features(stock)
    out_path = FEATURES_DIR / "event_features.parquet"
    save_parquet(df, out_path)
    log.info("=== Event Features Complete: %d rows ===", len(df))


if __name__ == "__main__":
    run()
