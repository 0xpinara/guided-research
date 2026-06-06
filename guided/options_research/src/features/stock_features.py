"""Stock and market features (feat_27 through feat_42) -- Model A features."""

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.io_helpers import save_parquet, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rsi(returns: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average of gains/losses."""
    gain = returns.clip(lower=0)
    loss = -returns.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_stock_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute features 27-42 for a single ticker group.

    Expects grp sorted by date with columns: date, prc, ret, vol,
    vix_close, vix3m_close, tbill_rate, sector_etf_return.
    All features are backward-looking only.
    """
    grp = grp.sort_values("date").copy()
    price = grp["prc"]
    ret = grp["ret"]

    # --- Lagged returns ---
    grp["feat_27"] = ret  # ret_1d_lag (today's realized return)
    grp["feat_28"] = price.pct_change(5)   # ret_5d_lag
    grp["feat_29"] = price.pct_change(10)  # ret_10d_lag

    # --- Historical volatility ---
    hvol_10d = ret.rolling(10, min_periods=10).std() * np.sqrt(252)
    grp["feat_31"] = ret.rolling(30, min_periods=30).std() * np.sqrt(252)  # hvol_30d
    # HVol ratio (10d/30d): captures vol-of-vol regime shifts — replaces raw HVol 10d
    # which was 0.819 correlated with feat_31 and added little incremental info
    grp["feat_30"] = hvol_10d / grp["feat_31"].replace(0, np.nan)  # hvol_ratio

    # --- RSI ---
    grp["feat_32"] = _rsi(ret, 14)

    # --- MACD (normalized by price for cross-ticker comparability) ---
    ema12 = _ema(price, 12)
    ema26 = _ema(price, 26)
    macd_raw = ema12 - ema26
    grp["feat_33"] = macd_raw / price.replace(0, np.nan)  # macd_value / price
    grp["feat_34"] = _ema(grp["feat_33"], 9)               # macd_signal (of normalized)

    # --- Bollinger position ---
    sma20 = _sma(price, 20)
    std20 = price.rolling(20, min_periods=20).std()
    grp["feat_35"] = np.where(
        std20 > 0, (price - sma20) / (2 * std20), 0
    )

    # --- Distance from SMA50 ---
    sma50 = _sma(price, 50)
    grp["feat_36"] = (price - sma50) / sma50.replace(0, np.nan)

    # --- ATR proxy (from returns since CRSP may not have high/low) ---
    atr_proxy = ret.abs().ewm(span=14, adjust=False).mean()
    grp["feat_37"] = atr_proxy  # already normalized (it's a return)

    # --- VIX ---
    grp["feat_38"] = grp["vix_close"]
    vix = grp["vix_close"]
    grp["feat_39"] = vix.pct_change(5)  # vix_5d_change

    # --- VIX term structure ---
    vix3m = grp.get("vix3m_close", pd.Series(np.nan, index=grp.index))
    grp["feat_40"] = np.where(
        vix > 0, (vix3m - vix) / vix, np.nan
    )

    # --- Risk-free rate ---
    grp["feat_41"] = grp.get("tbill_rate", pd.Series(np.nan, index=grp.index)) / 100.0

    # --- Sector ETF return ---
    grp["feat_42"] = grp.get("sector_etf_return", pd.Series(np.nan, index=grp.index))

    return grp


def run(cfg=None):
    """Compute stock features for all tickers."""
    if cfg is None:
        cfg = load_config()

    stock_path = INTERIM_DIR / "stock_clean" / "stock_daily.parquet"
    if not stock_path.exists():
        log.error("Clean stock data not found")
        return

    stock = pd.read_parquet(stock_path)
    stock["date"] = pd.to_datetime(stock["date"])

    results = []
    for ticker, grp in stock.groupby("ticker"):
        featured = compute_stock_features(grp)
        results.append(featured)
        log.info("Computed stock features for %s: %d rows", ticker, len(featured))

    df = pd.concat(results, ignore_index=True)

    # Keep only relevant columns
    feat_cols = [f"feat_{i:02d}" for i in range(27, 43)]
    keep = ["ticker", "date"] + feat_cols
    df = df[[c for c in keep if c in df.columns]]

    out_path = FEATURES_DIR / "stock_features.parquet"
    save_parquet(df, out_path)
    log.info("=== Stock Features Complete: %d rows ===", len(df))


if __name__ == "__main__":
    run()
