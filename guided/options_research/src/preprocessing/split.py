"""Temporal train/val/test split with leakage checks."""

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.io_helpers import save_parquet, FEATURES_DIR, SPLITS_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    max_target_horizon: int = 5,
) -> pd.DataFrame:
    """Assign each row to train/val/test based on date.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'date' column.
    train_frac, val_frac, test_frac : float
        Fractions of unique dates for each split.
    max_target_horizon : int
        Maximum forward horizon in trading days. Used to enforce a gap
        between train and val to prevent target leakage.

    Returns
    -------
    DataFrame with columns: ticker, date, split
    """
    dates = sorted(df["date"].unique())
    n = len(dates)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # rest goes to test

    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train: n_train + n_val])
    test_dates = set(dates[n_train + n_val:])

    # Enforce a target-horizon buffer between *both* adjacent splits.
    #
    # If the last training target is a 5-day-ahead return, it must not
    # include any validation date.  The same logic applies to the last
    # validation target and the first test date; otherwise early stopping
    # can be weakly conditioned on returns that belong to the test block.
    if max_target_horizon > 0:
        train_gap_dates = set(dates[max(0, n_train - max_target_horizon): n_train])
        val_gap_start = n_train + n_val - max_target_horizon
        val_gap_dates = set(dates[max(n_train, val_gap_start): n_train + n_val])

        train_dates -= train_gap_dates
        val_dates -= val_gap_dates

        log.warning(
            "Removed %d train dates and %d val dates to enforce %d-trading-day split buffers",
            len(train_gap_dates), len(val_gap_dates), max_target_horizon,
        )

    # Leakage check: ensure non-overlap after buffer removal.
    if train_dates and val_dates:
        assert max(train_dates) < min(val_dates), "Train/val overlap detected!"
    if val_dates and test_dates:
        assert max(val_dates) < min(test_dates), "Val/test overlap detected!"

    def assign(dt):
        if dt in train_dates:
            return "train"
        elif dt in val_dates:
            return "val"
        elif dt in test_dates:
            return "test"
        return "excluded"

    result = df[["ticker", "date"]].copy()
    result["split"] = result["date"].map(assign)

    # Drop excluded rows
    result = result[result["split"] != "excluded"]

    # Print stats
    for s in ["train", "val", "test"]:
        sub = result[result["split"] == s]
        s_dates = sub["date"].unique()
        log.info(
            "%s: %d rows, %d dates (%s to %s), %d tickers",
            s, len(sub), len(s_dates),
            min(s_dates).date() if len(s_dates) > 0 else "N/A",
            max(s_dates).date() if len(s_dates) > 0 else "N/A",
            sub["ticker"].nunique(),
        )

    return result


def run(cfg=None):
    """Create and save temporal splits."""
    if cfg is None:
        cfg = load_config()

    out_path = SPLITS_DIR / "split_indices.parquet"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Load any feature panel to get (ticker, date) pairs
    panel_path = FEATURES_DIR / "resolution_1_scalar" / "options_features_all.parquet"
    if not panel_path.exists():
        # Try stock features
        panel_path = FEATURES_DIR / "stock_features.parquet"
    if not panel_path.exists():
        log.error("No feature panel found to split")
        return

    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])

    splits = temporal_split(
        df,
        train_frac=cfg.split.train_frac,
        val_frac=cfg.split.val_frac,
        test_frac=cfg.split.test_frac,
        max_target_horizon=max(cfg.targets.horizons),
    )

    save_parquet(splits, out_path)
    log.info("=== Split Complete ===")


if __name__ == "__main__":
    run()
