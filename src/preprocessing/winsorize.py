"""Train-only winsorization of feature values."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.io_helpers import SPLITS_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)


def winsorize(
    df: pd.DataFrame,
    feature_cols: list[str],
    splits: pd.DataFrame,
    quantiles: tuple[float, float] = (0.001, 0.999),
) -> tuple[pd.DataFrame, dict]:
    """Clip features to train-set-derived quantile bounds.

    Parameters
    ----------
    df : DataFrame
        Feature panel with ticker, date, and feature columns.
    feature_cols : list[str]
        Names of feature columns to winsorize.
    splits : DataFrame
        Must have columns: ticker, date, split.
    quantiles : tuple
        Lower and upper quantile bounds.

    Returns
    -------
    df_clipped : DataFrame with clipped features.
    bounds : dict of {feature: (clip_low, clip_high)}.
    """
    # Merge splits
    merged = df.merge(splits[["ticker", "date", "split"]], on=["ticker", "date"], how="left")
    train_mask = merged["split"] == "train"

    bounds = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        train_vals = merged.loc[train_mask, col].dropna()
        if len(train_vals) == 0:
            bounds[col] = (np.nan, np.nan)
            continue
        lo = train_vals.quantile(quantiles[0])
        hi = train_vals.quantile(quantiles[1])
        bounds[col] = (lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)

    n_clipped = sum(1 for lo, hi in bounds.values() if not np.isnan(lo))
    log.info("Winsorized %d features at [%.4f, %.4f] quantiles", n_clipped, *quantiles)

    return df, bounds
