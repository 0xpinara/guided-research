"""Train-only z-score normalization."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

log = setup_logger(__name__)


def normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    splits: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Z-score normalize features using train-set mean and std.

    Parameters
    ----------
    df : DataFrame
        Feature panel.
    feature_cols : list[str]
        Feature columns to normalize.
    splits : DataFrame
        With columns: ticker, date, split.

    Returns
    -------
    df_normalized : DataFrame with normalized features.
    stats : dict of {feature: (mean, std)}.
    """
    merged = df.merge(splits[["ticker", "date", "split"]], on=["ticker", "date"], how="left")
    train_mask = merged["split"] == "train"

    stats = {}
    constant_features = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        train_vals = merged.loc[train_mask, col]
        mu = train_vals.mean()
        sigma = train_vals.std()
        stats[col] = (mu, sigma)

        if sigma == 0 or np.isnan(sigma):
            df[col] = 0.0
            constant_features.append(col)
        else:
            df[col] = (df[col] - mu) / sigma

    if constant_features:
        log.warning("Constant features (set to 0): %s", constant_features)

    log.info("Normalized %d features (z-score, train-derived)", len(stats))
    return df, stats


def rank_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Cross-sectional rank normalization per Gu, Kelly, and Xiu (2020).

    At each date, rank all tickers on each feature and map to [-1, 1].
    This removes time-varying level effects and isolates relative
    cross-sectional positioning, which drives return differences.

    Parameters
    ----------
    df : DataFrame
        Feature panel with columns: ticker, date, feat_*.
    feature_cols : list[str]
        Feature columns to rank-normalize.

    Returns
    -------
    df_ranked : DataFrame with rank-normalized features.
    """
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue
        # Rank within each date, map to [-1, 1]
        # pct=True gives percentile ranks in [0, 1]; rescale to [-1, 1]
        df[col] = df.groupby("date")[col].rank(pct=True, method="average")
        df[col] = df[col] * 2 - 1  # [0,1] → [-1,1]

    log.info("Rank-normalized %d features (cross-sectional, per-date)", len(feature_cols))
    return df
