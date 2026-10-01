"""Train-only imputation: ticker median -> global median -> zero."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

log = setup_logger(__name__)


def impute(
    df: pd.DataFrame,
    feature_cols: list[str],
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Impute NaN values using train-set statistics.

    Strategy (applied in order):
    1. Per-ticker median from train set.
    2. Global median from train set.
    3. Zero.

    Parameters
    ----------
    df : DataFrame
        Feature panel.
    feature_cols : list[str]
        Feature columns to impute.
    splits : DataFrame
        With columns: ticker, date, split.

    Returns
    -------
    DataFrame with no NaN in feature columns.
    """
    merged = df.merge(splits[["ticker", "date", "split"]], on=["ticker", "date"], how="left")
    train_mask = merged["split"] == "train"

    for col in feature_cols:
        if col not in df.columns:
            continue

        n_na_before = df[col].isna().sum()
        if n_na_before == 0:
            continue

        # Step 1: per-ticker median (train only)
        ticker_medians = merged.loc[train_mask].groupby("ticker")[col].median()
        for ticker, med in ticker_medians.items():
            if np.isnan(med):
                continue
            mask = (df["ticker"] == ticker) & df[col].isna()
            df.loc[mask, col] = med

        # Step 2: global median (train only)
        global_med = merged.loc[train_mask, col].median()
        if not np.isnan(global_med):
            df[col] = df[col].fillna(global_med)

        # Step 3: zero
        df[col] = df[col].fillna(0.0)

        n_imputed = n_na_before - df[col].isna().sum()
        pct = n_na_before / len(df) * 100
        if pct > 50:
            log.warning("Feature %s: %.1f%% imputed (>50%% — may be unreliable)", col, pct)
        elif n_imputed > 0:
            log.info("Feature %s: %d values imputed (%.1f%%)", col, n_imputed, pct)

    # Final check
    remaining_na = df[feature_cols].isna().sum().sum()
    assert remaining_na == 0, f"Still {remaining_na} NaN values remaining!"
    log.info("Imputation complete. Zero NaN remaining.")

    return df
