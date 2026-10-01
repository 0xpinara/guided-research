"""Verify no future information leaks into features or splits."""

import numpy as np
import pandas as pd
import pytest


def test_temporal_split_no_overlap():
    """Train, val, and test date ranges must not overlap."""
    from src.preprocessing.split import temporal_split

    dates = pd.bdate_range("2020-01-01", periods=500)
    df = pd.DataFrame({
        "ticker": "TEST",
        "date": dates,
    })
    splits = temporal_split(df, 0.6, 0.2, 0.2)

    train_dates = splits[splits["split"] == "train"]["date"]
    val_dates = splits[splits["split"] == "val"]["date"]
    test_dates = splits[splits["split"] == "test"]["date"]

    if len(val_dates) > 0:
        assert train_dates.max() < val_dates.min(), "Train/val dates overlap!"
    if len(test_dates) > 0 and len(val_dates) > 0:
        assert val_dates.max() < test_dates.min(), "Val/test dates overlap!"


def test_target_gap():
    """Last training date + max horizon should not reach validation."""
    from src.preprocessing.split import temporal_split

    dates = pd.bdate_range("2020-01-01", periods=500)
    df = pd.DataFrame({
        "ticker": "TEST",
        "date": dates,
    })
    splits = temporal_split(df, 0.6, 0.2, 0.2, max_target_horizon=5)

    train_dates = sorted(splits[splits["split"] == "train"]["date"].unique())
    val_dates = sorted(splits[splits["split"] == "val"]["date"].unique())

    if len(val_dates) > 0:
        last_train = train_dates[-1]
        first_val = val_dates[0]
        # There should be a gap of at least 5 business days
        gap = np.busday_count(
            last_train.date(),
            first_val.date(),
        )
        assert gap >= 5, f"Train-to-val gap ({gap}d) < max horizon (5d)"


def test_features_use_only_past_data():
    """Stock features should not use future data."""
    from src.features.stock_features import compute_stock_features

    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    price = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({
        "date": dates,
        "prc": price,
        "ret": np.concatenate([[0], np.diff(price) / price[:-1]]),
        "vol": 1e6,
        "vix_close": 20.0,
        "vix3m_close": 22.0,
        "tbill_rate": 2.0,
        "sector_etf_return": 0.001,
    })

    result = compute_stock_features(df)

    # SMA50 at day 50 should only use days 0-50, not beyond
    sma50_at_60 = result.iloc[59]["feat_36"]
    # Recompute manually
    manual_sma50 = price[10:60].mean()
    manual_dist = (price[59] - manual_sma50) / manual_sma50
    assert abs(sma50_at_60 - manual_dist) < 1e-6, (
        f"SMA50 distance mismatch: got {sma50_at_60}, expected {manual_dist}"
    )
