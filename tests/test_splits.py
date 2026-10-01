"""Verify temporal ordering and split integrity."""

import numpy as np
import pandas as pd
import pytest


def test_all_rows_assigned():
    """Every row should get a split assignment."""
    from src.preprocessing.split import temporal_split

    dates = pd.bdate_range("2020-01-01", periods=300)
    df = pd.DataFrame({
        "ticker": np.repeat(["AAPL", "MSFT"], 150),
        "date": np.tile(dates[:150], 2),
    })
    splits = temporal_split(df, 0.6, 0.2, 0.2)
    assert splits["split"].isin(["train", "val", "test"]).all()


def test_multi_ticker_same_split_dates():
    """All tickers on the same date should be in the same split."""
    from src.preprocessing.split import temporal_split

    dates = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame({
        "ticker": np.repeat(["AAPL", "MSFT", "GOOG"], 100),
        "date": np.tile(dates, 3),
    })
    splits = temporal_split(df, 0.6, 0.2, 0.2)

    # Check that each date has only one split
    date_splits = splits.groupby("date")["split"].nunique()
    assert (date_splits == 1).all(), "Some dates have multiple split assignments!"


def test_split_fractions_approximate():
    """Split sizes should roughly match requested fractions."""
    from src.preprocessing.split import temporal_split

    dates = pd.bdate_range("2020-01-01", periods=500)
    df = pd.DataFrame({"ticker": "TEST", "date": dates})
    splits = temporal_split(df, 0.6, 0.2, 0.2)

    n = len(splits)
    train_frac = (splits["split"] == "train").sum() / n
    val_frac = (splits["split"] == "val").sum() / n
    test_frac = (splits["split"] == "test").sum() / n

    assert abs(train_frac - 0.6) < 0.05, f"Train fraction {train_frac} too far from 0.6"
    assert abs(val_frac - 0.2) < 0.05, f"Val fraction {val_frac} too far from 0.2"
    assert abs(test_frac - 0.2) < 0.05, f"Test fraction {test_frac} too far from 0.2"
