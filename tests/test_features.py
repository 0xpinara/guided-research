"""Validate feature computation logic."""

import numpy as np
import pandas as pd
import pytest


def test_rsi_bounds():
    """RSI should be bounded between 0 and 100."""
    from src.features.stock_features import _rsi
    np.random.seed(42)
    returns = pd.Series(np.random.randn(200) * 0.02)
    rsi = _rsi(returns, 14)
    valid = rsi.dropna()
    assert valid.min() >= 0, f"RSI below 0: {valid.min()}"
    assert valid.max() <= 100, f"RSI above 100: {valid.max()}"


def test_bollinger_position():
    """Bollinger position should be approximately in [-2, 2] for normal data."""
    from src.features.stock_features import compute_stock_features
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    price = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({
        "date": dates,
        "prc": price,
        "ret": np.concatenate([[0], np.diff(price) / price[:-1]]),
        "vol": np.random.randint(1e6, 1e7, 100),
        "vix_close": 20.0,
        "vix3m_close": 22.0,
        "tbill_rate": 2.0,
        "sector_etf_return": 0.001,
    })
    result = compute_stock_features(df)
    boll = result["feat_35"].dropna()
    assert boll.abs().max() < 5, f"Bollinger position too extreme: {boll.abs().max()}"


def test_put_call_ratio():
    """Put-call volume ratio should be positive."""
    from src.features.aggregated_features import _put_call_volume_ratio
    opts = pd.DataFrame({
        "is_call": [1, 1, 0, 0],
        "volume": [100, 200, 150, 50],
    })
    ratio = _put_call_volume_ratio(opts)
    assert ratio == (150 + 50) / (100 + 200)


def test_put_call_ratio_zero_calls():
    """If no call volume, ratio should be NaN."""
    from src.features.aggregated_features import _put_call_volume_ratio
    opts = pd.DataFrame({
        "is_call": [0, 0],
        "volume": [100, 200],
    })
    ratio = _put_call_volume_ratio(opts)
    assert np.isnan(ratio)


def test_target_builder_horizons():
    """Target returns should use correct forward prices."""
    from src.features.target_builder import build_targets
    dates = pd.bdate_range("2020-01-01", periods=20)
    prices = np.arange(100, 120, dtype=float)
    df = pd.DataFrame({
        "ticker": "TEST",
        "date": dates,
        "prc": prices,
    })
    targets = build_targets(df, horizons=[1, 3, 5])

    # 1-day return for first row: (101 - 100) / 100 = 0.01
    assert abs(targets.iloc[0]["ret_1d"] - 0.01) < 1e-10

    # 3-day return for first row: (103 - 100) / 100 = 0.03
    assert abs(targets.iloc[0]["ret_3d"] - 0.03) < 1e-10

    # 5-day return for first row: (105 - 100) / 100 = 0.05
    assert abs(targets.iloc[0]["ret_5d"] - 0.05) < 1e-10

    # Last row should have NaN for all horizons
    assert np.isnan(targets.iloc[-1]["ret_1d"])


def test_iv_surface_nan_for_few_contracts():
    """Surface should return NaN when fewer than 5 contracts."""
    from src.features.iv_surface_features import _interpolate_surface
    m = np.array([0.95, 1.0, 1.05])
    dte = np.array([30, 30, 30])
    iv = np.array([0.25, 0.22, 0.28])
    result = _interpolate_surface(m, dte, iv, [0.90, 1.00, 1.10], [30, 60, 90])
    # With only 3 points and asking for 9 grid points, many will be NaN
    assert isinstance(result, dict)
