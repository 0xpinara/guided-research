"""Trading-cost helpers for long/flat and long/short strategies."""

from __future__ import annotations

import numpy as np


def effective_spread_cost(
    position_change: np.ndarray,
    quoted_spread: np.ndarray | float | None,
    effective_spread_fraction: float = 0.15,
    fallback_spread_bps: float = 5.0,
) -> np.ndarray:
    """Cost from trading through a fraction of the quoted spread.

    Parameters
    ----------
    position_change:
        Absolute change in portfolio weight for each name.
    quoted_spread:
        Relative quoted spread, e.g. (ask - bid) / mid.  If unavailable,
        a fallback stock-trading spread in basis points is used.
    effective_spread_fraction:
        Bali-style effective-spread assumption.  A value of 0.15 means the
        strategy pays 15% of the quoted spread on traded notional.
    fallback_spread_bps:
        Used when quoted spreads are absent or invalid.
    """
    delta = np.asarray(position_change, dtype=np.float64)
    if quoted_spread is None:
        spread = np.full_like(delta, fallback_spread_bps / 10_000)
    else:
        spread = np.asarray(quoted_spread, dtype=np.float64)
        if spread.ndim == 0:
            spread = np.full_like(delta, float(spread))
        spread = np.nan_to_num(spread, nan=fallback_spread_bps / 10_000,
                               posinf=fallback_spread_bps / 10_000,
                               neginf=fallback_spread_bps / 10_000)
        spread = np.clip(spread, fallback_spread_bps / 10_000, 1.0)
    return np.abs(delta) * spread * effective_spread_fraction


def annual_short_fee_cost(short_exposure: float, annual_fee_bps: float = 50.0,
                          periods_per_year: float = 252.0) -> float:
    """Borrow-fee drag for the short leg, charged per rebalance period."""
    return abs(float(short_exposure)) * (annual_fee_bps / 10_000) / periods_per_year
