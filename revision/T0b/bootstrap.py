"""Task 0b.7 — moving-block bootstrap for daily statistics with overlapping outcomes.

Replaces the BIR "date-block" bootstrap (return_extras.bootstrap_diff), which resampled
single days independently and resampled the two buckets independently. That is not a block
bootstrap and understates uncertainty when 3- and 5-day outcomes overlap.

Procedure (pre-specified; used unchanged in Task 4):
  * the unit is the calendar of trading dates of the cell, in order;
  * blocks of BLOCK_LENGTH = 21 consecutive trading dates (>= the longest horizon, h = 5;
    about one month), start points drawn uniformly with replacement (moving-block bootstrap,
    Kunsch 1989), concatenated and truncated to the original number of dates;
  * every daily series of the cell (e.g. the high and the low bucket, or IC_D and IC_A) is
    indexed by the same resampled dates, so same-day dependence between series is kept;
  * dates on which a series is undefined are skipped in that series' mean;
  * N_BOOT = 10,000 replications, seed 20261001; percentile 95% intervals; two-sided p-value
    2 * min(P(stat* <= 0), P(stat* >= 0)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_LENGTH = 21
N_BOOT = 10_000
SEED = 20261001


def block_indices(n: int, rng: np.random.Generator, block: int = BLOCK_LENGTH) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=nb)
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def mbb(series: dict[str, pd.Series], dates: np.ndarray, stat, block: int = BLOCK_LENGTH,
        n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Moving-block bootstrap of stat(means) where means = {name: mean of series over drawn dates}.

    series: daily values indexed by date (missing dates allowed); dates: the cell's calendar.
    stat: function of the dict of means returning a float (e.g. lambda m: m["high"] - m["low"]).
    """
    dates = np.sort(np.asarray(dates))
    arr = {k: s.reindex(dates).to_numpy(float) for k, s in series.items()}
    point = stat({k: np.nanmean(v) for k, v in arr.items()})
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = block_indices(len(dates), rng, block)
        draws[b] = stat({k: np.nanmean(v[idx]) for k, v in arr.items()})
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    p = 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))
    return {"estimate": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p, 1.0)),
            "block_length": block, "n_boot": n_boot, "seed": seed}


def _self_test() -> None:
    """AR(1) daily series: the block bootstrap SE should exceed the i.i.d. SE."""
    rng = np.random.default_rng(0)
    n, rho = 2000, 0.6
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    d = pd.date_range("2016-01-01", periods=n, freq="B")
    s = pd.Series(x, index=d)
    r = mbb({"x": s}, d.values, lambda m: m["x"], n_boot=2000)
    iid = np.std([rng.choice(x, n).mean() for _ in range(2000)])
    print(f"self-test AR(1) rho={rho}: block CI width {r['ci_hi'] - r['ci_lo']:.4f} vs i.i.d. width "
          f"{2 * 1.96 * iid:.4f} (theory ratio ~ sqrt((1+rho)/(1-rho)) = {np.sqrt((1 + rho) / (1 - rho)):.2f})")


if __name__ == "__main__":
    _self_test()
