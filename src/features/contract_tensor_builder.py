"""Resolution 3: Build contract tensors for Set Transformer input.

A liquid ticker like AAPL can have 5,000-10,000 contracts on a single day.
Naively taking the top-N by volume is biased toward ATM near-term contracts
and wastes capacity on near-duplicates.

This module implements three strategies:

1. **Stratified sampling** (default): Divide the (moneyness x DTE) space into
   buckets.  Sample contracts proportionally from each bucket, ensuring coverage
   across the full options surface.  Fills the tensor budget with a
   representative cross-section.

2. **Chain-level aggregation**: Instead of feeding individual contracts, group
   by expiration chain and aggregate each chain into a summary vector. This
   reduces 5,000 contracts to ~15 chain summaries, making the set manageable
   without any truncation.

3. **Top-N by volume** (legacy): Simple truncation, kept for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_npz, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

OPTIONS_CLEAN_DIR = INTERIM_DIR / "options_clean"
TENSOR_DIR = FEATURES_DIR / "resolution_3_contracts"

# Per-contract feature vector (10 dims)
CONTRACT_FEATURES = [
    "moneyness", "dte_scaled", "is_call", "impl_volatility",
    "log_volume", "log_oi", "delta", "gamma", "vega", "spread_pct",
]

# Moneyness x DTE stratification grid
MONEYNESS_BINS = [0.80, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.20]
DTE_BINS = [7, 30, 60, 90, 120, 180]

# Chain-level aggregation features (per expiration)
CHAIN_FEATURES = [
    # Metadata (2)
    "dte_scaled", "n_contracts_in_chain",
    # Volume/OI (4)
    "total_volume", "total_oi", "put_call_vol_ratio", "otm_vol_frac",
    # IV structure (5)
    "atm_iv", "iv_25d_put", "iv_25d_call", "iv_skew", "mean_iv",
    # Greeks (3)
    "net_delta", "total_gamma", "total_vega",
    # Microstructure (2)
    "vw_spread", "vw_moneyness",
]


# ---------------------------------------------------------------------------
# Strategy 1: Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample(day_opts: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Sample contracts across the moneyness x DTE surface.

    Allocates budget proportionally to each (moneyness_bin, dte_bin) bucket
    based on the bucket's share of total volume, with a minimum of 1 contract
    per non-empty bucket to ensure surface coverage.

    Parameters
    ----------
    day_opts : DataFrame of one day's contracts with computed features.
    budget : maximum number of contracts to return.

    Returns
    -------
    DataFrame of sampled contracts, at most `budget` rows.
    """
    if len(day_opts) <= budget:
        return day_opts

    # Assign buckets
    day_opts = day_opts.copy()
    day_opts["m_bin"] = pd.cut(
        day_opts["moneyness"], bins=MONEYNESS_BINS,
        labels=False, include_lowest=True,
    )
    day_opts["d_bin"] = pd.cut(
        day_opts["dte"], bins=DTE_BINS,
        labels=False, include_lowest=True,
    )
    # Drop contracts outside all bins
    day_opts = day_opts.dropna(subset=["m_bin", "d_bin"])

    if day_opts.empty:
        return day_opts

    # Count volume per bucket
    bucket_vol = day_opts.groupby(["m_bin", "d_bin"])["volume"].sum()
    total_vol = bucket_vol.sum()
    n_buckets = len(bucket_vol)

    if n_buckets == 0:
        return day_opts.head(budget)

    # Allocate: at least 1 per non-empty bucket, rest proportional to volume
    guaranteed = min(n_buckets, budget)
    remaining = budget - guaranteed

    if total_vol > 0 and remaining > 0:
        proportional = (bucket_vol / total_vol * remaining).astype(int)
    else:
        proportional = pd.Series(0, index=bucket_vol.index)

    allocation = proportional + 1  # +1 for the guaranteed slot
    # Trim if we overallocated
    while allocation.sum() > budget:
        # Remove from the largest bucket
        biggest = allocation.idxmax()
        allocation[biggest] -= 1

    # Sample within each bucket: pick highest-volume contracts
    sampled = []
    for (m_b, d_b), n_take in allocation.items():
        if n_take <= 0:
            continue
        bucket = day_opts[(day_opts["m_bin"] == m_b) & (day_opts["d_bin"] == d_b)]
        bucket = bucket.sort_values("volume", ascending=False)
        sampled.append(bucket.head(int(n_take)))

    if sampled:
        return pd.concat(sampled)
    return day_opts.head(budget)


# ---------------------------------------------------------------------------
# Strategy 2: Chain-level aggregation
# ---------------------------------------------------------------------------

def _aggregate_chain(chain: pd.DataFrame) -> dict:
    """Aggregate a single expiration chain into a summary feature vector.

    Parameters
    ----------
    chain : DataFrame of all contracts for one (date, expiration).

    Returns
    -------
    dict of chain-level feature values.
    """
    calls = chain[chain["is_call"] == 1]
    puts = chain[chain["is_call"] == 0]

    total_vol = chain["volume"].sum()
    total_oi = chain["open_interest"].sum()

    # ATM IV (closest to moneyness=1.0)
    atm_mask = (chain["moneyness"] - 1.0).abs()
    if len(atm_mask) > 0:
        atm_idx = atm_mask.idxmin()
        atm_iv = chain.loc[atm_idx, "impl_volatility"]
    else:
        atm_iv = chain["impl_volatility"].mean()

    # 25-delta IV
    iv_25d_put = np.nan
    if len(puts) > 0:
        p25 = (puts["delta"] + 0.25).abs()
        iv_25d_put = puts.loc[p25.idxmin(), "impl_volatility"]
    iv_25d_call = np.nan
    if len(calls) > 0:
        c25 = (calls["delta"] - 0.25).abs()
        iv_25d_call = calls.loc[c25.idxmin(), "impl_volatility"]

    # OTM fraction
    otm = chain[
        ((chain["is_call"] == 1) & (chain["moneyness"] > 1.05))
        | ((chain["is_call"] == 0) & (chain["moneyness"] < 0.95))
    ]
    otm_vol_frac = otm["volume"].sum() / total_vol if total_vol > 0 else 0

    # Volume-weighted spread and moneyness
    vol_mask = chain["volume"] > 0
    if vol_mask.any():
        vw = chain[vol_mask]
        vw_spread = (vw["volume"] * vw["spread_pct"]).sum() / vw["volume"].sum()
        vw_moneyness = (vw["volume"] * vw["moneyness"]).sum() / vw["volume"].sum()
    else:
        vw_spread = chain["spread_pct"].mean()
        vw_moneyness = chain["moneyness"].mean()

    return {
        "dte_scaled": chain["dte"].iloc[0] / 365.0,
        "n_contracts_in_chain": len(chain),
        "total_volume": np.log1p(total_vol),
        "total_oi": np.log1p(total_oi),
        "put_call_vol_ratio": (
            puts["volume"].sum() / calls["volume"].sum()
            if calls["volume"].sum() > 0 else np.nan
        ),
        "otm_vol_frac": otm_vol_frac,
        "atm_iv": atm_iv,
        "iv_25d_put": iv_25d_put if not np.isnan(iv_25d_put) else atm_iv,
        "iv_25d_call": iv_25d_call if not np.isnan(iv_25d_call) else atm_iv,
        "iv_skew": (iv_25d_put - iv_25d_call
                     if not np.isnan(iv_25d_put) and not np.isnan(iv_25d_call)
                     else 0.0),
        "mean_iv": chain["impl_volatility"].mean(),
        "net_delta": (chain["volume"] * chain["delta"] * 100).sum(),
        "total_gamma": (chain["open_interest"] * chain["gamma"] * 100).sum(),
        "total_vega": (chain["open_interest"] * chain["vega"] * 100).sum(),
        "vw_spread": vw_spread,
        "vw_moneyness": vw_moneyness,
    }


def _build_chain_tensor(day_opts: pd.DataFrame, max_chains: int = 20) -> tuple:
    """Aggregate day's contracts into per-chain summaries.

    Returns
    -------
    chain_tensor : ndarray (max_chains, n_chain_features)
    chain_mask : ndarray (max_chains,) bool
    n_chains : int (actual number before padding)
    """
    n_features = len(CHAIN_FEATURES)
    tensor = np.zeros((max_chains, n_features), dtype=np.float32)
    mask = np.zeros(max_chains, dtype=bool)

    if day_opts.empty:
        return tensor, mask, 0

    # Group by expiration, sort by DTE
    chains = []
    for exp, chain_df in day_opts.groupby("expiration"):
        agg = _aggregate_chain(chain_df)
        chains.append(agg)

    # Sort by DTE (near-term first)
    chains.sort(key=lambda x: x["dte_scaled"])
    n_chains = min(len(chains), max_chains)

    for i in range(n_chains):
        tensor[i] = [chains[i].get(f, 0.0) for f in CHAIN_FEATURES]
        mask[i] = True

    # Replace any NaN with 0
    tensor = np.nan_to_num(tensor, nan=0.0)

    return tensor, mask, len(chains)


# ---------------------------------------------------------------------------
# Main builder: supports both strategies
# ---------------------------------------------------------------------------

def _prepare_opts(opts: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for tensor building."""
    opts = opts.copy()
    opts["dte_scaled"] = opts["dte"] / 365.0
    opts["log_volume"] = np.log1p(opts["volume"].clip(lower=0))
    opts["log_oi"] = np.log1p(opts["open_interest"].clip(lower=0))
    for col in CONTRACT_FEATURES:
        if col in opts.columns:
            opts[col] = opts[col].fillna(0.0)
    return opts


def build_contract_tensors(
    ticker: str,
    max_contracts: int = 300,
    strategy: str = "stratified",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-contract tensors using stratified sampling.

    Returns
    -------
    dates, tensors (N, max_contracts, 10), masks (N, max_contracts), n_contracts (N,)
    """
    opts_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"
    if not opts_path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    opts = pd.read_parquet(opts_path)
    opts["date"] = pd.to_datetime(opts["date"])
    opts = _prepare_opts(opts)

    n_features = len(CONTRACT_FEATURES)
    dates_list = sorted(opts["date"].unique())
    N = len(dates_list)

    tensors = np.zeros((N, max_contracts, n_features), dtype=np.float32)
    masks = np.zeros((N, max_contracts), dtype=bool)
    n_contracts_arr = np.zeros(N, dtype=np.int32)

    for i, dt in enumerate(dates_list):
        day = opts[opts["date"] == dt]
        n_contracts_arr[i] = len(day)

        if strategy == "stratified":
            sampled = _stratified_sample(day, max_contracts)
        elif strategy == "top_volume":
            sampled = day.sort_values("volume", ascending=False).head(max_contracts)
        else:
            sampled = day.sort_values("volume", ascending=False).head(max_contracts)

        n = min(len(sampled), max_contracts)
        if n > 0:
            # Sort sampled contracts by moneyness for consistent ordering
            sampled = sampled.sort_values(["dte", "moneyness"])
            avail_cols = [c for c in CONTRACT_FEATURES if c in sampled.columns]
            vals = sampled[avail_cols].values[:n].astype(np.float32)
            tensors[i, :n, :len(avail_cols)] = vals
            masks[i, :n] = True

    return np.array(dates_list, dtype="datetime64[ns]"), tensors, masks, n_contracts_arr


def build_chain_tensors(
    ticker: str,
    max_chains: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-chain (expiration) aggregated tensors.

    Returns
    -------
    dates, tensors (N, max_chains, 16), masks (N, max_chains), n_chains (N,)
    """
    opts_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"
    if not opts_path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    opts = pd.read_parquet(opts_path)
    opts["date"] = pd.to_datetime(opts["date"])
    opts["expiration"] = pd.to_datetime(opts["expiration"])
    opts = _prepare_opts(opts)

    n_features = len(CHAIN_FEATURES)
    dates_list = sorted(opts["date"].unique())
    N = len(dates_list)

    tensors = np.zeros((N, max_chains, n_features), dtype=np.float32)
    masks = np.zeros((N, max_chains), dtype=bool)
    n_chains_arr = np.zeros(N, dtype=np.int32)

    for i, dt in enumerate(dates_list):
        day = opts[opts["date"] == dt]
        t, m, nc = _build_chain_tensor(day, max_chains)
        tensors[i] = t
        masks[i] = m
        n_chains_arr[i] = nc

    return np.array(dates_list, dtype="datetime64[ns]"), tensors, masks, n_chains_arr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(cfg=None):
    """Build contract tensors for all tickers (both strategies)."""
    if cfg is None:
        cfg = load_config()

    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    max_c = cfg.contract_tensor.max_contracts

    for ticker in all_tickers(cfg):
        # Strategy 1: Stratified per-contract tensors
        out_path = TENSOR_DIR / f"{ticker}.npz"
        if not out_path.exists():
            dates, tensors, masks, n_contracts = build_contract_tensors(
                ticker, max_contracts=max_c, strategy="stratified",
            )
            if len(dates) > 0:
                save_npz(out_path, dates=dates, tensors=tensors,
                         masks=masks, n_contracts=n_contracts)
                log.info(
                    "%s stratified: %d days, %s, avg %.0f total contracts/day "
                    "(sampled %d max), coverage: %.0f%% of days have >0 contracts",
                    ticker, len(dates), tensors.shape, n_contracts.mean(),
                    max_c, (n_contracts > 0).mean() * 100,
                )

        # Strategy 2: Chain-level aggregation
        chain_path = TENSOR_DIR / f"{ticker}_chains.npz"
        if not chain_path.exists():
            dates, tensors, masks, n_chains = build_chain_tensors(ticker, max_chains=20)
            if len(dates) > 0:
                save_npz(chain_path, dates=dates, tensors=tensors,
                         masks=masks, n_chains=n_chains)
                log.info(
                    "%s chains: %d days, %s, avg %.1f chains/day",
                    ticker, len(dates), tensors.shape, n_chains.mean(),
                )

    log.info("=== Contract Tensors Complete ===")


if __name__ == "__main__":
    run()
