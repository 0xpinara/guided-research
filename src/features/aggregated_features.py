"""Resolution 1: Aggregated scalar options features (feat_01 through feat_26)."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, load_parquet, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

OPTIONS_CLEAN_DIR = INTERIM_DIR / "options_clean"


# -----------------------------------------------------------------------
# Per-day feature computation helpers
# -----------------------------------------------------------------------

def _put_call_volume_ratio(opts: pd.DataFrame) -> float:
    """feat_01: total put volume / total call volume."""
    call_vol = opts.loc[opts["is_call"] == 1, "volume"].sum()
    put_vol = opts.loc[opts["is_call"] == 0, "volume"].sum()
    return put_vol / call_vol if call_vol > 0 else np.nan


def _put_call_oi_ratio(opts: pd.DataFrame) -> float:
    """feat_02: total put OI / total call OI."""
    call_oi = opts.loc[opts["is_call"] == 1, "open_interest"].sum()
    put_oi = opts.loc[opts["is_call"] == 0, "open_interest"].sum()
    return put_oi / call_oi if call_oi > 0 else np.nan


def _options_to_stock_volume_ratio(opts: pd.DataFrame, stock_vol: float) -> float:
    """feat_03: total options volume * 100 / stock share volume."""
    total_opt_vol = opts["volume"].sum() * 100
    return total_opt_vol / stock_vol if stock_vol > 0 else np.nan


def _net_delta_exposure(opts: pd.DataFrame, stock_vol: float) -> float:
    """feat_04: sum(volume * delta * 100) / stock_volume."""
    nde = (opts["volume"] * opts["delta"] * 100).sum()
    return nde / stock_vol if stock_vol > 0 else np.nan


def _volume_weighted_avg_iv(opts: pd.DataFrame) -> float:
    """feat_05: volume-weighted average IV."""
    mask = opts["volume"] > 0
    if not mask.any():
        return np.nan
    sub = opts[mask]
    return (sub["volume"] * sub["impl_volatility"]).sum() / sub["volume"].sum()


def _atm_iv(opts: pd.DataFrame) -> float:
    """feat_06: ATM implied volatility (moneyness 0.97-1.03, closest to 30 DTE)."""
    atm = opts[(opts["moneyness"] >= 0.97) & (opts["moneyness"] <= 1.03)]
    if len(atm) == 0:
        # Fallback: wider range, volume-weighted
        atm = opts[(opts["moneyness"] >= 0.95) & (opts["moneyness"] <= 1.05)]
        if len(atm) == 0:
            return np.nan
        mask = atm["volume"] > 0
        if mask.any():
            atm_v = atm[mask]
            return (atm_v["volume"] * atm_v["impl_volatility"]).sum() / atm_v["volume"].sum()
        return atm["impl_volatility"].mean()

    # Pick contracts closest to 30 DTE
    atm = atm.copy()
    atm["dte_dist"] = (atm["dte"] - 30).abs()
    best_dte = atm["dte_dist"].min()
    near = atm[atm["dte_dist"] == best_dte]
    return near["impl_volatility"].mean()


def _iv_skew(opts: pd.DataFrame) -> float:
    """feat_08: IV of 25-delta put minus IV of 25-delta call."""
    puts = opts[opts["is_call"] == 0].copy()
    calls = opts[opts["is_call"] == 1].copy()

    if puts.empty or calls.empty:
        return np.nan

    # Find put closest to -0.25 delta
    puts["delta_dist"] = (puts["delta"] + 0.25).abs()
    calls["delta_dist"] = (calls["delta"] - 0.25).abs()

    put_iv = puts.loc[puts["delta_dist"].idxmin(), "impl_volatility"]
    call_iv = calls.loc[calls["delta_dist"].idxmin(), "impl_volatility"]
    return put_iv - call_iv


def _iv_term_structure_slope(opts: pd.DataFrame) -> float:
    """feat_09: ATM IV near 30d minus ATM IV near 90d."""
    atm = opts[(opts["moneyness"] >= 0.97) & (opts["moneyness"] <= 1.03)]
    if len(atm) < 2:
        return np.nan

    atm = atm.copy()
    # Near 30d
    atm["dist_30"] = (atm["dte"] - 30).abs()
    atm["dist_90"] = (atm["dte"] - 90).abs()

    near_30 = atm.loc[atm["dist_30"].idxmin(), "impl_volatility"]
    near_90 = atm.loc[atm["dist_90"].idxmin(), "impl_volatility"]

    # Only meaningful if we actually have different expirations
    dte_30 = atm.loc[atm["dist_30"].idxmin(), "dte"]
    dte_90 = atm.loc[atm["dist_90"].idxmin(), "dte"]
    if abs(dte_30 - dte_90) < 10:
        return np.nan

    return near_30 - near_90


def _otm_volume_concentration(opts: pd.DataFrame) -> float:
    """feat_15: volume of OTM contracts / total volume."""
    total = opts["volume"].sum()
    if total == 0:
        return np.nan
    otm_mask = (
        ((opts["is_call"] == 1) & (opts["moneyness"] > 1.05))
        | ((opts["is_call"] == 0) & (opts["moneyness"] < 0.95))
    )
    return opts.loc[otm_mask, "volume"].sum() / total


def _call_volume_skew(opts: pd.DataFrame) -> float:
    """feat_16: OTM call volume / ATM call volume."""
    calls = opts[opts["is_call"] == 1]
    atm_vol = calls[(calls["moneyness"] >= 0.97) & (calls["moneyness"] <= 1.03)]["volume"].sum()
    otm_vol = calls[calls["moneyness"] > 1.05]["volume"].sum()
    return otm_vol / atm_vol if atm_vol > 0 else np.nan


def _put_volume_skew(opts: pd.DataFrame) -> float:
    """feat_17: OTM put volume / ATM put volume."""
    puts = opts[opts["is_call"] == 0]
    atm_vol = puts[(puts["moneyness"] >= 0.97) & (puts["moneyness"] <= 1.03)]["volume"].sum()
    otm_vol = puts[puts["moneyness"] < 0.95]["volume"].sum()
    return otm_vol / atm_vol if atm_vol > 0 else np.nan


def _vw_avg_spread(opts: pd.DataFrame) -> float:
    """feat_18: volume-weighted average spread_pct."""
    mask = (opts["volume"] > 0) & opts["spread_pct"].notna()
    if not mask.any():
        return np.nan
    sub = opts[mask]
    return (sub["volume"] * sub["spread_pct"]).sum() / sub["volume"].sum()


def _vw_moneyness(opts: pd.DataFrame) -> float:
    """feat_20: volume-weighted average moneyness."""
    mask = opts["volume"] > 0
    if not mask.any():
        return np.nan
    sub = opts[mask]
    return (sub["volume"] * sub["moneyness"]).sum() / sub["volume"].sum()


def _cw_parity_deviation(opts: pd.DataFrame) -> float:
    """feat_49: Cremers-Weinbaum (2010) put-call parity deviation.

    For each strike-expiry where both a call and put exist, compute
    σ_call − σ_put. Return the volume-weighted average across ATM strikes.
    Divergence from zero signals informed directional pressure that
    violates put-call parity.
    """
    # ATM-ish range where IV estimates are reliable
    atm = opts[(opts["moneyness"] >= 0.95) & (opts["moneyness"] <= 1.05)].copy()
    if atm.empty:
        return np.nan

    calls = atm[atm["is_call"] == 1].copy()
    puts = atm[atm["is_call"] == 0].copy()
    if calls.empty or puts.empty:
        return np.nan

    # Match on (strike, dte) — same strike and expiration
    calls = calls.set_index(["strike", "dte"])
    puts = puts.set_index(["strike", "dte"])
    common = calls.index.intersection(puts.index)

    if len(common) == 0:
        return np.nan

    c = calls.loc[common]
    p = puts.loc[common]

    iv_diff = c["impl_volatility"].values - p["impl_volatility"].values
    weights = (c["volume"].values + p["volume"].values).astype(float)
    total_w = weights.sum()
    if total_w <= 0:
        return np.mean(iv_diff)  # equal-weight fallback

    return np.sum(iv_diff * weights) / total_w


def _oi_decomposed_pc_ratio(opts: pd.DataFrame, prev_oi_by_contract: dict | None) -> tuple[float, dict]:
    """feat_50: Opening-trade P/C ratio (Pan & Poteshman 2006 proxy).

    If OI increased for a contract, today's volume likely represents new
    position openings. We sum "opening volume" for puts vs calls and
    compute the ratio. This isolates informed directional bets from
    routine closing trades.

    Returns (ratio, updated_oi_dict).
    """
    # Build current OI dict (vectorized)
    keys = list(zip(opts["strike"].values, opts["dte"].values, opts["is_call"].astype(int).values))
    oi_vals = opts["open_interest"].values
    current_oi = dict(zip(keys, oi_vals))

    if prev_oi_by_contract is None:
        return np.nan, current_oi

    # Vectorized: look up previous OI for each contract
    prev_oi = np.array([prev_oi_by_contract.get(k, 0) for k in keys])
    opening_mask = oi_vals > prev_oi  # OI increased → opening trades

    vol = opts["volume"].values
    is_call = opts["is_call"].values == 1

    open_call_vol = vol[opening_mask & is_call].sum()
    open_put_vol = vol[opening_mask & ~is_call].sum()

    ratio = open_put_vol / open_call_vol if open_call_vol > 0 else np.nan
    return ratio, current_oi


def _implied_earnings_move(opts: pd.DataFrame, days_to_earnings: float) -> float:
    """feat_51: Implied earnings move magnitude.

    ATM straddle price / stock price for the expiry nearest to the
    earnings date. This extracts the market's priced-in expected move
    around the announcement. Only meaningful within ~30 days of earnings.
    """
    if np.isnan(days_to_earnings) or days_to_earnings > 30 or days_to_earnings < 0:
        return np.nan

    spot = opts["underlying_price"].iloc[0]
    if spot <= 0 or np.isnan(spot):
        return np.nan

    # Find expiry chain closest to earnings date
    target_dte = max(days_to_earnings, 1)  # at least 1 DTE
    atm = opts[(opts["moneyness"] >= 0.97) & (opts["moneyness"] <= 1.03)].copy()
    if atm.empty:
        return np.nan

    # Pick chain nearest to earnings DTE
    atm["dte_dist"] = (atm["dte"] - target_dte).abs()
    best_dte = atm["dte_dist"].min()
    chain = atm[atm["dte_dist"] == best_dte]

    # Get ATM call and put mid prices
    call_mid = chain.loc[chain["is_call"] == 1, "mid_price"]
    put_mid = chain.loc[chain["is_call"] == 0, "mid_price"]

    if call_mid.empty or put_mid.empty:
        # Use available side only — approximate with 2x
        if not call_mid.empty:
            straddle = call_mid.mean() * 2
        elif not put_mid.empty:
            straddle = put_mid.mean() * 2
        else:
            return np.nan
    else:
        straddle = call_mid.mean() + put_mid.mean()

    return straddle / spot  # implied move as fraction of stock price


def _gex_estimate(opts: pd.DataFrame) -> float:
    """feat_21: estimated gamma exposure (GEX)."""
    sign = np.where(opts["is_call"] == 1, 1.0, -1.0)
    gex = (opts["open_interest"] * opts["gamma"] * 100
           * opts["underlying_price"] * sign).sum()
    return gex


def _bkm_contracts(opts: pd.DataFrame, rf: float) -> tuple[float, float, float]:
    """Compute Bakshi-Kapadia-Madan (2003) V, W, X contracts.

    Implements the exact BKM model-free moments using OTM options:
      V = e^{rτ} ∫ (2(1 - ln(K/F))) / K² × O(K) dK      (variance)
      W = e^{rτ} ∫ (6ln(K/F) - 3ln²(K/F)) / K² × O(K) dK (cubic)
      X = e^{rτ} ∫ (12ln²(K/F) + 4ln³(K/F)) / K² × O(K) dK (quartic)

    Reference: Bakshi, Kapadia, Madan (2003), "Stock Return Characteristics,
    Skew Laws, and the Differential Pricing of Individual Equity Options",
    Review of Financial Studies 16(1), pp. 101-143.
    """
    if opts.empty:
        return np.nan, np.nan, np.nan

    best_dte = opts.iloc[(opts["dte"] - 30).abs().argsort()[:1]]["dte"].iloc[0]
    sub = opts[(opts["dte"] >= best_dte - 5) & (opts["dte"] <= best_dte + 5)]
    if len(sub) < 5:
        return np.nan, np.nan, np.nan

    tau = best_dte / 365.0
    S = sub["underlying_price"].iloc[0]
    fwd = S * np.exp(rf * tau)
    disc = np.exp(rf * tau)

    # Separate OTM: puts below forward, calls above
    otm_puts = sub[(sub["is_call"] == 0) & (sub["strike"] < fwd)].sort_values("strike")
    otm_calls = sub[(sub["is_call"] == 1) & (sub["strike"] > fwd)].sort_values("strike")
    otm = pd.concat([otm_puts, otm_calls])

    if len(otm) < 5:
        return np.nan, np.nan, np.nan

    K = otm["strike"].values
    O = otm["mid_price"].values  # OTM option prices

    # Trapezoidal strike spacing
    dk = np.zeros(len(K))
    dk[0] = K[1] - K[0]
    dk[-1] = K[-1] - K[-2]
    for i in range(1, len(K) - 1):
        dk[i] = (K[i + 1] - K[i - 1]) / 2

    log_kf = np.log(K / fwd)

    # BKM integrands (exact formulas from the paper)
    v_integrand = (2 * (1 - log_kf)) / K ** 2 * O * dk
    w_integrand = (6 * log_kf - 3 * log_kf ** 2) / K ** 2 * O * dk
    x_integrand = (12 * log_kf ** 2 + 4 * log_kf ** 3) / K ** 2 * O * dk

    V = disc * np.sum(v_integrand)
    W = disc * np.sum(w_integrand)
    X = disc * np.sum(x_integrand)

    return V, W, X


def _implied_skewness_bkm(opts: pd.DataFrame, rf: float) -> float:
    """feat_25: BKM implied skewness.

    SKEW = (e^{rτ}W - 3μ·e^{rτ}V + 2μ³) / (e^{rτ}V - μ²)^{3/2}
    where μ = e^{rτ} - 1 - e^{rτ}V/2 - e^{rτ}W/6 - e^{rτ}X/24
    """
    V, W, X = _bkm_contracts(opts, rf)
    if np.isnan(V) or V <= 0:
        return np.nan

    best_dte = opts.iloc[(opts["dte"] - 30).abs().argsort()[:1]]["dte"].iloc[0]
    tau = best_dte / 365.0
    disc = np.exp(rf * tau)

    # Mean excess return
    mu = disc - 1 - disc * V / 2 - disc * W / 6 - disc * X / 24

    # Central moments
    sigma2 = disc * V - mu ** 2
    if sigma2 <= 0:
        return np.nan

    skew = (disc * W - 3 * mu * disc * V + 2 * mu ** 3) / sigma2 ** 1.5
    return skew


def _implied_kurtosis_bkm(opts: pd.DataFrame, rf: float) -> float:
    """feat_26: BKM implied kurtosis.

    KURT = (e^{rτ}X - 4μ·e^{rτ}W + 6μ²·e^{rτ}V - 3μ⁴) / (e^{rτ}V - μ²)²
    """
    V, W, X = _bkm_contracts(opts, rf)
    if np.isnan(V) or V <= 0:
        return np.nan

    best_dte = opts.iloc[(opts["dte"] - 30).abs().argsort()[:1]]["dte"].iloc[0]
    tau = best_dte / 365.0
    disc = np.exp(rf * tau)

    mu = disc - 1 - disc * V / 2 - disc * W / 6 - disc * X / 24

    sigma2 = disc * V - mu ** 2
    if sigma2 <= 0:
        return np.nan

    kurt = (disc * X - 4 * mu * disc * W + 6 * mu ** 2 * disc * V - 3 * mu ** 4) / sigma2 ** 2
    return kurt


# -----------------------------------------------------------------------
# Main per-ticker computation
# -----------------------------------------------------------------------

def compute_aggregated_features(ticker: str, stock_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 26 aggregated options features for one ticker.

    Returns a DataFrame with one row per trading day.
    """
    opts_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"
    if not opts_path.exists():
        log.warning("No cleaned options for %s", ticker)
        return pd.DataFrame()

    opts_all = pd.read_parquet(opts_path)
    opts_all["date"] = pd.to_datetime(opts_all["date"])

    # Prepare stock data
    stk = stock_df[stock_df["ticker"] == ticker].copy()
    stk["date"] = pd.to_datetime(stk["date"])
    stk = stk.sort_values("date").set_index("date")

    # Risk-free rate (needed for BKM)
    rf_series = stk.get("tbill_rate", pd.Series(0.0, index=stk.index))
    rf_series = rf_series.fillna(0.0) / 100.0

    dates = sorted(opts_all["date"].unique())
    rows = []

    # Lookback buffers for time-series features
    atm_iv_hist = {}        # date -> atm_iv
    total_vol_hist = {}     # date -> total options volume
    spread_hist = {}        # date -> vw_avg_spread
    total_oi_hist = {}      # date -> total OI
    gex_hist = {}           # date -> GEX (for charm proxy)
    prev_oi_by_contract = None  # per-contract OI for opening-trade P/C ratio

    for dt in dates:
        day_opts = opts_all[opts_all["date"] == dt]
        stock_vol = stk.loc[dt, "vol"] if dt in stk.index else 0
        mktcap = stk.loc[dt, "mktcap"] if dt in stk.index else np.nan
        rf = rf_series.get(dt, 0.0) if dt in rf_series.index else 0.0

        row = {"ticker": ticker, "date": dt}

        # Category 1: Options Sentiment
        row["feat_01"] = _put_call_volume_ratio(day_opts)
        row["feat_02"] = _put_call_oi_ratio(day_opts)
        row["feat_03"] = _options_to_stock_volume_ratio(day_opts, stock_vol)
        row["feat_04"] = _net_delta_exposure(day_opts, stock_vol)
        row["feat_05"] = _volume_weighted_avg_iv(day_opts)

        # Category 2: IV Signals
        current_atm_iv = _atm_iv(day_opts)
        row["feat_06"] = current_atm_iv
        atm_iv_hist[dt] = current_atm_iv

        # IV rank (feat_07) - needs 252 days of history
        hist_ivs = [atm_iv_hist[d] for d in sorted(atm_iv_hist.keys())
                     if d <= dt and not np.isnan(atm_iv_hist.get(d, np.nan))]
        if len(hist_ivs) >= 252:
            recent_252 = hist_ivs[-252:]
            iv_min, iv_max = min(recent_252), max(recent_252)
            row["feat_07"] = ((current_atm_iv - iv_min) / (iv_max - iv_min)
                              if iv_max > iv_min else np.nan)
        else:
            row["feat_07"] = np.nan

        row["feat_08"] = _iv_skew(day_opts)
        row["feat_09"] = _iv_term_structure_slope(day_opts)

        # IV changes (feat_10, 11) - need prior atm_iv
        sorted_dates = sorted(d for d in atm_iv_hist.keys() if d < dt)
        if len(sorted_dates) >= 1:
            prev_iv = atm_iv_hist[sorted_dates[-1]]
            row["feat_10"] = current_atm_iv - prev_iv if not np.isnan(prev_iv) else np.nan
        else:
            row["feat_10"] = np.nan

        if len(sorted_dates) >= 5:
            prev5_iv = atm_iv_hist[sorted_dates[-5]]
            row["feat_11"] = current_atm_iv - prev5_iv if not np.isnan(prev5_iv) else np.nan
        else:
            row["feat_11"] = np.nan

        # IV minus realized vol (feat_12)
        hvol_30d = np.nan
        if dt in stk.index:
            # Look back from stock features if available
            idx = stk.index.get_loc(dt)
            if idx >= 29:
                window_rets = stk.iloc[idx - 29: idx + 1]["ret"]
                hvol_30d = window_rets.std() * np.sqrt(252)
        row["feat_12"] = (current_atm_iv - hvol_30d
                          if not np.isnan(current_atm_iv) and not np.isnan(hvol_30d)
                          else np.nan)

        # Category 3: Unusual Activity
        total_vol = day_opts["volume"].sum()
        total_vol_hist[dt] = total_vol

        prior_vols = [total_vol_hist[d] for d in sorted_dates[-20:]
                      if d in total_vol_hist]
        if len(prior_vols) >= 5:
            row["feat_13"] = total_vol / np.mean(prior_vols) if np.mean(prior_vols) > 0 else np.nan
        else:
            row["feat_13"] = np.nan

        total_oi = day_opts["open_interest"].sum()
        total_oi_hist[dt] = total_oi
        if len(sorted_dates) >= 1 and sorted_dates[-1] in total_oi_hist:
            prev_oi = total_oi_hist[sorted_dates[-1]]
            row["feat_14"] = (total_oi - prev_oi) / prev_oi if prev_oi > 0 else np.nan
        else:
            row["feat_14"] = np.nan

        row["feat_15"] = _otm_volume_concentration(day_opts)
        row["feat_16"] = _call_volume_skew(day_opts)
        row["feat_17"] = _put_volume_skew(day_opts)

        # Category 4: Microstructure
        current_spread = _vw_avg_spread(day_opts)
        row["feat_18"] = current_spread
        spread_hist[dt] = current_spread

        if len(sorted_dates) >= 1 and sorted_dates[-1] in spread_hist:
            prev_spread = spread_hist[sorted_dates[-1]]
            if not np.isnan(prev_spread) and not np.isnan(current_spread):
                row["feat_19"] = current_spread - prev_spread
            else:
                row["feat_19"] = np.nan
        else:
            row["feat_19"] = np.nan

        row["feat_20"] = _vw_moneyness(day_opts)

        # Category 5: Dealer Positioning
        gex = _gex_estimate(day_opts)
        row["feat_21"] = gex
        row["feat_22"] = gex / mktcap * 100 if not np.isnan(mktcap) and mktcap > 0 else np.nan

        # Charm proxy (feat_23): daily change in GEX, normalized by market cap.
        # True charm = dDelta/dTime across all contracts, but that requires
        # matching individual contracts across days. ΔGEX/mktcap captures the
        # same directional effect: how dealer gamma-hedge pressure shifted.
        gex_hist[dt] = gex
        if len(sorted_dates) >= 1 and sorted_dates[-1] in gex_hist:
            prev_gex = gex_hist[sorted_dates[-1]]
            delta_gex = gex - prev_gex
            row["feat_23"] = delta_gex / mktcap * 100 if not np.isnan(mktcap) and mktcap > 0 else np.nan
        else:
            row["feat_23"] = np.nan

        # Zero DTE volume share (feat_24) - already computed in clean_options
        if "zero_dte_volume_share" in day_opts.columns:
            row["feat_24"] = day_opts["zero_dte_volume_share"].iloc[0]
        else:
            row["feat_24"] = np.nan

        # Category 6: Higher Moments
        row["feat_25"] = _implied_skewness_bkm(day_opts, rf)
        row["feat_26"] = _implied_kurtosis_bkm(day_opts, rf)

        # Category 7: Literature-derived features
        # feat_49: Cremers-Weinbaum put-call parity deviation
        row["feat_49"] = _cw_parity_deviation(day_opts)

        # feat_50: OI-decomposed opening-trade P/C ratio
        ratio_50, prev_oi_by_contract = _oi_decomposed_pc_ratio(day_opts, prev_oi_by_contract)
        row["feat_50"] = ratio_50

        # feat_51: Implied earnings move (needs days_to_earnings from stock data)
        days_to_earn = np.nan
        if dt in stk.index and "days_to_next_earnings" in stk.columns:
            days_to_earn = stk.loc[dt, "days_to_next_earnings"]
        row["feat_51"] = _implied_earnings_move(day_opts, days_to_earn)

        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Computed %d aggregated features for %s: %d days", 29, ticker, len(df))
    return df


def run(cfg=None):
    """Compute aggregated features for all tickers."""
    if cfg is None:
        cfg = load_config()

    stock_path = INTERIM_DIR / "stock_clean" / "stock_daily.parquet"
    if not stock_path.exists():
        log.error("Clean stock data not found")
        return

    stock = pd.read_parquet(stock_path)
    stock["date"] = pd.to_datetime(stock["date"])

    from src.utils.config import all_tickers
    results = []
    for ticker in all_tickers(cfg):
        out_path = FEATURES_DIR / "resolution_1_scalar" / f"{ticker}_options.parquet"
        if out_path.exists():
            log.info("Skipping %s (already computed)", ticker)
            results.append(pd.read_parquet(out_path))
            continue

        df = compute_aggregated_features(ticker, stock)
        if not df.empty:
            save_parquet(df, out_path)
            results.append(df)

    if results:
        combined = pd.concat(results, ignore_index=True)
        save_parquet(combined, FEATURES_DIR / "resolution_1_scalar" / "options_features_all.parquet")

    log.info("=== Aggregated Options Features Complete ===")


if __name__ == "__main__":
    run()
