"""Parametric volatility surface features using SVI and delta-tau grids.

This implements the approach used by quant firms and modern options research:

1. **SVI (Stochastic Volatility Inspired) parameterization**: Fit Gatheral's
   SVI model to each expiration's IV smile, producing 5 interpretable
   parameters per chain.  These capture level, slope, curvature, and
   asymmetry of the smile in a compact, noise-robust form.

2. **Standardized delta-tau grid**: Evaluate IV at fixed (delta, tau) points
   using the fitted surface.  Unlike moneyness-based grids, delta-based
   grids are comparable across tickers with different vol levels.

3. **Surface dynamics**: Day-over-day changes in SVI parameters and grid
   values, which often carry more signal than levels.

References:
  - Gatheral (2004): "A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility
    derivatives"
  - Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning"
  - Horvath, Muguruza, Tomas (2021): deep hedging with surface inputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

OPTIONS_CLEAN_DIR = INTERIM_DIR / "options_clean"


# ---------------------------------------------------------------------------
# SVI Model: w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))
#   where k = log(strike/forward) is log-moneyness
#   w = total implied variance = IV² * τ
#
# 5 parameters:
#   a : overall variance level
#   b : slope magnitude (how fast wings rise)
#   ρ : skew/asymmetry (-1 to 1; negative = put skew)
#   m : horizontal shift of the smile minimum
#   σ : smoothness of the ATM region
# ---------------------------------------------------------------------------

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                        m: float, sigma: float) -> np.ndarray:
    """SVI total implied variance as a function of log-moneyness."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(strikes: np.ndarray, forward: float, tau: float,
            ivs: np.ndarray, volumes: np.ndarray | None = None) -> dict:
    """Fit SVI to one expiration's IV smile.

    Parameters
    ----------
    strikes : array of strike prices
    forward : forward price F = S * exp(r * tau)
    tau : time to expiry in years
    ivs : implied volatilities
    volumes : optional weights (volume-based)

    Returns
    -------
    dict with keys: a, b, rho, m, sigma, fit_error, n_points
    """
    if len(strikes) < 5:
        return {"a": np.nan, "b": np.nan, "rho": np.nan,
                "m": np.nan, "sigma": np.nan, "fit_error": np.nan,
                "n_points": len(strikes)}

    k = np.log(strikes / forward)  # log-moneyness
    w_obs = ivs ** 2 * tau          # total variance

    # Volume weights (optional)
    if volumes is not None and volumes.sum() > 0:
        weights = np.sqrt(volumes / volumes.sum())
    else:
        weights = np.ones(len(k))

    # Initial guess
    atm_var = np.interp(0, k, w_obs) if len(k) > 1 else w_obs.mean()
    x0 = [atm_var, 0.1, -0.3, 0.0, 0.1]

    # Bounds: a > 0, b > 0, -1 < rho < 1, sigma > 0
    bounds = [
        (1e-6, None),     # a
        (1e-6, 5.0),      # b
        (-0.999, 0.999),  # rho
        (-2.0, 2.0),      # m
        (1e-4, 2.0),      # sigma
    ]

    def objective(params):
        a, b, rho, m_, sigma = params
        w_model = svi_total_variance(k, a, b, rho, m_, sigma)
        residuals = (w_model - w_obs) * weights
        return np.sum(residuals ** 2)

    try:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 200})
        a, b, rho, m, sigma = result.x
        fit_error = np.sqrt(result.fun / len(k))
    except Exception:
        return {"a": np.nan, "b": np.nan, "rho": np.nan,
                "m": np.nan, "sigma": np.nan, "fit_error": np.nan,
                "n_points": len(strikes)}

    return {
        "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
        "fit_error": fit_error, "n_points": len(strikes),
    }


# ---------------------------------------------------------------------------
# Delta-based standardized grid
# ---------------------------------------------------------------------------

def _bs_delta_to_moneyness(delta: float, iv: float, tau: float,
                            is_call: bool = True) -> float:
    """Convert Black-Scholes delta to moneyness (K/F).

    For calls:  delta = Φ(d1), so d1 = Φ⁻¹(delta)
    For puts:   delta = Φ(d1) - 1 = -(1 - Φ(d1)), so d1 = Φ⁻¹(delta + 1)
                But delta for puts is negative, e.g. -0.25 for a 25d put.

    K/F = exp(-d1 * σ√τ + 0.5 * σ²τ)
    """
    from scipy.stats import norm
    if is_call:
        # delta is positive, e.g. 0.25
        d1 = norm.ppf(delta)
    else:
        # delta is negative, e.g. -0.25; delta + 1 = 0.75
        d1 = norm.ppf(delta + 1)

    sqrt_tau = np.sqrt(tau)
    moneyness = np.exp(-d1 * iv * sqrt_tau + 0.5 * iv ** 2 * tau)
    return moneyness


def evaluate_delta_tau_grid(
    svi_params_by_tau: dict,
    delta_grid: list[float] = None,
    tau_grid: list[float] = None,
) -> dict[str, float]:
    """Evaluate IV at standardized (delta, tau) grid points.

    Parameters
    ----------
    svi_params_by_tau : dict mapping tau -> SVI parameter dict
    delta_grid : list of delta values (negative for puts, positive for calls)
    tau_grid : list of tau values (years)

    Returns
    -------
    dict of feature_name -> IV value
    """
    if delta_grid is None:
        # Standard grid: 10d-put, 25d-put, ATM, 25d-call, 10d-call
        delta_grid = [-0.10, -0.25, 0.50, 0.25, 0.10]
    if tau_grid is None:
        tau_grid = [30 / 365, 60 / 365, 90 / 365, 180 / 365]

    delta_names = {
        -0.10: "10dp", -0.25: "25dp", 0.50: "atm",
        0.25: "25dc", 0.10: "10dc",
    }
    tau_names = {
        30 / 365: "1m", 60 / 365: "2m", 90 / 365: "3m", 180 / 365: "6m",
    }

    result = {}
    available_taus = sorted(svi_params_by_tau.keys())

    for tau in tau_grid:
        # Find closest available expiration
        if not available_taus:
            for d in delta_grid:
                dn = delta_names.get(d, f"{d:.2f}")
                tn = tau_names.get(tau, f"{int(tau * 365)}d")
                result[f"iv_{dn}_{tn}"] = np.nan
            continue

        closest_tau = min(available_taus, key=lambda t: abs(t - tau))
        # Only use if within 50% of target
        if abs(closest_tau - tau) / tau > 0.5:
            for d in delta_grid:
                dn = delta_names.get(d, f"{d:.2f}")
                tn = tau_names.get(tau, f"{int(tau * 365)}d")
                result[f"iv_{dn}_{tn}"] = np.nan
            continue

        params = svi_params_by_tau[closest_tau]
        if np.isnan(params.get("a", np.nan)):
            for d in delta_grid:
                dn = delta_names.get(d, f"{d:.2f}")
                tn = tau_names.get(tau, f"{int(tau * 365)}d")
                result[f"iv_{dn}_{tn}"] = np.nan
            continue

        # For each delta, find the corresponding log-moneyness, then evaluate SVI
        atm_w = svi_total_variance(
            np.array([0.0]), params["a"], params["b"],
            params["rho"], params["m"], params["sigma"]
        )[0]
        atm_iv = np.sqrt(atm_w / closest_tau) if closest_tau > 0 and atm_w > 0 else 0.2

        for d in delta_grid:
            dn = delta_names.get(d, f"{d:.2f}")
            tn = tau_names.get(tau, f"{int(tau * 365)}d")

            if abs(d) == 0.50:
                # ATM
                result[f"iv_{dn}_{tn}"] = atm_iv
            else:
                is_call = d > 0
                try:
                    # Pass delta with its sign: positive for calls, negative for puts
                    moneyness = _bs_delta_to_moneyness(d if is_call else d, atm_iv, closest_tau, is_call)
                    # log-moneyness = log(K/F); moneyness is already K/F
                    k = np.log(moneyness)
                    w = svi_total_variance(
                        np.array([k]), params["a"], params["b"],
                        params["rho"], params["m"], params["sigma"]
                    )[0]
                    iv = np.sqrt(w / closest_tau) if closest_tau > 0 and w > 0 else np.nan
                    result[f"iv_{dn}_{tn}"] = iv
                except Exception:
                    result[f"iv_{dn}_{tn}"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Per-ticker feature computation
# ---------------------------------------------------------------------------

def compute_surface_model_features(ticker: str, stock_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SVI parameters + delta-tau grid + dynamics for one ticker.

    Returns one row per trading day with:
      - SVI params for nearest-30d expiry: svi_a, svi_b, svi_rho, svi_m, svi_sigma
      - SVI fit quality: svi_fit_error
      - 20 delta-tau grid IVs: iv_{delta}_{tau}
      - 5 derived surface features: rr_25d_1m, bf_25d_1m, term_slope_atm, etc.
      - 5 dynamics features: 1-day changes in SVI params
    """
    opts_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"
    if not opts_path.exists():
        return pd.DataFrame()

    opts = pd.read_parquet(opts_path)
    opts["date"] = pd.to_datetime(opts["date"])
    opts["expiration"] = pd.to_datetime(opts["expiration"])

    # Get risk-free rate from stock data
    stk = stock_df[stock_df["ticker"] == ticker].copy()
    stk["date"] = pd.to_datetime(stk["date"])
    stk = stk.set_index("date")
    rf_series = stk.get("tbill_rate", pd.Series(0.0, index=stk.index)).fillna(0) / 100.0

    dates = sorted(opts["date"].unique())
    rows = []
    prev_svi = None

    for dt in dates:
        day = opts[opts["date"] == dt]
        underlying = day["underlying_price"].iloc[0] if "underlying_price" in day.columns else np.nan
        rf = rf_series.get(dt, 0.0) if dt in rf_series.index else 0.0

        row = {"ticker": ticker, "date": dt}

        # Fit SVI per expiration
        svi_by_tau = {}
        for exp, chain in day.groupby("expiration"):
            tau = (exp - dt).days / 365.0
            if tau < 0.01:
                continue
            forward = underlying * np.exp(rf * tau) if not np.isnan(underlying) else np.nan
            if np.isnan(forward) or forward <= 0:
                continue

            params = fit_svi(
                chain["strike"].values, forward, tau,
                chain["impl_volatility"].values,
                chain["volume"].values,
            )
            svi_by_tau[tau] = params

        # Pick nearest-to-30d expiry for headline SVI params
        if svi_by_tau:
            target_tau = 30 / 365
            best_tau = min(svi_by_tau.keys(), key=lambda t: abs(t - target_tau))
            headline = svi_by_tau[best_tau]
        else:
            headline = {"a": np.nan, "b": np.nan, "rho": np.nan,
                        "m": np.nan, "sigma": np.nan, "fit_error": np.nan}

        row["svi_a"] = headline["a"]         # variance level
        row["svi_b"] = headline["b"]         # wing slope
        row["svi_rho"] = headline["rho"]     # skew (-1 to 1)
        row["svi_m"] = headline["m"]         # smile shift
        row["svi_sigma"] = headline["sigma"] # ATM curvature
        row["svi_fit_error"] = headline.get("fit_error", np.nan)

        # Delta-tau grid (5 deltas x 4 tenors = 20 features)
        grid = evaluate_delta_tau_grid(svi_by_tau)
        row.update(grid)

        # Derived features from the grid
        iv_25dp_1m = grid.get("iv_25dp_1m", np.nan)
        iv_25dc_1m = grid.get("iv_25dc_1m", np.nan)
        iv_atm_1m = grid.get("iv_atm_1m", np.nan)
        iv_atm_3m = grid.get("iv_atm_3m", np.nan)
        iv_10dp_1m = grid.get("iv_10dp_1m", np.nan)
        iv_10dc_1m = grid.get("iv_10dc_1m", np.nan)

        # Risk reversal: 25d call IV - 25d put IV (measures skew)
        row["rr_25d_1m"] = iv_25dc_1m - iv_25dp_1m if not (
            np.isnan(iv_25dc_1m) or np.isnan(iv_25dp_1m)) else np.nan

        # Butterfly: average of wing IVs - ATM IV (measures curvature/smile)
        row["bf_25d_1m"] = (
            (iv_25dp_1m + iv_25dc_1m) / 2 - iv_atm_1m
            if not any(np.isnan(x) for x in [iv_25dp_1m, iv_25dc_1m, iv_atm_1m])
            else np.nan
        )

        # Term slope: 3m ATM - 1m ATM
        row["term_slope_atm"] = (
            iv_atm_3m - iv_atm_1m
            if not (np.isnan(iv_atm_3m) or np.isnan(iv_atm_1m))
            else np.nan
        )

        # Wing spread: 10d put - 10d call (tail risk asymmetry)
        row["wing_spread_1m"] = (
            iv_10dp_1m - iv_10dc_1m
            if not (np.isnan(iv_10dp_1m) or np.isnan(iv_10dc_1m))
            else np.nan
        )

        # SVI dynamics: changes from previous day
        if prev_svi is not None:
            row["d_svi_a"] = headline["a"] - prev_svi["a"]
            row["d_svi_b"] = headline["b"] - prev_svi["b"]
            row["d_svi_rho"] = headline["rho"] - prev_svi["rho"]
            row["d_svi_m"] = headline["m"] - prev_svi["m"]
            row["d_svi_sigma"] = headline["sigma"] - prev_svi["sigma"]
        else:
            for p in ["a", "b", "rho", "m", "sigma"]:
                row[f"d_svi_{p}"] = np.nan

        prev_svi = headline
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("%s: fit SVI for %d days, avg fit error %.4f",
             ticker, len(df), df["svi_fit_error"].mean())
    return df


def run(cfg=None):
    """Compute surface model features for all tickers."""
    if cfg is None:
        cfg = load_config()

    stock_path = INTERIM_DIR / "stock_clean" / "stock_daily.parquet"
    if not stock_path.exists():
        log.error("Clean stock data not found")
        return

    stock = pd.read_parquet(stock_path)
    stock["date"] = pd.to_datetime(stock["date"])

    out_dir = FEATURES_DIR / "surface_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ticker in all_tickers(cfg):
        out_path = out_dir / f"{ticker}.parquet"
        if out_path.exists():
            results.append(pd.read_parquet(out_path))
            continue

        df = compute_surface_model_features(ticker, stock)
        if not df.empty:
            save_parquet(df, out_path)
            results.append(df)

    if results:
        combined = pd.concat(results, ignore_index=True)
        save_parquet(combined, out_dir / "surface_model_all.parquet")

    log.info("=== Surface Model Features Complete ===")


if __name__ == "__main__":
    run()
