"""Resolution 2: IV surface grid features (15 grid + 5 derived)."""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator

from src.utils.config import load_config, all_tickers
from src.utils.io_helpers import save_parquet, INTERIM_DIR, FEATURES_DIR
from src.utils.logger import setup_logger

log = setup_logger(__name__)

OPTIONS_CLEAN_DIR = INTERIM_DIR / "options_clean"


def _interpolate_surface(
    moneyness: np.ndarray,
    dte: np.ndarray,
    iv: np.ndarray,
    m_grid: list[float],
    d_grid: list[int],
) -> dict[str, float]:
    """Interpolate IV surface at grid points.

    Returns dict of feature_name -> value.
    """
    result = {}
    points = np.column_stack([moneyness, dte])
    mg, dg = np.meshgrid(m_grid, d_grid, indexing="ij")
    grid_points = np.column_stack([mg.ravel(), dg.ravel()])

    # Try cubic interpolation
    try:
        grid_iv = griddata(points, iv, grid_points, method="cubic")
    except Exception:
        grid_iv = np.full(len(grid_points), np.nan)

    # Fallback to nearest-neighbor for NaN values
    nan_mask = np.isnan(grid_iv)
    if nan_mask.any() and len(points) >= 3:
        try:
            nn = NearestNDInterpolator(points, iv)
            grid_iv[nan_mask] = nn(grid_points[nan_mask])
        except Exception:
            pass

    idx = 0
    for m in m_grid:
        for d in d_grid:
            m_str = f"{int(m * 100):03d}"
            d_str = f"{d:03d}"
            result[f"iv_surf_{m_str}_{d_str}"] = grid_iv[idx]
            idx += 1

    return result


def _surface_derived_features(grid: dict[str, float]) -> dict[str, float]:
    """Compute 5 derived features from the IV surface grid."""
    result = {}

    result["surface_level"] = grid.get("iv_surf_100_030", np.nan)

    iv_090_030 = grid.get("iv_surf_090_030", np.nan)
    iv_110_030 = grid.get("iv_surf_110_030", np.nan)
    iv_090_090 = grid.get("iv_surf_090_090", np.nan)
    iv_110_090 = grid.get("iv_surf_110_090", np.nan)
    iv_100_030 = grid.get("iv_surf_100_030", np.nan)
    iv_100_090 = grid.get("iv_surf_100_090", np.nan)

    result["surface_skew_30d"] = iv_090_030 - iv_110_030
    result["surface_skew_90d"] = iv_090_090 - iv_110_090
    result["surface_term_atm"] = iv_100_030 - iv_100_090
    result["surface_curvature_30d"] = iv_090_030 + iv_110_030 - 2 * iv_100_030

    return result


def compute_surface_features(ticker: str, cfg) -> pd.DataFrame:
    """Compute IV surface features for one ticker."""
    opts_path = OPTIONS_CLEAN_DIR / f"{ticker}.parquet"
    if not opts_path.exists():
        return pd.DataFrame()

    opts = pd.read_parquet(opts_path)
    opts["date"] = pd.to_datetime(opts["date"])

    m_grid = cfg.iv_surface.moneyness_grid
    d_grid = cfg.iv_surface.dte_grid

    rows = []
    for dt, day_opts in opts.groupby("date"):
        # Filter to contracts with reasonable data
        valid = day_opts[
            (day_opts["volume"] > 0) | (day_opts["open_interest"] > 100)
        ]
        valid = valid.dropna(subset=["moneyness", "dte", "impl_volatility"])

        row = {"ticker": ticker, "date": dt}

        if len(valid) < 5:
            # Not enough data for surface interpolation
            for m in m_grid:
                for d in d_grid:
                    m_str = f"{int(m * 100):03d}"
                    d_str = f"{d:03d}"
                    row[f"iv_surf_{m_str}_{d_str}"] = np.nan
            for k in ["surface_level", "surface_skew_30d", "surface_skew_90d",
                       "surface_term_atm", "surface_curvature_30d"]:
                row[k] = np.nan
        else:
            grid = _interpolate_surface(
                valid["moneyness"].values,
                valid["dte"].values,
                valid["impl_volatility"].values,
                m_grid, d_grid,
            )
            row.update(grid)
            row.update(_surface_derived_features(grid))

        rows.append(row)

    return pd.DataFrame(rows)


def run(cfg=None):
    """Compute IV surface features for all tickers."""
    if cfg is None:
        cfg = load_config()

    out_dir = FEATURES_DIR / "resolution_2_surface"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ticker in all_tickers(cfg):
        out_path = out_dir / f"{ticker}.parquet"
        if out_path.exists():
            log.info("Skipping %s (surface already computed)", ticker)
            results.append(pd.read_parquet(out_path))
            continue

        df = compute_surface_features(ticker, cfg)
        if not df.empty:
            save_parquet(df, out_path)
            results.append(df)
            log.info("Surface features for %s: %d days", ticker, len(df))

    if results:
        combined = pd.concat(results, ignore_index=True)
        save_parquet(combined, out_dir / "surface_features_all.parquet")

    log.info("=== IV Surface Features Complete ===")


if __name__ == "__main__":
    run()
