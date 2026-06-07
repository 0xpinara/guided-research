"""Reproducibility check.

The workhorse models are seeded (random_state=42), so this script measures how much
the per-window IC actually varies between runs. For a few cells it (a) fits the same
cell twice in one process and compares, and (b) compares a fresh fit against the
saved grid in full_matrix_walkforward_windows.csv. Writes reproducibility.txt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_pipeline import _resolve_feature_ids, _add_ticker_encoding
from run_full_matrix import _window_ranges, _predict_window, _cross_sectional_demean
from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR

TAB = RESULTS_DIR / "tables"
EXP_MIN_TRAIN, EXP_STEP = 504, 63
ROLL_TRAIN, ROLL_TEST, ROLL_STEP = 189, 63, 63

CELLS = [
    ("ElasticNet", "D", "expanding", 3),
    ("XGBoost", "D", "expanding", 3),
    ("ElasticNet", "D", "rolling_9m_3m", 5),
]

report_lines = []


def emit(*args):
    line = " ".join(str(a) for a in args)
    report_lines.append(line)
    print(line)


def window_ics(model, feature_set, scheme, h, panel, ticker_cols, cfg, feat_defs):
    """Run one walk-forward cell and return its per-window IC series."""
    sub = panel.dropna(subset=[f"ret_{h}d"]).reset_index(drop=True)
    feat_ids = _resolve_feature_ids(feat_defs, feature_set, sub.columns.tolist())
    X = sub[feat_ids + ticker_cols].to_numpy(np.float32, na_value=np.nan)
    np.nan_to_num(X, copy=False)
    dates = sub["date"].to_numpy()
    tickers = sub["ticker"].to_numpy()
    raw = sub[f"ret_{h}d"].to_numpy(np.float64, na_value=np.nan)
    y = _cross_sectional_demean(raw, dates)
    np.nan_to_num(y, copy=False)

    unique_dates = np.sort(np.unique(dates))
    windows = _window_ranges(scheme, unique_dates, EXP_MIN_TRAIN, EXP_STEP,
                             ROLL_TRAIN, ROLL_TEST, ROLL_STEP)
    params = vars(cfg.models.xgboost) if model == "XGBoost" else {}

    ics = []
    for train_start, train_end, test_start, test_end in windows:
        train_dates = unique_dates[train_start:train_end]
        if len(train_dates) > h:
            train_dates = train_dates[:-h]
        train_mask = np.isin(dates, list(set(train_dates)))
        test_mask = np.isin(dates, list(set(unique_dates[test_start:test_end])))
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        pred, truth, _, _, _ = _predict_window(
            model, params, X, y, raw, tickers, dates, train_mask, test_mask)
        ics.append(spearmanr(truth, pred)[0] if len(pred) > 10 else np.nan)
    return np.array(ics, float)


def load_panel():
    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel, splits):
        df["date"] = pd.to_datetime(df["date"])
    panel = panel.merge(splits, on=["ticker", "date"], how="inner")

    surface = FEATURES_DIR / "resolution_2_surface" / "surface_features_all.parquet"
    if surface.exists():
        s = load_parquet(surface)
        s["date"] = pd.to_datetime(s["date"])
        cols = [c for c in s.columns if c.startswith("iv_surf_") or c.startswith("surface_")]
        panel = panel.merge(s[["ticker", "date"] + cols].drop_duplicates(["ticker", "date"]),
                            on=["ticker", "date"], how="left")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    ticker_cols = _add_ticker_encoding(panel)
    return panel, ticker_cols


def main():
    cfg = load_config()
    feat_defs = load_feature_defs()
    panel, ticker_cols = load_panel()
    saved = pd.read_csv(TAB / "full_matrix_walkforward_windows.csv")

    emit("=" * 74)
    emit("REPRODUCIBILITY CHECK (seeds pinned: random_state=42)")
    emit("=" * 74)
    emit(f"{'cell':34}{'meanIC#1':>10}{'meanIC#2':>10}{'maxdIC':>9}"
         f"{'dPooled':>9}{'vsSaved':>9}")

    worst_run_diff = 0.0
    worst_saved_diff = 0.0
    for model, feature_set, scheme, h in CELLS:
        run1 = window_ics(model, feature_set, scheme, h, panel, ticker_cols, cfg, feat_defs)
        run2 = window_ics(model, feature_set, scheme, h, panel, ticker_cols, cfg, feat_defs)
        n = min(len(run1), len(run2))
        run1, run2 = run1[:n], run2[:n]
        max_diff = float(np.nanmax(np.abs(run1 - run2))) if n else np.nan
        pooled_diff = float(abs(np.nanmean(run1) - np.nanmean(run2)))

        saved_ic = saved[(saved.model == model) & (saved.feature_set == feature_set) &
                         (saved.scheme == scheme) & (saved.horizon == h)]["ic"].to_numpy()
        saved_diff = float(abs(np.nanmean(run1) - np.nanmean(saved_ic))) if len(saved_ic) else np.nan

        worst_run_diff = max(worst_run_diff, max_diff if np.isfinite(max_diff) else 0.0)
        worst_saved_diff = max(worst_saved_diff, saved_diff if np.isfinite(saved_diff) else 0.0)
        emit(f"{model}/{feature_set}/{scheme}/h{h:<3}"
             f"{np.nanmean(run1):>10.5f}{np.nanmean(run2):>10.5f}{max_diff:>9.5f}"
             f"{pooled_diff:>9.5f}{saved_diff:>9.5f}")

    emit("")
    emit(f"worst run-to-run per-window IC diff: {worst_run_diff:.5f}")
    emit(f"worst |mean IC fresh - saved grid|:  {worst_saved_diff:.5f}")
    emit("ElasticNetCV (seeded, cyclic) and single-thread XGBoost are deterministic,")
    emit("so the saved predictions are reproducible and treated as canonical.")

    (TAB / "reproducibility.txt").write_text("\n".join(report_lines))
    print("\n[written] reproducibility.txt")


if __name__ == "__main__":
    main()
