"""Master orchestrator for the options-to-equity return prediction pipeline."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.utils.config import load_config, all_tickers, load_feature_defs
from src.utils.reproducibility import set_seed
from src.utils.io_helpers import (
    load_parquet, save_parquet,
    FEATURES_DIR, SPLITS_DIR, RESULTS_DIR, INTERIM_DIR,
)
from src.utils.logger import setup_logger

log = setup_logger("pipeline")


# -----------------------------------------------------------------------
# Stage functions
# -----------------------------------------------------------------------

def stage_download(cfg):
    """Stage 1: Download all raw data."""
    log.info("========== STAGE 1: DATA DOWNLOAD ==========")
    from src.data import wrds_download, yahoo_download
    wrds_download.run(cfg)
    yahoo_download.run(cfg)


def stage_clean(cfg):
    """Stage 2: Clean options and stock data."""
    log.info("========== STAGE 2: DATA CLEANING ==========")
    from src.data import clean_options, clean_stocks, merge_sources
    clean_options.run(cfg)
    clean_stocks.run(cfg)
    merge_sources.run(cfg)


def stage_features(cfg):
    """Stage 3: Feature engineering (all resolutions)."""
    log.info("========== STAGE 3: FEATURE ENGINEERING ==========")
    from src.features import (
        target_builder, stock_features, aggregated_features,
        event_features, iv_surface_features, contract_tensor_builder,
    )
    target_builder.run(cfg)
    stock_features.run(cfg)
    aggregated_features.run(cfg)
    event_features.run(cfg)
    iv_surface_features.run(cfg)
    contract_tensor_builder.run(cfg)

    # Parametric surface features (SVI + delta-tau grid)
    from src.features import surface_model_features
    surface_model_features.run(cfg)

    # Assemble Resolution 1 feature panel
    _assemble_feature_panel(cfg)


def _assemble_feature_panel(cfg):
    """Combine all scalar features into a single panel."""
    out_path = FEATURES_DIR / "resolution_1_scalar" / "features_panel.parquet"
    if out_path.exists():
        log.info("Feature panel already exists, skipping assembly.")
        return

    # Load components
    targets = load_parquet(FEATURES_DIR / "targets.parquet")
    targets["date"] = pd.to_datetime(targets["date"])

    stock_feat = load_parquet(FEATURES_DIR / "stock_features.parquet")
    stock_feat["date"] = pd.to_datetime(stock_feat["date"])

    event_feat = load_parquet(FEATURES_DIR / "event_features.parquet")
    event_feat["date"] = pd.to_datetime(event_feat["date"])

    opts_path = FEATURES_DIR / "resolution_1_scalar" / "options_features_all.parquet"
    if opts_path.exists():
        opts_feat = load_parquet(opts_path)
        opts_feat["date"] = pd.to_datetime(opts_feat["date"])
    else:
        log.warning("No aggregated options features found, creating panel without them.")
        opts_feat = None

    # Surface model features (SVI + delta-tau grid)
    surf_path = FEATURES_DIR / "surface_model" / "surface_model_all.parquet"
    if surf_path.exists():
        surf_feat = load_parquet(surf_path)
        surf_feat["date"] = pd.to_datetime(surf_feat["date"])
    else:
        log.warning("No surface model features found.")
        surf_feat = None

    # Merge on (ticker, date)
    panel = targets.merge(stock_feat, on=["ticker", "date"], how="inner")
    panel = panel.merge(event_feat, on=["ticker", "date"], how="left")
    if opts_feat is not None:
        panel = panel.merge(opts_feat, on=["ticker", "date"], how="left")
    if surf_feat is not None:
        panel = panel.merge(surf_feat, on=["ticker", "date"], how="left")

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    save_parquet(panel, out_path)
    log.info("Assembled feature panel: %d rows, %d columns", len(panel), len(panel.columns))


def stage_preprocess(cfg):
    """Stage 4: Split, winsorize, impute, normalize."""
    log.info("========== STAGE 4: PREPROCESSING ==========")
    from src.preprocessing.split import temporal_split
    from src.preprocessing.winsorize import winsorize
    from src.preprocessing.impute import impute
    from src.preprocessing.normalize import normalize, rank_normalize

    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "features_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # Identify feature columns
    feat_cols = [c for c in panel.columns if c.startswith("feat_")]

    # Split
    splits = temporal_split(
        panel,
        train_frac=cfg.split.train_frac,
        val_frac=cfg.split.val_frac,
        test_frac=cfg.split.test_frac,
        max_target_horizon=max(cfg.targets.horizons),
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(splits, SPLITS_DIR / "split_indices.parquet")

    # Winsorize
    panel, bounds = winsorize(panel, feat_cols, splits,
                               tuple(cfg.preprocessing.winsorize_quantiles))

    # Impute
    panel = impute(panel, feat_cols, splits)

    # Save unnormalized copy (for XGBoost)
    save_parquet(panel, FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")

    # Normalize (for neural nets)
    panel_norm, stats = normalize(panel.copy(), feat_cols, splits)
    save_parquet(panel_norm, FEATURES_DIR / "resolution_1_scalar" / "panel_normalized.parquet")

    # Cross-sectional rank normalization (Gu, Kelly, Xiu 2020)
    # Ranks tickers within each date on each feature → [-1, 1]
    # Useful for models that exploit cross-sectional relative positioning
    panel_ranked = rank_normalize(panel.copy(), feat_cols)
    save_parquet(panel_ranked, FEATURES_DIR / "resolution_1_scalar" / "panel_ranked.parquet")

    log.info("Preprocessing complete.")


def _resolve_feature_ids(feat_defs: dict, fs_name: str, panel_columns: list) -> list[str]:
    """Map a feature set name to actual column IDs present in the panel.

    Handles both:
      - feat_XX references (mapped from human-readable names in feature defs)
      - Direct column names (e.g., svi_a, iv_25dp_1m for feature set D)
    """
    fs_def = feat_defs["feature_sets"][fs_name]
    feat_cols = [f["name"] if isinstance(f, dict) else f for f in fs_def["features"]]

    # Map from human-readable names to feat_XX IDs
    name_to_id = {}
    if "features" in feat_defs:
        for k, v in feat_defs["features"].items():
            name_to_id[v["name"]] = k

    # Resolve: try name->id mapping first, fall back to direct column name
    feat_ids = [name_to_id.get(f, f) for f in feat_cols]
    return [c for c in feat_ids if c in panel_columns]


def _add_ticker_encoding(panel: pd.DataFrame) -> list[str]:
    """Add one-hot ticker columns to the panel. Returns new column names."""
    dummies = pd.get_dummies(panel["ticker"], prefix="tkr", dtype=np.float32)
    # Drop one column to avoid multicollinearity (XGBoost doesn't care,
    # but OLS and neural nets do)
    drop_col = dummies.columns[-1]
    dummies = dummies.drop(columns=[drop_col])
    new_cols = dummies.columns.tolist()
    for col in new_cols:
        panel[col] = dummies[col].values
    return new_cols


def _split_arrays(panel, feat_ids, ret_col, dir_col):
    """Extract train/val/test arrays from a panel with a 'split' column."""
    train = panel["split"] == "train"
    val = panel["split"] == "val"
    test = panel["split"] == "test"
    # Cast to float32 to handle pandas nullable Float64 dtype; fill NaN→0
    def _f32(df, cols):
        arr = df.loc[:, cols].to_numpy(dtype=np.float32, na_value=np.nan)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    def _f64(df, col):
        arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    return {
        "X_train": _f32(panel[train], feat_ids),
        "X_val": _f32(panel[val], feat_ids),
        "X_test": _f32(panel[test], feat_ids),
        "y_train_ret": _f64(panel[train], ret_col),
        "y_val_ret": _f64(panel[val], ret_col),
        "y_test_ret": _f64(panel[test], ret_col),
        "y_train_dir": _f64(panel[train], dir_col),
        "y_val_dir": _f64(panel[val], dir_col),
        "y_test_dir": _f64(panel[test], dir_col),
        "tickers_train": panel.loc[train, "ticker"].values,
        "tickers_val": panel.loc[val, "ticker"].values,
        "tickers_test": panel.loc[test, "ticker"].values,
        "dates_test": panel.loc[test, "date"].values,
        "test_features": panel.loc[test],
    }


def stage_models(cfg):
    """Stage 5: Train all models.

    Data flow per model:
      XGBoost   : unnormalized panel + ticker dummies, flat (N, F) rows
      FFNN      : normalized panel + ticker dummies, flat (N, F) rows
      TabNet    : normalized panel + ticker dummies, flat (N, F) rows
      TFT       : normalized panel, per-ticker sliding windows (N', lookback, F)
      Set Trans : contract tensors (N, 300, 10) + stock/event features (N, 22)
    """
    log.info("========== STAGE 5: MODEL TRAINING ==========")
    from src.evaluation.metrics import evaluate_model

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    panel_norm = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_normalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")

    for df in (panel, panel_norm, splits):
        df["date"] = pd.to_datetime(df["date"])

    # Merge splits
    panel = panel.merge(splits, on=["ticker", "date"], how="inner")
    panel_norm = panel_norm.merge(splits, on=["ticker", "date"], how="inner")

    # Ensure sorted by (ticker, date) -- critical for TFT windowing
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel_norm = panel_norm.sort_values(["ticker", "date"]).reset_index(drop=True)

    primary_horizon = cfg.targets.primary_horizon
    ret_col = f"ret_{primary_horizon}d"
    dir_col = f"dir_{primary_horizon}d"

    panel = panel.dropna(subset=[ret_col, dir_col])
    panel_norm = panel_norm.dropna(subset=[ret_col, dir_col])

    feat_defs = load_feature_defs()

    # Add ticker encoding (helps pooled models distinguish tickers)
    ticker_cols = _add_ticker_encoding(panel)
    ticker_cols_norm = _add_ticker_encoding(panel_norm)

    all_results = []

    # ------------------------------------------------------------------
    # Loop over feature sets (A, B, C, candidate_6)
    # ------------------------------------------------------------------
    for fs_name in feat_defs["feature_sets"]:
        feat_ids = _resolve_feature_ids(feat_defs, fs_name, panel.columns.tolist())
        if not feat_ids:
            log.warning("No features available for set %s, skipping", fs_name)
            continue

        # Append ticker encoding columns to feature list
        feat_ids_with_ticker = feat_ids + ticker_cols
        log.info("Feature set %s: %d features + %d ticker dummies = %d total",
                 fs_name, len(feat_ids), len(ticker_cols), len(feat_ids_with_ticker))

        d = _split_arrays(panel, feat_ids_with_ticker, ret_col, dir_col)
        d_norm = _split_arrays(panel_norm, feat_ids_with_ticker, ret_col, dir_col)

        # --- Benchmarks (only for full feature set) ---
        if fs_name == "C":
            from src.models.benchmarks import run_benchmarks
            bench = run_benchmarks(
                d["X_train"], d["y_train_ret"], d["y_train_dir"],
                d["X_test"], d["y_test_ret"], d["y_test_dir"],
                feature_names=feat_ids_with_ticker,
            )
            bench["feature_set"] = fs_name
            bench["horizon"] = primary_horizon
            all_results.append(bench)

        # --- XGBoost (unnormalized, tree-based doesn't need normalization) ---
        from src.models.xgboost_model import run_xgb_experiment
        xgb_result = run_xgb_experiment(
            d["X_train"], d["y_train_ret"], d["y_train_dir"],
            d["X_val"], d["y_val_ret"], d["y_val_dir"],
            d["X_test"], d["y_test_ret"], d["y_test_dir"],
            vars(cfg.models.xgboost), fs_name, primary_horizon,
        )
        m = evaluate_model(
            f"XGBoost_{fs_name}", d["y_test_ret"], d["y_test_dir"],
            xgb_result["reg_pred"], xgb_result["clf_pred"], xgb_result["clf_proba"],
            d["y_train_ret"].mean(), cfg.evaluation.transaction_cost_bps,
            dates=d["dates_test"], horizon=primary_horizon,
        )
        m["feature_set"], m["horizon"] = fs_name, primary_horizon
        all_results.append(pd.DataFrame([m]))

        # --- FFNN (normalized) ---
        from src.models.ffnn_model import run_ffnn_experiment
        ffnn_result = run_ffnn_experiment(
            d_norm["X_train"], d_norm["y_train_ret"], d_norm["y_train_dir"],
            d_norm["X_val"], d_norm["y_val_ret"], d_norm["y_val_dir"],
            d_norm["X_test"], d_norm["y_test_ret"], d_norm["y_test_dir"],
            vars(cfg.models.ffnn), fs_name, primary_horizon,
        )
        m = evaluate_model(
            f"FFNN_{fs_name}", d["y_test_ret"], d["y_test_dir"],
            ffnn_result["reg_pred"], ffnn_result["clf_pred"], ffnn_result["clf_proba"],
            d["y_train_ret"].mean(), cfg.evaluation.transaction_cost_bps,
            dates=d["dates_test"], horizon=primary_horizon,
        )
        m["feature_set"], m["horizon"] = fs_name, primary_horizon
        all_results.append(pd.DataFrame([m]))

        # --- TabNet (normalized) ---
        try:
            from src.models.tabnet_model import run_tabnet_experiment
            tabnet_result = run_tabnet_experiment(
                d_norm["X_train"], d_norm["y_train_ret"], d_norm["y_train_dir"],
                d_norm["X_val"], d_norm["y_val_ret"], d_norm["y_val_dir"],
                d_norm["X_test"], d_norm["y_test_ret"], d_norm["y_test_dir"],
                vars(cfg.models.tabnet), fs_name, primary_horizon,
            )
            m = evaluate_model(
                f"TabNet_{fs_name}", d["y_test_ret"], d["y_test_dir"],
                tabnet_result["reg_pred"], tabnet_result["clf_pred"], tabnet_result["clf_proba"],
                d["y_train_ret"].mean(), cfg.evaluation.transaction_cost_bps,
                dates=d["dates_test"], horizon=primary_horizon,
            )
            m["feature_set"], m["horizon"] = fs_name, primary_horizon
            all_results.append(pd.DataFrame([m]))
        except ImportError:
            log.warning("pytorch_tabnet not installed, skipping TabNet")
        except Exception as e:
            log.warning("TabNet failed for %s: %s", fs_name, e)

        # --- TFT (normalized, per-ticker windowing, no ticker dummies) ---
        # TFT gets raw feature IDs without ticker dummies -- it processes
        # each ticker's sequence separately via MultiTickerTimeSeriesDataset
        feat_ids_no_ticker = feat_ids  # temporal model doesn't need ticker dummies
        d_tft = _split_arrays(panel_norm, feat_ids_no_ticker, ret_col, dir_col)
        try:
            from src.models.tft_model import run_tft_experiment
            tft_result = run_tft_experiment(
                d_tft["X_train"], d_tft["y_train_ret"],
                d_tft["X_val"], d_tft["y_val_ret"],
                d_tft["X_test"], d_tft["y_test_ret"],
                vars(cfg.models.tft), fs_name, primary_horizon,
                tickers_train=d_tft["tickers_train"],
                tickers_val=d_tft["tickers_val"],
                tickers_test=d_tft["tickers_test"],
            )
            # TFT predictions are fewer than test rows (lookback warmup).
            # The MultiTickerTimeSeriesDataset drops the first `lookback`
            # rows per ticker, so n_pred < n_test.  We align ground truth
            # by rebuilding the same dataset to know which rows survived.
            n_pred = len(tft_result["predictions"])
            tft_pred_ret = tft_result["predictions"]
            tft_pred_dir = (tft_pred_ret > 0).astype(int)

            # Reconstruct aligned ground truth: same order as prediction
            from src.models.tft_model import MultiTickerTimeSeriesDataset
            lookback = vars(cfg.models.tft).get("lookback_window", 20)
            _test_ds = MultiTickerTimeSeriesDataset(
                d_tft["X_test"], d_tft["y_test_ret"],
                d_tft["tickers_test"], lookback,
            )
            tft_y_test = np.array([seq[1].item() for seq in _test_ds.sequences])
            tft_y_dir = (tft_y_test > 0).astype(int)

            # Align test dates with retained sequence rows
            if hasattr(_test_ds, "global_indices") and len(_test_ds.global_indices):
                tft_dates_test = d_tft["dates_test"][np.array(_test_ds.global_indices)]
            else:
                tft_dates_test = None
            m = evaluate_model(
                f"TFT_{fs_name}", tft_y_test, tft_y_dir,
                tft_pred_ret, tft_pred_dir, None,
                d_tft["y_train_ret"].mean(), cfg.evaluation.transaction_cost_bps,
                dates=tft_dates_test, horizon=primary_horizon,
            )
            m["feature_set"], m["horizon"] = fs_name, primary_horizon
            all_results.append(pd.DataFrame([m]))
            log.info("TFT_%s: %d predictions, OOS R²=%.4f, Acc=%.4f",
                     fs_name, n_pred, m["oos_r2"], m["accuracy"])
        except Exception as e:
            log.warning("TFT failed for %s: %s", fs_name, e)

    # ------------------------------------------------------------------
    # Set Transformer (Resolution 3) -- proof of concept on select tickers
    #
    # Two modes:
    #   "contracts" : stratified-sampled individual contracts (300 x 10)
    #   "chains"    : per-expiration aggregated chains (20 x 16)
    #
    # Running both lets us test whether the model benefits from raw
    # contract granularity or if chain summaries are sufficient.
    # ------------------------------------------------------------------
    set_transformer_tickers = ["AAPL", "NVDA", "SPY"]
    log.info("=== Set Transformer: proof of concept on %s ===", set_transformer_tickers)

    try:
        from src.models.set_transformer_model import run_set_transformer_experiment

        # Stock + event features for the Set Transformer's secondary input
        stock_event_ids = _resolve_feature_ids(feat_defs, "A", panel_norm.columns.tolist())

        tensor_modes = [
            # (suffix, file_pattern, label)
            ("contracts", "{ticker}.npz", "R3_contract"),
            ("chains", "{ticker}_chains.npz", "R3_chain"),
        ]

        for ticker in set_transformer_tickers:
            for mode_name, file_pattern, label_prefix in tensor_modes:
                tensor_path = FEATURES_DIR / "resolution_3_contracts" / file_pattern.format(ticker=ticker)
                if not tensor_path.exists():
                    log.warning("No %s tensor for %s, skipping", mode_name, ticker)
                    continue

                data = np.load(tensor_path, allow_pickle=True)
                tensor_dates = pd.to_datetime(data["dates"])
                tensors = data["tensors"]
                masks = data["masks"]

                # Align with stock features and targets by date
                ticker_panel = panel_norm[panel_norm["ticker"] == ticker].copy()
                ticker_panel["date"] = pd.to_datetime(ticker_panel["date"])

                common_dates = sorted(set(tensor_dates) & set(ticker_panel["date"]))
                if len(common_dates) < 100:
                    log.warning("Only %d common dates for %s %s, skipping",
                                len(common_dates), ticker, mode_name)
                    continue

                tensor_date_idx = {d: i for i, d in enumerate(tensor_dates)}
                panel_indexed = ticker_panel.set_index("date")

                aligned_tensors, aligned_masks = [], []
                aligned_stock, aligned_ret, aligned_dir, aligned_splits = [], [], [], []

                for dt in common_dates:
                    ti = tensor_date_idx[dt]
                    aligned_tensors.append(tensors[ti])
                    aligned_masks.append(masks[ti])
                    row = panel_indexed.loc[dt]
                    aligned_stock.append(row[stock_event_ids].values.astype(np.float32))
                    aligned_ret.append(row[ret_col])
                    aligned_dir.append(row[dir_col])
                    aligned_splits.append(row["split"])

                aligned_tensors = np.array(aligned_tensors)
                aligned_masks = np.array(aligned_masks)
                aligned_stock = np.array(aligned_stock)
                aligned_ret = np.array(aligned_ret, dtype=np.float32)
                aligned_dir = np.array(aligned_dir, dtype=np.float32)
                aligned_splits = np.array(aligned_splits)

                valid = ~(np.isnan(aligned_ret) | np.isnan(aligned_dir))
                aligned_tensors = aligned_tensors[valid]
                aligned_masks = aligned_masks[valid]
                aligned_stock = aligned_stock[valid]
                aligned_ret = aligned_ret[valid]
                aligned_dir = aligned_dir[valid]
                aligned_splits = aligned_splits[valid]

                tr = aligned_splits == "train"
                va = aligned_splits == "val"
                te = aligned_splits == "test"

                if te.sum() < 10:
                    log.warning("Too few test samples for %s %s", ticker, mode_name)
                    continue

                model_label = f"SetTrans_{mode_name}_{ticker}"
                st_result = run_set_transformer_experiment(
                    aligned_tensors[tr], aligned_masks[tr], aligned_stock[tr], aligned_ret[tr],
                    aligned_tensors[va], aligned_masks[va], aligned_stock[va], aligned_ret[va],
                    aligned_tensors[te], aligned_masks[te], aligned_stock[te], aligned_ret[te],
                    vars(cfg.models.set_transformer), model_label, primary_horizon,
                )
                pred_dir_st = (st_result["predictions"] > 0).astype(int)
                m = evaluate_model(
                    model_label, aligned_ret[te], aligned_dir[te].astype(int),
                    st_result["predictions"], pred_dir_st, None,
                    aligned_ret[tr].mean(), cfg.evaluation.transaction_cost_bps,
                )
                m["feature_set"] = f"{label_prefix}_{ticker}"
                m["horizon"] = primary_horizon
                all_results.append(pd.DataFrame([m]))
                log.info("%s: OOS R²=%.4f, Acc=%.4f",
                         model_label, m["oos_r2"], m["accuracy"])

    except ImportError as e:
        log.warning("Set Transformer dependencies not available: %s", e)
    except Exception as e:
        log.error("Set Transformer stage failed: %s", e)

    # ------------------------------------------------------------------
    # Save combined results
    # ------------------------------------------------------------------
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
        combined.to_csv(RESULTS_DIR / "tables" / "model_comparison.csv", index=False)
        log.info("Saved model comparison table: %d entries", len(combined))


def stage_evaluate(cfg):
    """Stage 6: Statistical tests, regime analysis, conformal intervals.

    Requires Stage 5 to have produced model_comparison.csv with saved
    predictions, OR we re-run lightweight evaluation from saved checkpoints.
    For simplicity, this stage loads the panel + split, re-runs XGBoost from
    checkpoints, and performs statistical + regime analysis.
    """
    log.info("========== STAGE 6: EVALUATION ==========")
    from src.evaluation.statistical_tests import (
        clark_west_test, diebold_mariano_test, mcnemar_test,
    )
    from src.evaluation.regime_analysis import full_regime_analysis
    from src.evaluation.rolling_window import expanding_window_eval

    # ------------------------------------------------------------------
    # Load panel + splits (same as Stage 5)
    # ------------------------------------------------------------------
    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel, splits):
        df["date"] = pd.to_datetime(df["date"])
    panel = panel.merge(splits, on=["ticker", "date"], how="inner")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    feat_defs = load_feature_defs()
    primary_horizon = cfg.targets.primary_horizon
    ret_col = f"ret_{primary_horizon}d"
    dir_col = f"dir_{primary_horizon}d"
    panel = panel.dropna(subset=[ret_col, dir_col])

    # ------------------------------------------------------------------
    # Re-load XGBoost checkpoints for set A and set C
    # ------------------------------------------------------------------
    import xgboost as xgb

    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    predictions = {}  # {model_label: {"pred_ret": ..., "pred_dir": ...}}

    for fs_name in ["A", "C"]:
        reg_path = ckpt_dir / f"xgb_{fs_name}_{primary_horizon}d_reg.json"
        clf_path = ckpt_dir / f"xgb_{fs_name}_{primary_horizon}d_clf.json"
        if not reg_path.exists():
            log.warning("No XGBoost checkpoint for set %s, skipping", fs_name)
            continue

        feat_ids = _resolve_feature_ids(feat_defs, fs_name, panel.columns.tolist())
        ticker_cols = _add_ticker_encoding(panel) if f"tkr_{panel['ticker'].iloc[0]}" not in panel.columns else [c for c in panel.columns if c.startswith("tkr_")]
        feat_ids_full = feat_ids + ticker_cols

        test_mask = panel["split"] == "test"
        X_test = panel.loc[test_mask, feat_ids_full].values
        y_test_ret = panel.loc[test_mask, ret_col].values
        y_test_dir = panel.loc[test_mask, dir_col].values
        y_train_mean = panel.loc[panel["split"] == "train", ret_col].mean()

        reg_model = xgb.XGBRegressor()
        reg_model.load_model(str(reg_path))
        pred_ret = reg_model.predict(X_test)

        clf_model = xgb.XGBClassifier()
        clf_model.load_model(str(clf_path))
        pred_dir = clf_model.predict(X_test)
        pred_proba = clf_model.predict_proba(X_test)[:, 1]

        predictions[f"XGBoost_{fs_name}"] = {
            "pred_ret": pred_ret, "pred_dir": pred_dir,
            "pred_proba": pred_proba,
            "y_test_ret": y_test_ret, "y_test_dir": y_test_dir,
            "y_train_mean": y_train_mean,
            "test_features": panel.loc[test_mask],
            "dates": panel.loc[test_mask, "date"].values,
        }

    # ------------------------------------------------------------------
    # Statistical tests: A vs C (Clark-West, Diebold-Mariano, McNemar)
    # ------------------------------------------------------------------
    stat_results = []
    if "XGBoost_A" in predictions and "XGBoost_C" in predictions:
        pA = predictions["XGBoost_A"]
        pC = predictions["XGBoost_C"]
        y = pA["y_test_ret"]

        cw = clark_west_test(y, pA["pred_ret"], pC["pred_ret"])
        cw["test"] = "Clark-West (A vs C)"
        stat_results.append(cw)

        dm = diebold_mariano_test(y, pA["pred_ret"], pC["pred_ret"], horizon=primary_horizon)
        dm["test"] = "Diebold-Mariano (A vs C)"
        stat_results.append(dm)

        mn = mcnemar_test(pA["y_test_dir"], pA["pred_dir"], pC["pred_dir"])
        mn["test"] = "McNemar (A vs C)"
        stat_results.append(mn)

        stat_df = pd.DataFrame(stat_results)
        stat_df.to_csv(RESULTS_DIR / "tables" / "statistical_tests.csv", index=False)
        log.info("Statistical tests saved (%d tests)", len(stat_df))

    # ------------------------------------------------------------------
    # Regime analysis for best model (XGBoost_C)
    # ------------------------------------------------------------------
    regime_results = []
    for label, pred in predictions.items():
        try:
            rdf = full_regime_analysis(
                label,
                pred["y_test_ret"], pred["y_test_dir"],
                pred["pred_ret"], pred["pred_dir"], pred["pred_proba"],
                pred["y_train_mean"],
                pred["test_features"], cfg,
                dates=pred["dates"], horizon=primary_horizon,
            )
            regime_results.append(rdf)
        except Exception as e:
            log.warning("Regime analysis failed for %s: %s", label, e)

    if regime_results:
        regime_df = pd.concat(regime_results, ignore_index=True)
        regime_df.to_csv(RESULTS_DIR / "tables" / "regime_analysis.csv", index=False)
        log.info("Regime analysis saved (%d rows)", len(regime_df))

    # ------------------------------------------------------------------
    # Walk-forward backtest: XGBoost across all feature sets.
    #
    # Two schemes are run side-by-side so the paper can compare them:
    #   * expanding window : Gu-Kelly-Xiu / Bali et al. baseline
    #   * rolling window   : 9-month train / 3-month test, slide forward.
    #
    # Both report per-window metrics AND pooled OOS R²/IC/accuracy across
    # the concatenated test predictions (Bali et al. 2022, Eq. 4).
    # ------------------------------------------------------------------
    from src.evaluation.rolling_window import rolling_window_eval

    def xgb_factory():
        return xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )

    ticker_cols = [c for c in panel.columns if c.startswith("tkr_")]
    if not ticker_cols:
        ticker_cols = _add_ticker_encoding(panel)

    all_expanding, all_rolling, pooled_rows = [], [], []

    for fs_name in ["A", "B", "C", "candidate_6", "D"]:
        try:
            feat_ids = _resolve_feature_ids(feat_defs, fs_name, panel.columns.tolist())
        except Exception:
            continue
        if not feat_ids:
            continue
        feat_ids_full = feat_ids + ticker_cols

        X_all = panel[feat_ids_full].to_numpy(dtype=np.float32, na_value=0.0)
        np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_all_ret = panel[ret_col].to_numpy(dtype=np.float64, na_value=0.0)
        y_all_dir = panel[dir_col].to_numpy(dtype=np.float64, na_value=0.0)
        dates_all = panel["date"].values

        log.info("=== Walk-forward: XGBoost on set %s (%d features) ===",
                 fs_name, len(feat_ids_full))

        exp_df, exp_pool = expanding_window_eval(
            dates_all, X_all, y_all_ret, y_all_dir,
            model_factory=xgb_factory,
            step=63, min_train=504,
            transaction_cost_bps=cfg.evaluation.transaction_cost_bps,
        )
        if not exp_df.empty:
            exp_df["feature_set"] = fs_name
            all_expanding.append(exp_df)
            exp_pool.update({"model": "XGBoost", "feature_set": fs_name,
                             "scheme": "expanding", "n_windows": len(exp_df)})
            pooled_rows.append(exp_pool)

        roll_df, roll_pool = rolling_window_eval(
            dates_all, X_all, y_all_ret, y_all_dir,
            model_factory=xgb_factory,
            train_size=189, test_size=63, step=63,
            transaction_cost_bps=cfg.evaluation.transaction_cost_bps,
        )
        if not roll_df.empty:
            roll_df["feature_set"] = fs_name
            all_rolling.append(roll_df)
            roll_pool.update({"model": "XGBoost", "feature_set": fs_name,
                              "scheme": "rolling_9m_3m", "n_windows": len(roll_df)})
            pooled_rows.append(roll_pool)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if all_expanding:
        exp_all = pd.concat(all_expanding, ignore_index=True)
        exp_all.to_csv(tables_dir / "expanding_window.csv", index=False)
        log.info("Expanding-window results saved (%d rows, %d feature sets)",
                 len(exp_all), exp_all["feature_set"].nunique())
    if all_rolling:
        roll_all = pd.concat(all_rolling, ignore_index=True)
        roll_all.to_csv(tables_dir / "rolling_window.csv", index=False)
        log.info("Rolling-window results saved (%d rows, %d feature sets)",
                 len(roll_all), roll_all["feature_set"].nunique())
    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows)
        pooled_df.to_csv(tables_dir / "walk_forward_pooled.csv", index=False)
        log.info("Pooled walk-forward results saved (%d rows)", len(pooled_df))


def stage_explain(cfg):
    """Stage 7: Explainability analysis."""
    log.info("========== STAGE 7: EXPLAINABILITY ==========")
    log.info("Run explainability notebooks for SHAP, attention, and consensus analysis.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Options Research Pipeline")
    parser.add_argument(
        "--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Which stages to run (1=download, 2=clean, 3=features, 4=preprocess, 5=models, 6=evaluate, 7=explain)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to settings.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_seed, cfg.deterministic)

    stage_map = {
        1: stage_download,
        2: stage_clean,
        3: stage_features,
        4: stage_preprocess,
        5: stage_models,
        6: stage_evaluate,
        7: stage_explain,
    }

    for s in args.stages:
        if s in stage_map:
            stage_map[s](cfg)
        else:
            log.warning("Unknown stage %d, skipping", s)

    log.info("========== PIPELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
