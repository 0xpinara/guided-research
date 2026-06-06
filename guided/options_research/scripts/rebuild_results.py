"""Rebuild model_comparison.csv from saved checkpoints.

Used when the economic-metrics code has been updated and we want to
re-evaluate without retraining. Works for the panel models (XGBoost,
FFNN, TabNet, TFT). Set-Transformer rows are carried over unchanged
(they are single-ticker and already evaluate sensibly).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR
from src.evaluation.metrics import evaluate_model
from src.models.ffnn_model import FFNN, predict_ffnn
from src.models.tft_model import SimpleTFT, MultiTickerTimeSeriesDataset, predict_tft
from run_pipeline import _resolve_feature_ids, _add_ticker_encoding, _split_arrays


def _load_xgb_predictions(fs_name, d):
    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    reg_path = ckpt_dir / f"xgb_{fs_name}_3d_reg.json"
    clf_path = ckpt_dir / f"xgb_{fs_name}_3d_clf.json"
    if not reg_path.exists():
        return None
    reg = xgb.XGBRegressor(); reg.load_model(str(reg_path))
    clf = xgb.XGBClassifier(); clf.load_model(str(clf_path))
    return {
        "reg_pred": reg.predict(d["X_test"]),
        "clf_pred": clf.predict(d["X_test"]),
        "clf_proba": clf.predict_proba(d["X_test"])[:, 1],
    }


def _load_ffnn_predictions(fs_name, d_norm):
    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    reg_path = ckpt_dir / f"ffnn_{fs_name}_3d_reg.pt"
    clf_path = ckpt_dir / f"ffnn_{fs_name}_3d_clf.pt"
    if not reg_path.exists():
        return None

    input_dim = d_norm["X_test"].shape[1]
    reg_model = FFNN(input_dim=input_dim)
    reg_model.load_state_dict(torch.load(reg_path, map_location="cpu"))
    clf_model = FFNN(input_dim=input_dim)
    clf_model.load_state_dict(torch.load(clf_path, map_location="cpu"))

    reg_pred = predict_ffnn(reg_model, d_norm["X_test"], "regression")
    clf_proba = predict_ffnn(clf_model, d_norm["X_test"], "classification")
    clf_pred = (clf_proba > 0.5).astype(int)
    return {"reg_pred": reg_pred, "clf_pred": clf_pred, "clf_proba": clf_proba}


def _load_tabnet_predictions(fs_name, d_norm):
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    except ImportError:
        return None

    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    reg_path = ckpt_dir / f"tabnet_{fs_name}_3d_reg.zip"
    clf_path = ckpt_dir / f"tabnet_{fs_name}_3d_clf.zip"
    if not reg_path.exists():
        return None

    reg = TabNetRegressor(); reg.load_model(str(reg_path))
    clf = TabNetClassifier(); clf.load_model(str(clf_path))
    reg_pred = reg.predict(d_norm["X_test"].astype(np.float32)).ravel()
    clf_pred = clf.predict(d_norm["X_test"].astype(np.float32)).ravel()
    clf_proba = clf.predict_proba(d_norm["X_test"].astype(np.float32))[:, 1]
    return {"reg_pred": reg_pred, "clf_pred": clf_pred, "clf_proba": clf_proba}


def _load_tft_predictions(fs_name, d_tft, params):
    ckpt_dir = RESULTS_DIR / "model_checkpoints"
    path = ckpt_dir / f"tft_{fs_name}_3d.pt"
    if not path.exists():
        return None

    lookback = params.get("lookback_window", 20)
    state = torch.load(path, map_location="cpu")
    n_features = d_tft["X_test"].shape[1]
    model = SimpleTFT(
        n_features=n_features,
        hidden_size=params.get("hidden_size", 64),
        attention_heads=params.get("attention_heads", 4),
        dropout=params.get("dropout", 0.1),
        lookback=lookback,
    )
    model.load_state_dict(state)

    preds, _ = predict_tft(model, d_tft["X_test"], tickers=d_tft["tickers_test"], lookback=lookback)

    test_ds = MultiTickerTimeSeriesDataset(
        d_tft["X_test"], d_tft["y_test_ret"], d_tft["tickers_test"], lookback,
    )
    y_test = np.array([seq[1].item() for seq in test_ds.sequences])
    y_dir = (y_test > 0).astype(int)
    dates = d_tft["dates_test"][np.array(test_ds.global_indices)]
    return {"preds": preds, "y_test": y_test, "y_dir": y_dir, "dates": dates}


def main():
    cfg = load_config()
    feat_defs = load_feature_defs()
    primary_horizon = cfg.targets.primary_horizon
    ret_col = f"ret_{primary_horizon}d"
    dir_col = f"dir_{primary_horizon}d"
    tc = cfg.evaluation.transaction_cost_bps

    panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
    panel_norm = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_normalized.parquet")
    splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
    for df in (panel, panel_norm, splits):
        df["date"] = pd.to_datetime(df["date"])
    panel = panel.merge(splits, on=["ticker", "date"], how="inner")
    panel_norm = panel_norm.merge(splits, on=["ticker", "date"], how="inner")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel_norm = panel_norm.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.dropna(subset=[ret_col, dir_col])
    panel_norm = panel_norm.dropna(subset=[ret_col, dir_col])

    ticker_cols = _add_ticker_encoding(panel)
    _add_ticker_encoding(panel_norm)

    results = []
    for fs_name in feat_defs["feature_sets"]:
        feat_ids = _resolve_feature_ids(feat_defs, fs_name, panel.columns.tolist())
        if not feat_ids:
            continue
        feat_ids_full = feat_ids + ticker_cols
        d = _split_arrays(panel, feat_ids_full, ret_col, dir_col)
        d_norm = _split_arrays(panel_norm, feat_ids_full, ret_col, dir_col)
        d_tft = _split_arrays(panel_norm, feat_ids, ret_col, dir_col)
        dates = d["dates_test"]

        # Benchmarks for set C
        if fs_name == "C":
            from src.models.benchmarks import run_benchmarks
            bench = run_benchmarks(
                d["X_train"], d["y_train_ret"], d["y_train_dir"],
                d["X_test"], d["y_test_ret"], d["y_test_dir"],
                feature_names=feat_ids_full,
            )
            bench["feature_set"] = fs_name
            bench["horizon"] = primary_horizon
            results.append(bench)

        for label, loader, data in [
            ("XGBoost", _load_xgb_predictions, d),
            ("FFNN", _load_ffnn_predictions, d_norm),
            ("TabNet", _load_tabnet_predictions, d_norm),
        ]:
            p = loader(fs_name, data)
            if p is None:
                print(f"[skip] {label}_{fs_name}")
                continue
            m = evaluate_model(
                f"{label}_{fs_name}", d["y_test_ret"], d["y_test_dir"],
                p["reg_pred"], p["clf_pred"], p["clf_proba"],
                d["y_train_ret"].mean(), tc,
                dates=dates, horizon=primary_horizon,
            )
            m["feature_set"], m["horizon"] = fs_name, primary_horizon
            results.append(pd.DataFrame([m]))
            print(f"[ok]   {label}_{fs_name}: R2={m['oos_r2']:+.4f}, IC={m['ic']:+.4f}, "
                  f"acc={m['accuracy']:.3f}, sharpe={m['sharpe_ratio']:+.2f}")

        # TFT
        p = _load_tft_predictions(fs_name, d_tft, vars(cfg.models.tft))
        if p is not None:
            pred_dir = (p["preds"] > 0).astype(int)
            m = evaluate_model(
                f"TFT_{fs_name}", p["y_test"], p["y_dir"],
                p["preds"], pred_dir, None,
                d_tft["y_train_ret"].mean(), tc,
                dates=p["dates"], horizon=primary_horizon,
            )
            m["feature_set"], m["horizon"] = fs_name, primary_horizon
            results.append(pd.DataFrame([m]))
            print(f"[ok]   TFT_{fs_name}: R2={m['oos_r2']:+.4f}, IC={m['ic']:+.4f}, "
                  f"acc={m['accuracy']:.3f}, sharpe={m['sharpe_ratio']:+.2f}")
        else:
            print(f"[skip] TFT_{fs_name}")

    # Carry over Set Transformer rows from existing CSV (single-ticker; sensible already)
    prior_path = RESULTS_DIR / "tables" / "model_comparison.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        st_rows = prior[prior["model"].str.startswith("SetTrans", na=False)]
        if len(st_rows):
            results.append(st_rows)
            print(f"[carry] {len(st_rows)} SetTransformer rows preserved")

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(prior_path, index=False)
    print(f"Saved {len(combined)} rows to {prior_path}")


if __name__ == "__main__":
    main()
