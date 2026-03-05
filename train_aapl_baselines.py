#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor


def get_feature_cols(df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    cols: list[str] = []
    for i in range(start_idx, end_idx + 1):
        prefix = f"feat_{i:02d}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    return sorted(cols)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(y_true)))
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum(np.square(y_pred - y_true)) / denom)


def simulate_pnl(initial_capital: float, actual_returns: np.ndarray, pred_returns: np.ndarray) -> tuple[float, float]:
    capital_strategy = float(initial_capital)
    capital_bh = float(initial_capital)
    for r, p in zip(actual_returns, pred_returns, strict=False):
        signal = 1.0 if p > 0 else 0.0
        capital_strategy *= 1.0 + signal * float(r)
        capital_bh *= 1.0 + float(r)
    return capital_strategy, capital_bh


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_dir = root / "data/processed/training_ready"
    out_dir = root / "results/aapl_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(in_dir / "aapl_train_no_sentiment.parquet")
    val_df = pd.read_parquet(in_dir / "aapl_val_no_sentiment.parquet")
    test_df = pd.read_parquet(in_dir / "aapl_test_no_sentiment.parquet")
    for d in (train_df, val_df, test_df):
        d["date"] = pd.to_datetime(d["date"], utc=True)

    target_reg = "target_r_t_plus_1"
    target_cls = "target_direction_t_plus_1"

    feature_sets = {
        "Model_A_stock_only_27_42": get_feature_cols(train_df, 27, 42),
        "Model_B_options_only_01_26": get_feature_cols(train_df, 1, 26),
        "Model_C_all_01_48": get_feature_cols(train_df, 1, 48),
    }

    reg_results: list[dict] = []
    cls_results: list[dict] = []

    # Naive baseline on test.
    y_test = test_df[target_reg].to_numpy(dtype=float)
    y_zero = np.zeros_like(y_test)
    naive_rmse = float(np.sqrt(mean_squared_error(y_test, y_zero)))
    naive_mae = float(mean_absolute_error(y_test, y_zero))
    naive_r2 = float(r2_score(y_test, y_zero))
    naive_oos = oos_r2(y_test, y_zero)
    naive_dir_pred = np.zeros_like(y_test, dtype=int)
    naive_dir_true = (y_test > 0).astype(int)
    naive_acc = float(accuracy_score(naive_dir_true, naive_dir_pred))
    naive_cap, bh_cap = simulate_pnl(1000.0, y_test, y_zero)
    reg_results.append(
        {
            "model": "Naive_predict_zero",
            "rmse": naive_rmse,
            "mae": naive_mae,
            "r2": naive_r2,
            "oos_r2": naive_oos,
            "direction_accuracy_from_reg_sign": naive_acc,
            "pnl_final_capital_long_flat_usd": naive_cap,
            "buy_hold_final_capital_usd": bh_cap,
            "n_features": 0,
        }
    )

    for model_name, cols in feature_sets.items():
        X_train = train_df[cols].to_numpy(dtype=float)
        y_train_reg = train_df[target_reg].to_numpy(dtype=float)
        X_val = val_df[cols].to_numpy(dtype=float)
        y_val_reg = val_df[target_reg].to_numpy(dtype=float)
        X_test = test_df[cols].to_numpy(dtype=float)
        y_test_reg = y_test

        # Regressor
        reg = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )
        reg.fit(
            X_train,
            y_train_reg,
            eval_set=[(X_val, y_val_reg)],
            verbose=False,
        )
        pred_test = reg.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test_reg, pred_test)))
        mae = float(mean_absolute_error(y_test_reg, pred_test))
        r2 = float(r2_score(y_test_reg, pred_test))
        oos = oos_r2(y_test_reg, pred_test)
        dir_acc = float(accuracy_score((y_test_reg > 0).astype(int), (pred_test > 0).astype(int)))
        strat_cap, bh_cap = simulate_pnl(1000.0, y_test_reg, pred_test)

        reg_results.append(
            {
                "model": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "oos_r2": oos,
                "direction_accuracy_from_reg_sign": dir_acc,
                "pnl_final_capital_long_flat_usd": strat_cap,
                "buy_hold_final_capital_usd": bh_cap,
                "n_features": len(cols),
            }
        )

        pred_df = pd.DataFrame(
            {
                "date": test_df["date"],
                "y_true_return": y_test_reg,
                "y_pred_return": pred_test,
                "signal_long_flat": (pred_test > 0).astype(int),
            }
        )
        pred_df.to_csv(out_dir / f"{model_name}_reg_test_predictions.csv", index=False)

        # Classifier
        y_train_cls = train_df[target_cls].to_numpy(dtype=int)
        y_val_cls = val_df[target_cls].to_numpy(dtype=int)
        y_test_cls = test_df[target_cls].to_numpy(dtype=int)
        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )
        clf.fit(
            X_train,
            y_train_cls,
            eval_set=[(X_val, y_val_cls)],
            verbose=False,
        )
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        cls_results.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test_cls, pred)),
                "precision": float(precision_score(y_test_cls, pred, zero_division=0)),
                "recall": float(recall_score(y_test_cls, pred, zero_division=0)),
                "f1": float(f1_score(y_test_cls, pred, zero_division=0)),
                "auc": float(roc_auc_score(y_test_cls, prob)),
                "n_features": len(cols),
            }
        )

        cls_pred_df = pd.DataFrame(
            {
                "date": test_df["date"],
                "y_true_direction": y_test_cls,
                "y_pred_direction": pred,
                "y_pred_prob_up": prob,
            }
        )
        cls_pred_df.to_csv(out_dir / f"{model_name}_cls_test_predictions.csv", index=False)

    reg_table = pd.DataFrame(reg_results)
    cls_table = pd.DataFrame(cls_results)
    reg_table.to_csv(out_dir / "regression_summary.csv", index=False)
    cls_table.to_csv(out_dir / "classification_summary.csv", index=False)

    meta = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "targets": {"regression": target_reg, "classification": target_cls},
        "outputs": {
            "regression_summary": str(out_dir / "regression_summary.csv"),
            "classification_summary": str(out_dir / "classification_summary.csv"),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("=== Regression Summary ===")
    print(reg_table.to_string(index=False))
    print("\n=== Classification Summary ===")
    print(cls_table.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
