#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
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


def temporal_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def reg_model() -> XGBRegressor:
    return XGBRegressor(
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


def cls_model() -> XGBClassifier:
    return XGBClassifier(
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


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_path = root / "data/processed/aapl_features_daily_from_options_repo.parquet"
    underlying_path = root / "data/raw/options_data_repo/aapl/underlying.parquet"
    out_dir = root / "results/aapl_horizons"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # No-sentiment feature set
    feat_no_sent = [c for c in sorted(df.columns) if c.startswith("feat_") and c != "feat_49_news_sentiment_score"]
    # Consistent subset from report for Model A/B/C
    feats_a = get_feature_cols(df, 27, 42)
    feats_b = get_feature_cols(df, 1, 26)
    feats_c = get_feature_cols(df, 1, 48)
    feature_sets = {
        "Model_A_stock_only_27_42": feats_a,
        "Model_B_options_only_01_26": feats_b,
        "Model_C_all_01_48": feats_c,
    }

    horizon_results_reg: list[dict] = []
    horizon_results_cls: list[dict] = []

    under = pd.read_parquet(underlying_path)[["date", "close"]].copy()
    under["date"] = pd.to_datetime(under["date"], utc=True, errors="coerce")
    under = under.sort_values("date").drop_duplicates("date")

    for h in [3, 5]:
        work = df.copy()
        source = under.copy()
        source[f"target_r_t_plus_{h}"] = source["close"].shift(-h) / source["close"] - 1.0
        target_map = dict(zip(source["date"], source[f"target_r_t_plus_{h}"]))
        work[f"target_r_t_plus_{h}"] = work["date"].map(target_map)
        work[f"target_direction_t_plus_{h}"] = (work[f"target_r_t_plus_{h}"] > 0).astype("Int64")

        required = feat_no_sent + [f"target_r_t_plus_{h}", f"target_direction_t_plus_{h}"]
        clean = work.dropna(subset=required).copy()
        train_df, val_df, test_df = temporal_split(clean, 0.6, 0.2)

        y_test_reg = test_df[f"target_r_t_plus_{h}"].to_numpy(dtype=float)
        y_zero = np.zeros_like(y_test_reg)
        horizon_results_reg.append(
            {
                "horizon_days": h,
                "model": "Naive_predict_zero",
                "rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_zero))),
                "mae": float(mean_absolute_error(y_test_reg, y_zero)),
                "r2": float(r2_score(y_test_reg, y_zero)),
                "oos_r2": oos_r2(y_test_reg, y_zero),
                "direction_accuracy_from_reg_sign": float(accuracy_score((y_test_reg > 0).astype(int), np.zeros_like(y_test_reg, dtype=int))),
                "n_features": 0,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
        )

        for model_name, cols in feature_sets.items():
            X_train = train_df[cols].to_numpy(dtype=float)
            y_train_reg = train_df[f"target_r_t_plus_{h}"].to_numpy(dtype=float)
            X_val = val_df[cols].to_numpy(dtype=float)
            y_val_reg = val_df[f"target_r_t_plus_{h}"].to_numpy(dtype=float)
            X_test = test_df[cols].to_numpy(dtype=float)

            reg = reg_model()
            reg.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], verbose=False)
            pred_reg = reg.predict(X_test)

            horizon_results_reg.append(
                {
                    "horizon_days": h,
                    "model": model_name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test_reg, pred_reg))),
                    "mae": float(mean_absolute_error(y_test_reg, pred_reg)),
                    "r2": float(r2_score(y_test_reg, pred_reg)),
                    "oos_r2": oos_r2(y_test_reg, pred_reg),
                    "direction_accuracy_from_reg_sign": float(accuracy_score((y_test_reg > 0).astype(int), (pred_reg > 0).astype(int))),
                    "n_features": len(cols),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                }
            )

            y_train_cls = train_df[f"target_direction_t_plus_{h}"].to_numpy(dtype=int)
            y_val_cls = val_df[f"target_direction_t_plus_{h}"].to_numpy(dtype=int)
            y_test_cls = test_df[f"target_direction_t_plus_{h}"].to_numpy(dtype=int)
            clf = cls_model()
            clf.fit(X_train, y_train_cls, eval_set=[(X_val, y_val_cls)], verbose=False)
            prob = clf.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            horizon_results_cls.append(
                {
                    "horizon_days": h,
                    "model": model_name,
                    "accuracy": float(accuracy_score(y_test_cls, pred)),
                    "auc": float(roc_auc_score(y_test_cls, prob)),
                    "n_features": len(cols),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                }
            )

    reg_df = pd.DataFrame(horizon_results_reg)
    cls_df = pd.DataFrame(horizon_results_cls)
    reg_df.to_csv(out_dir / "horizon_regression_summary.csv", index=False)
    cls_df.to_csv(out_dir / "horizon_classification_summary.csv", index=False)

    # Convenience comparison: best A/B/C by horizon
    pivot_reg = reg_df[reg_df["model"] != "Naive_predict_zero"].pivot_table(
        index="horizon_days",
        columns="model",
        values=["rmse", "oos_r2", "direction_accuracy_from_reg_sign"],
    )
    pivot_reg.to_csv(out_dir / "horizon_regression_comparison_pivot.csv")

    meta = {
        "input_file": str(in_path),
        "output_dir": str(out_dir),
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "note": "Targets recomputed for 3-day and 5-day horizons; sentiment excluded.",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("=== 3D/5D Regression Summary ===")
    print(reg_df.to_string(index=False))
    print("\n=== 3D/5D Classification Summary ===")
    print(cls_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
