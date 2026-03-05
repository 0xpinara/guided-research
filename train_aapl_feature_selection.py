#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


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


def make_regressor() -> XGBRegressor:
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


def evaluate_model(
    model_name: str,
    cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> dict:
    X_train = train_df[cols].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_val = val_df[cols].to_numpy(dtype=float)
    y_val = val_df[target].to_numpy(dtype=float)
    X_test = test_df[cols].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    reg = make_regressor()
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_val = reg.predict(X_val)
    pred_test = reg.predict(X_test)

    val_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    test_mae = float(mean_absolute_error(y_test, pred_test))
    test_r2 = float(r2_score(y_test, pred_test))
    test_oos = oos_r2(y_test, pred_test)
    dir_acc = float(accuracy_score((y_test > 0).astype(int), (pred_test > 0).astype(int)))
    strat_cap, bh_cap = simulate_pnl(1000.0, y_test, pred_test)

    return {
        "model": model_name,
        "n_features": len(cols),
        "features": ",".join(cols),
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_oos_r2": test_oos,
        "test_direction_accuracy_from_reg_sign": dir_acc,
        "test_pnl_final_capital_long_flat_usd": strat_cap,
        "test_buy_hold_final_capital_usd": bh_cap,
    }


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_dir = root / "data/processed/training_ready"
    out_dir = root / "results/aapl_feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(in_dir / "aapl_train_no_sentiment.parquet")
    val_df = pd.read_parquet(in_dir / "aapl_val_no_sentiment.parquet")
    test_df = pd.read_parquet(in_dir / "aapl_test_no_sentiment.parquet")

    options_cols = get_feature_cols(train_df, 1, 26)
    stock_cols = get_feature_cols(train_df, 27, 42)
    target = "target_r_t_plus_1"

    # Baselines for comparison (B and C)
    baseline_rows = [
        evaluate_model("Baseline_B_options_01_26", options_cols, train_df, val_df, test_df, target),
        evaluate_model("Baseline_C_all_01_48", get_feature_cols(train_df, 1, 48), train_df, val_df, test_df, target),
    ]

    # 1) XGBoost importance on training set only
    selector = make_regressor()
    X_tr = train_df[options_cols].to_numpy(dtype=float)
    y_tr = train_df[target].to_numpy(dtype=float)
    X_va = val_df[options_cols].to_numpy(dtype=float)
    y_va = val_df[target].to_numpy(dtype=float)
    selector.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    imps = pd.Series(selector.feature_importances_, index=options_cols).sort_values(ascending=False)
    imps.to_csv(out_dir / "xgb_options_feature_importance_train_only.csv", header=["importance"])

    results_rows: list[dict] = []

    # 5-8 features sweep as requested
    for k in [5, 6, 7, 8]:
        sel = imps.head(k).index.tolist()
        results_rows.append(
            evaluate_model(
                f"Bprime_XGB_top{k}_options_only",
                sel,
                train_df,
                val_df,
                test_df,
                target,
            )
        )
        results_rows.append(
            evaluate_model(
                f"Cprime_XGB_top{k}_stock_plus_options",
                sorted(stock_cols + sel),
                train_df,
                val_df,
                test_df,
                target,
            )
        )

    # 2) L1 regularized linear regression selection on training only
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=5, random_state=42, max_iter=20000)),
        ]
    )
    pipe.fit(train_df[options_cols].to_numpy(dtype=float), y_tr)
    coefs = pd.Series(pipe.named_steps["lasso"].coef_, index=options_cols)
    nz = coefs[coefs.abs() > 1e-10].sort_values(key=np.abs, ascending=False)
    if len(nz) == 0:
        # fallback to top 5 absolute coefs if all shrunk to zero
        nz = coefs.sort_values(key=np.abs, ascending=False).head(5)
    nz.to_csv(out_dir / "lasso_options_coefficients_train_only.csv", header=["coef"])
    l1_sel = nz.index.tolist()

    results_rows.append(
        evaluate_model("Bprime_L1_nonzero_options_only", l1_sel, train_df, val_df, test_df, target)
    )
    results_rows.append(
        evaluate_model(
            "Cprime_L1_stock_plus_nonzero_options",
            sorted(stock_cols + l1_sel),
            train_df,
            val_df,
            test_df,
            target,
        )
    )

    baseline_df = pd.DataFrame(baseline_rows)
    results_df = pd.DataFrame(results_rows)
    combined = pd.concat([baseline_df, results_df], ignore_index=True)
    combined.to_csv(out_dir / "feature_selection_model_comparison.csv", index=False)

    # Best picks by validation RMSE among B' and C'
    bprime = results_df[results_df["model"].str.startswith("Bprime_")].sort_values("val_rmse")
    cprime = results_df[results_df["model"].str.startswith("Cprime_")].sort_values("val_rmse")
    summary = {
        "top8_options_by_xgb_train_only": imps.head(8).index.tolist(),
        "lasso_nonzero_options_train_only": l1_sel,
        "best_bprime_by_val_rmse": bprime.iloc[0].to_dict() if len(bprime) else {},
        "best_cprime_by_val_rmse": cprime.iloc[0].to_dict() if len(cprime) else {},
        "outputs": {
            "comparison_csv": str(out_dir / "feature_selection_model_comparison.csv"),
            "xgb_importance_csv": str(out_dir / "xgb_options_feature_importance_train_only.csv"),
            "lasso_coef_csv": str(out_dir / "lasso_options_coefficients_train_only.csv"),
        },
    }
    (out_dir / "feature_selection_summary.json").write_text(json.dumps(summary, indent=2))

    print("=== Feature Selection Comparison (sorted by val_rmse) ===")
    print(combined.sort_values("val_rmse")[["model", "n_features", "val_rmse", "test_rmse", "test_oos_r2", "test_direction_accuracy_from_reg_sign", "test_pnl_final_capital_long_flat_usd"]].to_string(index=False))
    print("\nTop-8 options by XGB (train-only):")
    print(imps.head(8).to_string())
    print("\nL1 non-zero selected options (train-only):")
    print(", ".join(l1_sel))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
