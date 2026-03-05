#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


def get_feature_cols(df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    cols: list[str] = []
    for i in range(start_idx, end_idx + 1):
        prefix = f"feat_{i:02d}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    return sorted(cols)


def simulate_long_flat(initial_capital: float, returns: np.ndarray, signal: np.ndarray) -> float:
    capital = float(initial_capital)
    for r, s in zip(returns, signal, strict=False):
        capital *= 1.0 + float(r) * float(s)
    return float(capital)


def make_selector_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
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


def classifier_grid() -> list[dict]:
    grid: list[dict] = []
    for max_depth in [2, 3, 4]:
        for lr in [0.02, 0.03, 0.05]:
            for subsample in [0.7, 0.85]:
                grid.append(
                    {
                        "objective": "binary:logistic",
                        "n_estimators": 1800,
                        "learning_rate": lr,
                        "max_depth": max_depth,
                        "subsample": subsample,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.0,
                        "reg_lambda": 2.0 if max_depth >= 4 else 1.0,
                        "random_state": 42,
                        "n_jobs": 4,
                    }
                )
    return grid


def fit_best_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[XGBClassifier, dict, np.ndarray]:
    best_model: XGBClassifier | None = None
    best_cfg: dict | None = None
    best_auc = -np.inf
    best_val_prob: np.ndarray | None = None

    for cfg in classifier_grid():
        clf = XGBClassifier(**cfg)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, prob))
        if auc > best_auc:
            best_auc = auc
            best_model = clf
            best_cfg = cfg
            best_val_prob = prob

    assert best_model is not None and best_cfg is not None and best_val_prob is not None
    return best_model, best_cfg, best_val_prob


def choose_threshold_for_pnl(val_prob: np.ndarray, val_returns: np.ndarray) -> tuple[float, float]:
    best_thr = 0.5
    best_cap = -np.inf
    for thr in np.linspace(0.30, 0.70, 81):
        signal = (val_prob >= thr).astype(int)
        cap = simulate_long_flat(1000.0, val_returns, signal)
        if cap > best_cap:
            best_cap = cap
            best_thr = float(thr)
    return best_thr, float(best_cap)


def evaluate(
    name: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)

    y_train_cls = train_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    y_val_cls = val_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    y_test_cls = test_df["target_direction_t_plus_1"].to_numpy(dtype=int)

    y_val_ret = val_df["target_r_t_plus_1"].to_numpy(dtype=float)
    y_test_ret = test_df["target_r_t_plus_1"].to_numpy(dtype=float)

    model, cfg, val_prob = fit_best_classifier(X_train, y_train_cls, X_val, y_val_cls)
    test_prob = model.predict_proba(X_test)[:, 1]

    thr_pnl, val_cap = choose_threshold_for_pnl(val_prob, y_val_ret)
    test_signal = (test_prob >= thr_pnl).astype(int)
    test_cap = simulate_long_flat(1000.0, y_test_ret, test_signal)
    bh_cap = simulate_long_flat(1000.0, y_test_ret, np.ones_like(y_test_ret))

    test_pred = test_signal
    return {
        "model": name,
        "n_features": len(feature_cols),
        "threshold_selected_on_val_for_pnl": thr_pnl,
        "val_pnl_capital": float(val_cap),
        "test_pnl_capital": float(test_cap),
        "buy_hold_test_capital": float(bh_cap),
        "test_accuracy": float(accuracy_score(y_test_cls, test_pred)),
        "test_precision": float(precision_score(y_test_cls, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test_cls, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test_cls, test_pred, zero_division=0)),
        "test_auc": float(roc_auc_score(y_test_cls, test_prob)),
        "best_val_auc": float(roc_auc_score(y_val_cls, val_prob)),
        "best_config": json.dumps(cfg),
        "features": ",".join(feature_cols),
    }


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_dir = root / "data/processed/training_ready"
    out_dir = root / "results/aapl_model_improvement"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(in_dir / "aapl_train_no_sentiment.parquet")
    val_df = pd.read_parquet(in_dir / "aapl_val_no_sentiment.parquet")
    test_df = pd.read_parquet(in_dir / "aapl_test_no_sentiment.parquet")

    stock_cols = get_feature_cols(train_df, 27, 42)
    options_cols = get_feature_cols(train_df, 1, 26)
    all_cols = get_feature_cols(train_df, 1, 48)

    # Train-only options feature selection (XGB importance on regression target).
    selector = make_selector_regressor()
    selector.fit(
        train_df[options_cols].to_numpy(dtype=float),
        train_df["target_r_t_plus_1"].to_numpy(dtype=float),
        eval_set=[(val_df[options_cols].to_numpy(dtype=float), val_df["target_r_t_plus_1"].to_numpy(dtype=float))],
        verbose=False,
    )
    imps = pd.Series(selector.feature_importances_, index=options_cols).sort_values(ascending=False)
    imps.to_csv(out_dir / "options_importance_train_only.csv", header=["importance"])

    candidates: dict[str, list[str]] = {
        "A_stock_only": stock_cols,
        "B_options_only": options_cols,
        "C_all_features": all_cols,
    }
    for k in [5, 8, 12]:
        topk = imps.head(k).index.tolist()
        candidates[f"Bprime_top{k}_options"] = topk
        candidates[f"Cprime_top{k}_stock_plus_options"] = sorted(stock_cols + topk)

    rows: list[dict] = []
    for name, cols in candidates.items():
        rows.append(evaluate(name, cols, train_df, val_df, test_df))

    res = pd.DataFrame(rows).sort_values(["test_pnl_capital", "test_accuracy"], ascending=False)
    res.to_csv(out_dir / "improvement_comparison.csv", index=False)

    best = res.iloc[0].to_dict()
    summary = {
        "goal": "Beat stock-only with options or stock+options",
        "best_model_by_test_pnl": best["model"],
        "best_test_pnl_capital": float(best["test_pnl_capital"]),
        "stock_only_test_pnl_capital": float(res.loc[res["model"] == "A_stock_only", "test_pnl_capital"].iloc[0]),
        "best_test_accuracy": float(best["test_accuracy"]),
        "stock_only_test_accuracy": float(res.loc[res["model"] == "A_stock_only", "test_accuracy"].iloc[0]),
        "top12_options_features_train_only": imps.head(12).index.tolist(),
    }
    (out_dir / "improvement_summary.json").write_text(json.dumps(summary, indent=2))

    print("=== Improvement Results (sorted by test P&L) ===")
    print(
        res[
            [
                "model",
                "n_features",
                "threshold_selected_on_val_for_pnl",
                "test_pnl_capital",
                "buy_hold_test_capital",
                "test_accuracy",
                "test_auc",
            ]
        ].to_string(index=False)
    )
    print("\nTop 12 options features (train-only importance):")
    print(imps.head(12).to_string())
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
