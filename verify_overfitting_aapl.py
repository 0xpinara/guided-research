#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def get_feature_cols(df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    cols: list[str] = []
    for i in range(start_idx, end_idx + 1):
        prefix = f"feat_{i:02d}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    return sorted(cols)


def simulate_long_flat(initial_capital: float, returns: np.ndarray, signal: np.ndarray, cost_bps: float = 0.0) -> float:
    capital = float(initial_capital)
    prev_sig = 0
    for r, sig in zip(returns, signal, strict=False):
        s = int(sig)
        # transaction cost applied on position change
        if s != prev_sig and cost_bps > 0:
            capital *= 1.0 - (cost_bps / 10000.0)
        capital *= 1.0 + float(r) * s
        prev_sig = s
    return float(capital)


def model_cfg() -> dict:
    # Conservative config for diagnostics speed/stability.
    return {
        "objective": "binary:logistic",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 4,
    }


def choose_threshold_by_val_pnl(prob: np.ndarray, returns: np.ndarray) -> float:
    best_t = 0.5
    best_cap = -np.inf
    for t in np.linspace(0.30, 0.70, 81):
        sig = (prob >= t).astype(int)
        cap = simulate_long_flat(1000.0, returns, sig, cost_bps=0.0)
        if cap > best_cap:
            best_cap = cap
            best_t = float(t)
    return best_t


def evaluate_window(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> dict:
    Xtr = train_df[cols].to_numpy(dtype=float)
    Xva = val_df[cols].to_numpy(dtype=float)
    Xte = test_df[cols].to_numpy(dtype=float)
    ytr = train_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    yva = val_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    yte = test_df["target_direction_t_plus_1"].to_numpy(dtype=int)
    rva = val_df["target_r_t_plus_1"].to_numpy(dtype=float)
    rte = test_df["target_r_t_plus_1"].to_numpy(dtype=float)

    clf = XGBClassifier(**model_cfg())
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    ptr = clf.predict_proba(Xtr)[:, 1]
    pva = clf.predict_proba(Xva)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    thr = choose_threshold_by_val_pnl(pva, rva)

    str_sig = (ptr >= thr).astype(int)
    ste_sig = (pte >= thr).astype(int)

    return {
        "train_auc": float(roc_auc_score(ytr, ptr)),
        "val_auc": float(roc_auc_score(yva, pva)),
        "test_auc": float(roc_auc_score(yte, pte)),
        "train_acc": float(accuracy_score(ytr, str_sig)),
        "test_acc": float(accuracy_score(yte, ste_sig)),
        "thr": float(thr),
        "test_pnl_0bps": float(simulate_long_flat(1000.0, rte, ste_sig, cost_bps=0.0)),
        "test_pnl_5bps": float(simulate_long_flat(1000.0, rte, ste_sig, cost_bps=5.0)),
        "bh_pnl": float(simulate_long_flat(1000.0, rte, np.ones_like(rte), cost_bps=0.0)),
    }


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    df = pd.read_parquet(root / "data/processed/training_ready/aapl_model_dataset_no_sentiment_clean.parquet").copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Feature sets
    sets = {
        "A_stock_only": get_feature_cols(df, 27, 42),
        "B_options_only": get_feature_cols(df, 1, 26),
        "C_all_01_48": get_feature_cols(df, 1, 48),
    }

    # Rolling windows: ~3y train, 6m val, 6m test, step 3m
    train_len = 756
    val_len = 126
    test_len = 126
    step = 63

    rows: list[dict] = []
    for start in range(0, len(df) - (train_len + val_len + test_len) + 1, step):
        tr = df.iloc[start : start + train_len].copy()
        va = df.iloc[start + train_len : start + train_len + val_len].copy()
        te = df.iloc[start + train_len + val_len : start + train_len + val_len + test_len].copy()
        for name, cols in sets.items():
            out = evaluate_window(tr, va, te, cols)
            out.update(
                {
                    "window_start": str(tr["date"].min().date()),
                    "window_end": str(te["date"].max().date()),
                    "model": name,
                    "n_features": len(cols),
                    "train_rows": len(tr),
                    "val_rows": len(va),
                    "test_rows": len(te),
                }
            )
            rows.append(out)

    res = pd.DataFrame(rows)
    out_dir = root / "results/aapl_overfitting_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "rolling_window_diagnostics.csv", index=False)

    # Aggregate diagnostics
    grp = res.groupby("model")
    summary = grp.agg(
        windows=("model", "size"),
        train_auc_mean=("train_auc", "mean"),
        test_auc_mean=("test_auc", "mean"),
        auc_gap_mean=("train_auc", lambda s: float(np.mean(s - res.loc[s.index, "test_auc"]))),
        test_acc_mean=("test_acc", "mean"),
        test_pnl_0bps_median=("test_pnl_0bps", "median"),
        test_pnl_5bps_median=("test_pnl_5bps", "median"),
        bh_pnl_median=("bh_pnl", "median"),
    ).reset_index()

    # consistency checks
    pivot = res.pivot_table(index=["window_start", "window_end"], columns="model", values="test_pnl_5bps")
    c_beats_a = float((pivot["C_all_01_48"] > pivot["A_stock_only"]).mean())
    b_beats_a = float((pivot["B_options_only"] > pivot["A_stock_only"]).mean())

    verdict = {
        "c_beats_a_fraction_windows_pnl_5bps": c_beats_a,
        "b_beats_a_fraction_windows_pnl_5bps": b_beats_a,
        "models_with_large_auc_gap_flag": summary.loc[summary["auc_gap_mean"] > 0.10, "model"].tolist(),
    }

    summary.to_csv(out_dir / "rolling_window_summary.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("=== Rolling Window Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Overfitting Verdict ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
