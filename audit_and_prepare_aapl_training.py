#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_path = root / "data/processed/aapl_features_daily_from_options_repo.parquet"
    out_dir = root / "data/processed/training_ready"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {in_path}")

    df = pd.read_parquet(in_path).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    feature_cols_no_sent = [c for c in feature_cols if c != "feat_49_news_sentiment_score"]
    target_cols = ["target_r_t_plus_1", "target_direction_t_plus_1"]

    # Replace infinities and enforce numeric parse for feature/target columns.
    for c in feature_cols + target_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Basic anomaly checks.
    anomalies = {
        "dup_dates": int(df["date"].duplicated().sum()),
        "bad_feat_24_outside_0_1_5": int(((df["feat_24_0dte_volume_share"] < 0) | (df["feat_24_0dte_volume_share"] > 1.5)).fillna(False).sum()),
        "bad_feat_18_negative_spread": int((df["feat_18_volume_weighted_avg_spread"] < 0).fillna(False).sum()),
        "bad_feat_06_iv_nonpositive_or_gt5": int(((df["feat_06_atm_iv"] <= 0) | (df["feat_06_atm_iv"] > 5)).fillna(False).sum()),
    }

    # Clean strategy for phase-1 modeling (no sentiment):
    # 1) Keep non-null date.
    # 2) Drop rows with any NaN in features 1..48 or targets.
    before_rows = len(df)
    df = df[df["date"].notna()].copy()
    required_cols = feature_cols_no_sent + target_cols
    clean = df.dropna(subset=required_cols).copy()
    after_rows = len(clean)

    # Temporal splits.
    train_df, val_df, test_df = temporal_split(clean, train_frac=0.6, val_frac=0.2)

    # Save outputs.
    clean_path = out_dir / "aapl_model_dataset_no_sentiment_clean.parquet"
    train_path = out_dir / "aapl_train_no_sentiment.parquet"
    val_path = out_dir / "aapl_val_no_sentiment.parquet"
    test_path = out_dir / "aapl_test_no_sentiment.parquet"

    clean.to_parquet(clean_path, index=False)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    report = {
        "input_file": str(in_path),
        "rows_before_cleaning": before_rows,
        "rows_after_cleaning_no_sentiment": after_rows,
        "rows_removed": before_rows - after_rows,
        "date_min": str(clean["date"].min()) if len(clean) else None,
        "date_max": str(clean["date"].max()) if len(clean) else None,
        "feature_count_total": len(feature_cols),
        "feature_count_used_no_sentiment": len(feature_cols_no_sent),
        "anomalies": anomalies,
        "split_rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "split_date_ranges": {
            "train": [str(train_df["date"].min()), str(train_df["date"].max())] if len(train_df) else [None, None],
            "val": [str(val_df["date"].min()), str(val_df["date"].max())] if len(val_df) else [None, None],
            "test": [str(test_df["date"].min()), str(test_df["date"].max())] if len(test_df) else [None, None],
        },
        "output_files": {
            "clean": str(clean_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    report_path = out_dir / "aapl_data_quality_report_no_sentiment.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
