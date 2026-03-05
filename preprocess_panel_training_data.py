#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def temporal_split_by_date(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(df["date"].dropna().unique()))
    n = len(unique_dates)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    train_dates = set(unique_dates[:i1])
    val_dates = set(unique_dates[i1:i2])
    test_dates = set(unique_dates[i2:])
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, val_df, test_df


def anomaly_counts(df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    if "feat_24_0dte_volume_share" in df.columns:
        x = df["feat_24_0dte_volume_share"]
        out["bad_feat_24_outside_0_1_5"] = int(((x < 0) | (x > 1.5)).fillna(False).sum())
    if "feat_18_volume_weighted_avg_spread" in df.columns:
        x = df["feat_18_volume_weighted_avg_spread"]
        out["bad_feat_18_negative_spread"] = int((x < 0).fillna(False).sum())
    if "feat_06_atm_iv" in df.columns:
        x = df["feat_06_atm_iv"]
        out["bad_feat_06_iv_nonpositive_or_gt5"] = int(((x <= 0) | (x > 5)).fillna(False).sum())
    if "feat_07_iv_rank_52w_percentile" in df.columns:
        x = df["feat_07_iv_rank_52w_percentile"]
        out["bad_feat_07_outside_0_1"] = int(((x < 0) | (x > 1)).fillna(False).sum())
    return out


def apply_domain_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Convert impossible values to NaN; these get imputed later.
    if "feat_24_0dte_volume_share" in df.columns:
        s = df["feat_24_0dte_volume_share"]
        df.loc[(s < 0) | (s > 1.5), "feat_24_0dte_volume_share"] = np.nan
    if "feat_18_volume_weighted_avg_spread" in df.columns:
        s = df["feat_18_volume_weighted_avg_spread"]
        df.loc[s < 0, "feat_18_volume_weighted_avg_spread"] = np.nan
    if "feat_06_atm_iv" in df.columns:
        s = df["feat_06_atm_iv"]
        df.loc[(s <= 0) | (s > 5), "feat_06_atm_iv"] = np.nan
    if "feat_07_iv_rank_52w_percentile" in df.columns:
        s = df["feat_07_iv_rank_52w_percentile"]
        df.loc[(s < 0) | (s > 1), "feat_07_iv_rank_52w_percentile"] = np.nan
    return df


def winsorize_with_train_quantiles(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    lower_q: float = 0.001,
    upper_q: float = 0.999,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    lo = train_df[feature_cols].quantile(lower_q, interpolation="linear")
    hi = train_df[feature_cols].quantile(upper_q, interpolation="linear")
    for c in feature_cols:
        l = lo.get(c, np.nan)
        h = hi.get(c, np.nan)
        if np.isfinite(l) and np.isfinite(h) and l <= h:
            train_df[c] = train_df[c].clip(lower=l, upper=h)
            val_df[c] = val_df[c].clip(lower=l, upper=h)
            test_df[c] = test_df[c].clip(lower=l, upper=h)
    return train_df, val_df, test_df


def impute_split_from_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Ticker-level medians from train only.
    ticker_medians = train_df.groupby("ticker")[feature_cols].median(numeric_only=True)
    global_medians = train_df[feature_cols].median(numeric_only=True)

    # Structural default for 0DTE share.
    if "feat_24_0dte_volume_share" in feature_cols:
        if "feat_24_0dte_volume_share" in ticker_medians.columns:
            ticker_medians["feat_24_0dte_volume_share"] = ticker_medians["feat_24_0dte_volume_share"].fillna(0.0)
        global_medians["feat_24_0dte_volume_share"] = float(
            0.0 if pd.isna(global_medians.get("feat_24_0dte_volume_share", np.nan)) else global_medians["feat_24_0dte_volume_share"]
        )

    def fill_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in feature_cols:
            med_by_ticker = out["ticker"].map(ticker_medians[c]) if c in ticker_medians.columns else pd.Series(np.nan, index=out.index)
            out[c] = out[c].fillna(med_by_ticker)
            out[c] = out[c].fillna(global_medians.get(c, np.nan))
            # Final fallback if a feature is entirely NaN in train.
            out[c] = out[c].fillna(0.0)
        return out

    train_f = fill_df(train_df)
    val_f = fill_df(val_df)
    test_f = fill_df(test_df)

    impute_meta = {
        "features_with_all_nan_in_train": sorted([c for c in feature_cols if train_df[c].notna().sum() == 0]),
        "global_median_snapshot": {k: (None if pd.isna(v) else float(v)) for k, v in global_medians.to_dict().items()},
    }
    return train_f, val_f, test_f, impute_meta


def main() -> None:
    root = Path("/Users/pa/Desktop/guided")
    in_path = root / "data/processed/panel/panel_features_all_tickers.parquet"
    out_dir = root / "data/processed/panel/training_ready"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {in_path}")

    df = pd.read_parquet(in_path).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = sorted(
        [c for c in df.columns if c.startswith("feat_") and c != "feat_49_news_sentiment_score"]
    )
    target_cols = ["target_r_t_plus_1", "target_direction_t_plus_1"]
    id_cols = ["date", "ticker"]

    # Numeric coercion and inf cleanup.
    for c in feature_cols + target_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before_rows = len(df)
    before_nan_total = int(df[feature_cols + target_cols].isna().sum().sum())
    before_anom = anomaly_counts(df)

    # Base row filters: keep valid id + targets.
    df = df[df["date"].notna() & df["ticker"].notna()].copy()
    df = df.dropna(subset=target_cols).copy()

    # Domain cleanup.
    df = apply_domain_cleanup(df)
    after_cleanup_anom = anomaly_counts(df)

    # Time split before imputation (no leakage).
    train_df, val_df, test_df = temporal_split_by_date(df, train_frac=0.6, val_frac=0.2)

    # Winsorize then impute using train-only statistics.
    train_df, val_df, test_df = winsorize_with_train_quantiles(train_df, val_df, test_df, feature_cols)
    train_df, val_df, test_df, impute_meta = impute_split_from_train(train_df, val_df, test_df, feature_cols)

    # Ensure no NaNs in required modeling columns after preprocessing.
    required_cols = id_cols + feature_cols + target_cols
    train_df = train_df.dropna(subset=required_cols).copy()
    val_df = val_df.dropna(subset=required_cols).copy()
    test_df = test_df.dropna(subset=required_cols).copy()

    clean_df = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    # Save outputs.
    clean_path = out_dir / "panel_model_dataset_no_sentiment_clean.parquet"
    train_path = out_dir / "panel_train_no_sentiment.parquet"
    val_path = out_dir / "panel_val_no_sentiment.parquet"
    test_path = out_dir / "panel_test_no_sentiment.parquet"
    report_path = out_dir / "panel_data_quality_report_no_sentiment.json"
    ticker_cov_path = out_dir / "panel_training_ticker_coverage.csv"

    clean_df.to_parquet(clean_path, index=False)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    cov = (
        clean_df.groupby("ticker", as_index=False)
        .agg(
            rows=("ticker", "size"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .sort_values("ticker")
    )
    cov.to_csv(ticker_cov_path, index=False)

    report = {
        "input_file": str(in_path),
        "rows_before_cleaning": int(before_rows),
        "rows_after_target_and_id_filter": int(len(df)),
        "rows_after_final_preprocessing": int(len(clean_df)),
        "rows_removed_total": int(before_rows - len(clean_df)),
        "feature_count_used_no_sentiment": len(feature_cols),
        "target_columns": target_cols,
        "nan_counts_before_total_features_targets": int(before_nan_total),
        "nan_counts_after_total_features_targets": int(
            clean_df[feature_cols + target_cols].isna().sum().sum()
        ),
        "anomalies_before": before_anom,
        "anomalies_after_domain_cleanup": after_cleanup_anom,
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_date_ranges": {
            "train": [str(train_df["date"].min()), str(train_df["date"].max())] if len(train_df) else [None, None],
            "val": [str(val_df["date"].min()), str(val_df["date"].max())] if len(val_df) else [None, None],
            "test": [str(test_df["date"].min()), str(test_df["date"].max())] if len(test_df) else [None, None],
        },
        "imputation": impute_meta,
        "output_files": {
            "clean": str(clean_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "ticker_coverage": str(ticker_cov_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
