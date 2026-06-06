"""Compare feature importance across explanation methods."""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def cross_method_consensus(
    importance_dfs: dict[str, pd.DataFrame],
    feature_col: str = "feature",
    importance_col: str = "mean_abs_shap",
) -> pd.DataFrame:
    """Compare feature rankings across multiple explainability methods.

    Parameters
    ----------
    importance_dfs : dict mapping method_name -> DataFrame with
        feature names and importance scores. Each DataFrame must have
        columns: `feature_col` and an importance column.
    feature_col : column name for feature identifiers.
    importance_col : column name for importance scores (varies by method).

    Returns
    -------
    DataFrame with per-feature ranks across methods and a consensus rank.
    """
    all_features = set()
    for df in importance_dfs.values():
        all_features.update(df[feature_col].tolist())
    all_features = sorted(all_features)

    rank_data = {}
    for method, df in importance_dfs.items():
        # Determine which importance column to use
        imp_cols = [c for c in df.columns if c != feature_col]
        # Use the first numeric column as importance
        for c in imp_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                imp_col = c
                break
        else:
            continue

        df_sorted = df.sort_values(imp_col, ascending=False).reset_index(drop=True)
        feature_to_rank = {
            row[feature_col]: i + 1 for i, row in df_sorted.iterrows()
        }
        rank_data[f"rank_{method}"] = [
            feature_to_rank.get(f, len(all_features)) for f in all_features
        ]

    result = pd.DataFrame({"feature": all_features, **rank_data})

    # Consensus: average rank across methods
    rank_cols = [c for c in result.columns if c.startswith("rank_")]
    result["avg_rank"] = result[rank_cols].mean(axis=1)
    result["rank_std"] = result[rank_cols].std(axis=1)
    result = result.sort_values("avg_rank")

    # Rank correlation matrix
    log.info("=== Cross-Method Rank Correlations ===")
    for i, m1 in enumerate(rank_cols):
        for m2 in rank_cols[i + 1:]:
            corr, _ = spearmanr(result[m1], result[m2])
            log.info("  %s vs %s: rho=%.3f", m1, m2, corr)

    # Top consensus features
    log.info("Top 10 consensus features:")
    for _, row in result.head(10).iterrows():
        ranks = [f"{row[c]:.0f}" for c in rank_cols]
        log.info("  %s: avg_rank=%.1f, ranks=%s", row["feature"], row["avg_rank"], ranks)

    return result


def save_consensus(result: pd.DataFrame, name: str = "consensus") -> None:
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f"feature_{name}.csv", index=False)
