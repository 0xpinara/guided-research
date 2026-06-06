"""Extract and analyze TabNet feature attention masks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def analyze_tabnet_attention(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Extract and summarize TabNet attention masks.

    Returns
    -------
    dict with:
        masks : ndarray (n_samples, n_features)
        importance_df : DataFrame with mean attention per feature
        feature_importances : built-in TabNet importance
    """
    masks, _ = model.explain(X_test)
    feature_importances = model.feature_importances_

    mean_attention = masks.mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_attention": mean_attention,
        "tabnet_importance": feature_importances,
    }).sort_values("mean_attention", ascending=False)

    log.info("Top 10 TabNet attention features:")
    for _, row in importance_df.head(10).iterrows():
        log.info("  %s: attn=%.4f, imp=%.4f",
                 row["feature"], row["mean_attention"], row["tabnet_importance"])

    return {
        "masks": masks,
        "importance_df": importance_df,
    }


def save_tabnet_attention(results: dict, model_name: str) -> None:
    """Save attention results."""
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results["importance_df"].to_csv(
        out_dir / f"tabnet_attention_{model_name}.csv", index=False
    )
