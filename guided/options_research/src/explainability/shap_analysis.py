"""SHAP analysis for XGBoost and FFNN models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def xgb_shap(model, X_test: np.ndarray, feature_names: list[str]) -> dict:
    """Compute SHAP values for an XGBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return _summarize_shap(shap_values, feature_names, "xgboost")


def ffnn_shap(
    model, X_train_sample: np.ndarray, X_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compute SHAP values for an FFNN using DeepExplainer."""
    import torch
    device = next(model.parameters()).device
    model.eval()

    bg = torch.tensor(X_train_sample[:500], dtype=torch.float32).to(device)
    test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    explainer = shap.DeepExplainer(model, bg)
    shap_values = explainer.shap_values(test_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, "cpu"):
        shap_values = shap_values.cpu().numpy()

    return _summarize_shap(shap_values, feature_names, "ffnn")


def _summarize_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_type: str,
) -> dict:
    """Create summary DataFrames from SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    log.info("Top 10 features by |SHAP| (%s):", model_type)
    for _, row in importance_df.head(10).iterrows():
        log.info("  %s: %.6f", row["feature"], row["mean_abs_shap"])

    return {
        "shap_values": shap_values,
        "importance_df": importance_df,
    }


def save_shap_results(results: dict, model_name: str) -> None:
    """Save SHAP values and importance to disk."""
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["importance_df"].to_csv(
        out_dir / f"shap_importance_{model_name}.csv", index=False
    )
    np.save(
        out_dir / f"shap_values_{model_name}.npy",
        results["shap_values"],
    )
