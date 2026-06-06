"""Analyze Set Transformer contract-level attention weights."""

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def analyze_contract_attention(
    attn_weights: np.ndarray,
    contract_tensors: np.ndarray,
    masks: np.ndarray,
) -> dict:
    """Analyze which contracts the Set Transformer attends to.

    Parameters
    ----------
    attn_weights : ndarray, shape (n_samples, 1, max_contracts)
        PMA attention weights.
    contract_tensors : ndarray, shape (n_samples, max_contracts, 10)
        Contract features: [moneyness, dte_scaled, is_call, iv, ...].
    masks : ndarray, shape (n_samples, max_contracts)
        True for real contracts.

    Returns
    -------
    dict with attention-by-moneyness and attention-by-dte analysis.
    """
    attn = attn_weights.squeeze(1)  # (n_samples, max_contracts)

    # Moneyness bins
    moneyness = contract_tensors[:, :, 0]  # feature index 0
    dte_scaled = contract_tensors[:, :, 1]  # feature index 1
    is_call = contract_tensors[:, :, 2]     # feature index 2

    # Collect attention by moneyness bucket
    moneyness_bins = [(0.80, 0.90), (0.90, 0.95), (0.95, 1.00),
                       (1.00, 1.05), (1.05, 1.10), (1.10, 1.20)]
    moneyness_attn = {}
    for lo, hi in moneyness_bins:
        bucket_mask = (moneyness >= lo) & (moneyness < hi) & masks
        if bucket_mask.any():
            moneyness_attn[f"{lo:.2f}-{hi:.2f}"] = attn[bucket_mask].mean()
        else:
            moneyness_attn[f"{lo:.2f}-{hi:.2f}"] = 0.0

    # Attention by DTE bucket (dte_scaled is dte/365)
    dte_bins = [(7 / 365, 30 / 365), (30 / 365, 60 / 365),
                (60 / 365, 90 / 365), (90 / 365, 180 / 365)]
    dte_labels = ["7-30d", "30-60d", "60-90d", "90-180d"]
    dte_attn = {}
    for (lo, hi), label in zip(dte_bins, dte_labels):
        bucket_mask = (dte_scaled >= lo) & (dte_scaled < hi) & masks
        if bucket_mask.any():
            dte_attn[label] = attn[bucket_mask].mean()
        else:
            dte_attn[label] = 0.0

    # Calls vs puts
    call_mask = (is_call == 1) & masks
    put_mask = (is_call == 0) & masks
    call_attn = attn[call_mask].mean() if call_mask.any() else 0
    put_attn = attn[put_mask].mean() if put_mask.any() else 0

    moneyness_df = pd.DataFrame([
        {"moneyness_bucket": k, "mean_attention": v}
        for k, v in moneyness_attn.items()
    ])
    dte_df = pd.DataFrame([
        {"dte_bucket": k, "mean_attention": v}
        for k, v in dte_attn.items()
    ])

    log.info("Contract attention by moneyness: %s", moneyness_attn)
    log.info("Contract attention by DTE: %s", dte_attn)
    log.info("Call attention: %.4f, Put attention: %.4f", call_attn, put_attn)

    return {
        "moneyness_attention": moneyness_df,
        "dte_attention": dte_df,
        "call_attention": call_attn,
        "put_attention": put_attn,
    }


def save_contract_attention(results: dict, model_name: str) -> None:
    """Save contract attention analysis."""
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results["moneyness_attention"].to_csv(
        out_dir / f"contract_attn_moneyness_{model_name}.csv", index=False
    )
    results["dte_attention"].to_csv(
        out_dir / f"contract_attn_dte_{model_name}.csv", index=False
    )
