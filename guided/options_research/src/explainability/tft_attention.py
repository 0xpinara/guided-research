"""Extract and analyze TFT temporal attention patterns."""

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def analyze_tft_attention(
    temporal_attention: np.ndarray,
    lookback: int = 20,
) -> dict:
    """Analyze TFT temporal attention weights.

    Parameters
    ----------
    temporal_attention : ndarray, shape (n_samples, 1, lookback)
        Attention weights from TFT's self-attention over the time dimension.
    lookback : int
        Number of lookback timesteps.

    Returns
    -------
    dict with mean attention profile and recency analysis.
    """
    # Squeeze query dimension
    attn = temporal_attention.squeeze(1)  # (n_samples, lookback)

    # Mean attention across all samples
    mean_attn = attn.mean(axis=0)

    # How much attention goes to recent vs. distant past
    recent_5 = mean_attn[-5:].sum()
    mid_10 = mean_attn[-15:-5].sum() if lookback >= 15 else 0
    distant = mean_attn[:-15].sum() if lookback >= 15 else 0

    profile_df = pd.DataFrame({
        "lag": list(range(lookback, 0, -1)),
        "mean_attention": mean_attn,
    })

    log.info(
        "TFT temporal attention: recent_5d=%.3f, mid_10d=%.3f, distant=%.3f",
        recent_5, mid_10, distant,
    )

    return {
        "profile_df": profile_df,
        "recent_5d_attention": recent_5,
        "mid_10d_attention": mid_10,
        "distant_attention": distant,
        "raw_attention": attn,
    }


def save_tft_attention(results: dict, model_name: str) -> None:
    """Save TFT attention analysis."""
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results["profile_df"].to_csv(
        out_dir / f"tft_temporal_attention_{model_name}.csv", index=False
    )
