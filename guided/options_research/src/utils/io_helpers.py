"""Parquet read/write helpers and checkpoint management."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

log = setup_logger(__name__)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to parquet, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    log.info("Saved %s rows to %s", len(df), path)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet file."""
    path = Path(path)
    df = pd.read_parquet(path)
    log.info("Loaded %s rows from %s", len(df), path)
    return df


def save_npz(path: str | Path, **arrays) -> None:
    """Save multiple numpy arrays to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    log.info("Saved npz to %s (keys: %s)", path, list(arrays.keys()))


def load_npz(path: str | Path) -> dict:
    """Load an .npz file and return a dict of arrays."""
    data = np.load(path, allow_pickle=True)
    return dict(data)


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Save a PyTorch-style checkpoint dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.save(state, path)
        log.info("Saved checkpoint to %s", path)
    except ImportError:
        # Fallback: pickle
        import pickle
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info("Saved checkpoint (pickle) to %s", path)


def load_checkpoint(path: str | Path) -> dict:
    """Load a PyTorch-style checkpoint dict."""
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Project path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"
