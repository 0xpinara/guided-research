"""Seed setting and deterministic flags for reproducibility."""

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds and deterministic flags.

    Parameters
    ----------
    seed : int
        Global random seed.
    deterministic : bool
        If True, enable CUDA deterministic mode (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass  # torch not installed; skip
