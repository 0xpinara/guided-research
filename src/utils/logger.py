"""Centralized logging configuration for the research pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "options_research",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Create a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name (usually module __name__).
    level : int
        Logging level.
    log_file : str, optional
        Path to a log file. If provided, logs are also written to disk.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
