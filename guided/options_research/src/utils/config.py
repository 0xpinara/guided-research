"""Load and expose the project configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(i) for i in d]
    return d


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load settings.yaml and return as a nested namespace.

    Parameters
    ----------
    path : str or Path, optional
        Path to settings.yaml.  Defaults to ``config/settings.yaml``
        relative to the project root.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def load_feature_defs(path: str | Path | None = None) -> dict:
    """Load feature_definitions.yaml as a plain dict."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "feature_definitions.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def all_tickers(cfg: SimpleNamespace) -> list[str]:
    """Return a flat list of all tickers from config."""
    tickers = []
    for group in vars(cfg.tickers).values():
        tickers.extend(group)
    return tickers


def ticker_to_sector(cfg: SimpleNamespace) -> dict[str, str]:
    """Return a dict mapping ticker -> sector name."""
    return {k: v for k, v in vars(cfg.ticker_sector_map).items()}


def sector_to_etf(cfg: SimpleNamespace) -> dict[str, str]:
    """Return a dict mapping sector name -> sector ETF ticker."""
    return {k: v for k, v in vars(cfg.sector_etfs).items()}
