"""Market-level features: VIX, rates, sector, Fama-French.

These are already merged into stock_daily during clean_stocks, so this
module exists primarily for documentation and any additional market-level
computations not covered by stock_features.py.
"""

from src.utils.logger import setup_logger

log = setup_logger(__name__)

# Market features (feat_38-42) are computed in stock_features.py since
# they are already available in the cleaned stock DataFrame.
# This module is a placeholder for any future market-level features
# that need separate computation (e.g., cross-sectional dispersion).


def run(cfg=None):
    log.info("Market features are computed within stock_features.py")
