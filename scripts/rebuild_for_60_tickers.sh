#!/bin/bash
# ============================================================================
# rebuild_for_60_tickers.sh
#
# After updating settings.yaml to the 60-ticker universe, run this script to:
#   1. Download new GitHub options data
#   2. Remove stale cached files that only cover the old 17 tickers
#   3. Re-download WRDS/Yahoo data for the full universe
#   4. Re-run the full pipeline
#
# Usage:
#   cd options_research
#   bash scripts/rebuild_for_60_tickers.sh
# ============================================================================

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " Step 1: Download GitHub options for new tickers"
echo "============================================"
python scripts/download_new_tickers.py

echo ""
echo "============================================"
echo " Step 2: Remove stale cached files"
echo "============================================"
echo "These files only contain the old 17-ticker data and must be rebuilt."

# Raw data that's ticker-dependent (needs re-download with expanded universe)
rm -f data/raw/wrds/crsp/stock_daily.parquet
echo "  Removed: data/raw/wrds/crsp/stock_daily.parquet"

rm -f data/raw/wrds/compustat/earnings_dates.parquet
echo "  Removed: data/raw/wrds/compustat/earnings_dates.parquet"

rm -f data/raw/yahoo/dividends.parquet
echo "  Removed: data/raw/yahoo/dividends.parquet"

# All interim data (built from old ticker set)
rm -rf data/interim/options_clean/
rm -rf data/interim/stock_clean/
rm -rf data/interim/merged/
echo "  Removed: data/interim/ (all cleaned/merged data)"

# All feature files (built from old ticker set)
rm -rf data/features/
echo "  Removed: data/features/ (all engineered features)"

# Split indices (built from old ticker set)
rm -rf data/splits/
echo "  Removed: data/splits/"

# Model checkpoints (trained on old data)
rm -rf results/model_checkpoints/
echo "  Removed: results/model_checkpoints/"

# Result tables (from old experiments)
rm -rf results/tables/
echo "  Removed: results/tables/"

echo ""
echo "============================================"
echo " Step 3: Re-download WRDS data (new tickers)"
echo "============================================"
echo "This requires WRDS credentials. Run:"
echo "  python -m src.data.wrds_download"
echo ""

echo "============================================"
echo " Step 4: Re-download Yahoo data (new sector ETFs)"
echo "============================================"
echo "Run:"
echo "  python -m src.data.yahoo_download"
echo ""

echo "============================================"
echo " Step 5: Re-run full pipeline"
echo "============================================"
echo "Run:"
echo "  python run_pipeline.py"
echo ""
echo "Or run each step individually:"
echo "  python -m src.data.clean_options"
echo "  python -m src.data.clean_stocks"
echo "  python -m src.data.merge_sources"
echo "  python -m src.features.stock_features"
echo "  python -m src.features.aggregated_features"
echo "  python -m src.features.event_features"
echo "  python -m src.features.surface_model_features"
echo "  python -m src.preprocessing.split"
echo "  python -m src.preprocessing.winsorize"
echo "  python -m src.preprocessing.impute"
echo "  python -m src.preprocessing.normalize"
echo ""
echo "============================================"
echo " Cleanup complete. Ready to rebuild."
echo "============================================"
