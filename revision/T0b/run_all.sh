#!/usr/bin/env bash
# Task 0b: corrected data layer, walk-forward benchmark and economics, in order.
# Reads data/ and results/ (BIR era) read-only; writes only to revision/out/T0b/.
# Never deletes BIR-era files. Run from the project root:  bash revision/T0b/run_all.sh
set -euo pipefail
mkdir -p revision/out/T0b/logs

# 0. one-time Yahoo snapshot of earnings dates (network; already done on 2026-10-01)
[ -f revision/out/T0b/raw/yahoo_earnings_dates.csv ] || python3 revision/T0b/fetch_yahoo_earnings.py

# 1. corrected panels (about 1.5 min)
python3 revision/T0b/build_panel.py

# 2. corrected walk-forward grid, primary panel: 324 cells (about 40 min on 7 cores)
python3 revision/T0b/run_grid.py --panel primary --workers 7

# 3. robustness: same pipeline without the 50-contract rule (return grid; volatility for A and D)
python3 revision/T0b/run_grid.py --panel unfiltered --targets return --workers 7
python3 revision/T0b/run_grid.py --panel unfiltered --targets volatility --sets A D --workers 7

# 4. economics, tables and comparisons
python3 revision/T0b/economics.py
python3 revision/T0b/summarize.py
python3 revision/T0b/robustness.py

# 5. checksums of the large regenerable outputs (kept out of git)
(cd revision/out/T0b && find panel predictions predictions_unfiltered -type f -name '*.parquet' -print0 \
   | sort -z | xargs -0 shasum -a 256 > MANIFEST_large_files.sha256)
