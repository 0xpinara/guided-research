#!/bin/bash
# Download GitHub options data for all 60 tickers.
# Saves to data/raw/github_options/{ticker}_options.parquet etc.
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="data/raw/github_options"
BASE_URL="https://static.philippdubach.com/data/options"
mkdir -p "$DEST"

TICKERS=(
  aapl msft nvda amd meta nflx goog crm adbe csco avgo orcl
  jpm gs bac c wfc v ma ms blk axp
  jnj unh pfe lly abbv mrk amgn tmo
  hd mcd amzn nke low tsla
  wmt ko pep pg cost
  xom cvx cop
  ba cat hon ge de lmt
  dis cmcsa t
  nee duk so
  amt
  spy qqq iwm
)

downloaded=0
skipped=0
failed=0

for ticker in "${TICKERS[@]}"; do
  for kind in options underlying; do
    dest_file="$DEST/${ticker}_${kind}.parquet"
    if [[ -f "$dest_file" ]]; then
      skipped=$((skipped + 1))
      continue
    fi

    url="$BASE_URL/$ticker/$kind.parquet"
    echo "[$ticker] Downloading $kind.parquet ..."
    if curl -fSL --progress-bar "$url" -o "$dest_file" 2>&1; then
      size=$(du -h "$dest_file" | cut -f1)
      echo "[$ticker] $kind.parquet done ($size)"
      downloaded=$((downloaded + 1))
    else
      echo "[$ticker] $kind.parquet FAILED"
      rm -f "$dest_file"
      failed=$((failed + 1))
    fi
  done
done

echo ""
echo "================================"
echo "Skipped (already exist): $skipped"
echo "Downloaded:              $downloaded"
echo "Failed:                  $failed"
echo "Total size: $(du -sh "$DEST" | cut -f1)"
echo "================================"
