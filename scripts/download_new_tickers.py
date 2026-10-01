"""Download GitHub options data for all tickers in the expanded universe.

Fetches options.parquet and underlying.parquet for each ticker that
doesn't already exist in data/raw/github_options/.

Usage:
    python scripts/download_new_tickers.py
"""

import os
import sys
import urllib.request
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
GITHUB_OPTIONS_DIR = ROOT / "data" / "raw" / "github_options"

BASE_URL = "https://static.philippdubach.com/data/options"

# Full 60-ticker universe from settings.yaml (lowercase for URL)
NEW_TICKERS = [
    # Tech
    "aapl", "msft", "nvda", "amd", "meta", "nflx", "goog", "crm", "adbe", "csco", "avgo", "orcl",
    # Financials
    "jpm", "gs", "bac", "c", "wfc", "v", "ma", "ms", "blk", "axp",
    # Healthcare
    "jnj", "unh", "pfe", "lly", "abbv", "mrk", "amgn", "tmo",
    # Consumer Discretionary
    "hd", "mcd", "amzn", "nke", "low", "tsla",
    # Consumer Staples
    "wmt", "ko", "pep", "pg", "cost",
    # Energy
    "xom", "cvx", "cop",
    # Industrials
    "ba", "cat", "hon", "ge", "de", "lmt",
    # Communication
    "dis", "cmcsa", "t",
    # Utilities
    "nee", "duk", "so",
    # Real Estate
    "amt",
    # ETFs
    "spy", "qqq", "iwm",
]


def download_file(url: str, dest: str) -> bool:
    try:
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def main():
    GITHUB_OPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    already = 0
    downloaded = 0
    failed = []

    for ticker in NEW_TICKERS:
        for kind in ["options", "underlying"]:
            dest = GITHUB_OPTIONS_DIR / f"{ticker}_{kind}.parquet"
            if dest.exists():
                already += 1
                continue

            url = f"{BASE_URL}/{ticker}/{kind}.parquet"
            print(f"[{ticker.upper()}] Downloading {kind}.parquet ...")

            if download_file(url, str(dest)):
                downloaded += 1
                size_mb = dest.stat().st_size / 1024 / 1024
                print(f"[{ticker.upper()}] {kind}.parquet done ({size_mb:.1f} MB)")
            else:
                failed.append(f"{ticker}_{kind}")
                if dest.exists():
                    dest.unlink()

    print(f"\n{'='*50}")
    print(f"Already existed: {already}")
    print(f"Downloaded:      {downloaded}")
    print(f"Failed:          {len(failed)}")
    if failed:
        print(f"  Failed files: {failed}")
    print("Done!")


if __name__ == "__main__":
    main()
