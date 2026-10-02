"""Task 0b.1 — one-time fetch of earnings announcement dates from Yahoo Finance.

Compustat (the BIR-era WRDS pull) has no rows for GOOG or GOOGL, and the WRDS
re-pull could not connect (Task 0b README), so GOOG's earnings dates come from
Yahoo Finance (GOOGL and GOOG share Alphabet's announcement dates). Ten tickers
that also have Compustat report dates (rdq) are fetched so the two sources can
be compared before the Yahoo dates are used.

Writes revision/out/T0b/raw/yahoo_earnings_dates.csv (the raw snapshot, with the
fetch time). build_panel.py reads only this snapshot, never the network.
Run once:  python revision/T0b/fetch_yahoo_earnings.py
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parents[2] / "revision" / "out" / "T0b" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "JPM", "JNJ", "XOM", "WMT", "KO", "DIS"]

rows = []
fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
for sym in SYMBOLS:
    e = yf.Ticker(sym).get_earnings_dates(limit=60)
    if e is None or e.empty:
        print(f"{sym}: no data")
        continue
    for ts in e.index:
        rows.append({"symbol": sym, "announce_ts": ts.isoformat(),
                     "announce_date": ts.tz_localize(None).normalize().date().isoformat(),
                     "fetched_at_utc": fetched_at, "source": f"yfinance {yf.__version__} get_earnings_dates"})
    print(f"{sym}: {len(e)} dates")
df = pd.DataFrame(rows).sort_values(["symbol", "announce_date"])
df.to_csv(OUT / "yahoo_earnings_dates.csv", index=False)
print(f"wrote {len(df)} rows to {OUT / 'yahoo_earnings_dates.csv'}")
