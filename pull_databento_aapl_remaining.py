#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull remaining AAPL Databento data (OPRA statistics) in monthly chunks."
    )
    parser.add_argument("--start", default="2023-03-28", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/raw/databento", help="Output directory")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per chunk")
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("Set DATABENTO_API_KEY in your environment before running.")
    return api_key


def cap_end(client: db.Historical, dataset: str, schema: str, requested_end: str) -> str:
    rng = client.metadata.get_dataset_range(dataset=dataset)
    schema_end = rng["schema"][schema]["end"][:10]
    return min(requested_end, schema_end)


def fetch_monthly(
    client: db.Historical,
    dataset: str,
    schema: str,
    symbol: str,
    start: str,
    end: str,
    retries: int,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    month_starts = pd.date_range(start=start_ts, end=end_ts, freq="MS")
    if len(month_starts) == 0 or month_starts[0] != start_ts.normalize().replace(day=1):
        month_starts = month_starts.insert(0, start_ts.normalize().replace(day=1))

    chunks: list[pd.DataFrame] = []
    for month_start in month_starts:
        chunk_start = max(start_ts, month_start)
        chunk_end = min(end_ts, month_start + pd.offsets.MonthBegin(1))
        if chunk_start >= chunk_end:
            continue

        got = False
        err_msg = ""
        for attempt in range(1, retries + 1):
            try:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema=schema,
                    stype_in="parent",
                    symbols=[symbol],
                    start=chunk_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                )
                df = data.to_df()
                chunks.append(df)
                if checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    ck_name = (
                        f"aapl_{schema}_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.parquet"
                    )
                    df.to_parquet(checkpoint_dir / ck_name)
                print(
                    f"{schema} {chunk_start.date()}->{chunk_end.date()} rows={len(df)} attempt={attempt}"
                )
                got = True
                break
            except Exception as exc:  # noqa: BLE001
                err_msg = str(exc)
                wait_s = min(2**attempt, 30)
                print(
                    f"{schema} {chunk_start.date()}->{chunk_end.date()} failed attempt={attempt}: {err_msg}"
                )
                time.sleep(wait_s)

        if not got:
            raise RuntimeError(
                f"Failed chunk {chunk_start.date()}->{chunk_end.date()} for {schema}: {err_msg}"
            )

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks).sort_index()
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = db.Historical(require_api_key())
    end_stats = cap_end(client, dataset="OPRA.PILLAR", schema="statistics", requested_end=args.end)
    checkpoint_dir = out_dir / "checkpoints" / "statistics"

    stats = fetch_monthly(
        client=client,
        dataset="OPRA.PILLAR",
        schema="statistics",
        symbol="AAPL.OPT",
        start=args.start,
        end=end_stats,
        retries=args.max_retries,
        checkpoint_dir=checkpoint_dir,
    )
    stats_path = out_dir / "aapl_options_statistics.parquet"
    stats.to_parquet(stats_path)

    print(f"Saved: {stats_path} rows={len(stats)}")
    print(f"Effective end date for statistics: {end_stats}")


if __name__ == "__main__":
    main()
