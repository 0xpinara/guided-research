#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull AAPL stock + options data from Databento."
    )
    parser.add_argument(
        "--start",
        default="2023-03-28",
        help="Start date (YYYY-MM-DD). Defaults to OPRA availability start.",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/databento",
        help="Output directory for parquet files.",
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Export it and rerun the script."
        )
    return api_key


def fetch_monthly_chunks(
    client: db.Historical,
    dataset: str,
    schema: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    # OPRA range requests can timeout when too wide; chunk by calendar month.
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

        data = client.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            stype_in="parent",
            symbols=[symbol],
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
        )
        chunk_df = data.to_df()
        if not chunk_df.empty:
            chunks.append(chunk_df)
        print(
            f"{dataset}/{schema} {chunk_start.date()} to {chunk_end.date()}: "
            f"{len(chunk_df)} rows"
        )

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).sort_index()


def cap_end_for_schema(
    client: db.Historical,
    dataset: str,
    schema: str,
    requested_end: str,
) -> str:
    dataset_range = client.metadata.get_dataset_range(dataset=dataset)
    schema_end = dataset_range["schema"][schema]["end"][:10]
    return min(requested_end, schema_end)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = db.Historical(ensure_api_key())

    stock_end = cap_end_for_schema(
        client=client,
        dataset="XNAS.ITCH",
        schema="ohlcv-1d",
        requested_end=args.end,
    )
    options_ohlcv_end = cap_end_for_schema(
        client=client,
        dataset="OPRA.PILLAR",
        schema="ohlcv-1d",
        requested_end=args.end,
    )
    options_def_end = cap_end_for_schema(
        client=client,
        dataset="OPRA.PILLAR",
        schema="definition",
        requested_end=args.end,
    )

    stock = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        schema="ohlcv-1d",
        stype_in="raw_symbol",
        symbols=["AAPL"],
        start=args.start,
        end=stock_end,
    ).to_df()
    stock_path = out_dir / "aapl_stock_ohlcv_1d.parquet"
    stock.to_parquet(stock_path)

    options_ohlcv = fetch_monthly_chunks(
        client=client,
        dataset="OPRA.PILLAR",
        schema="ohlcv-1d",
        symbol="AAPL.OPT",
        start=args.start,
        end=options_ohlcv_end,
    )
    options_ohlcv_path = out_dir / "aapl_options_ohlcv_1d.parquet"
    options_ohlcv.to_parquet(options_ohlcv_path)

    options_def = fetch_monthly_chunks(
        client=client,
        dataset="OPRA.PILLAR",
        schema="definition",
        symbol="AAPL.OPT",
        start=args.start,
        end=options_def_end,
    )
    options_def_path = out_dir / "aapl_options_definition.parquet"
    options_def.to_parquet(options_def_path)

    print(f"Saved: {stock_path} rows={len(stock)}")
    print(f"Saved: {options_ohlcv_path} rows={len(options_ohlcv)}")
    print(f"Saved: {options_def_path} rows={len(options_def)}")
    print(
        f"Effective end dates -> stock: {stock_end}, options_ohlcv: {options_ohlcv_end}, "
        f"options_definition: {options_def_end}"
    )
    print(
        "Note: Databento OPRA does not provide open interest, implied volatility, "
        "or Greeks as native fields."
    )


if __name__ == "__main__":
    main()
