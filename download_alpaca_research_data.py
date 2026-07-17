"""Download historical GLD daily bars from Alpaca for research only."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="GLD")
    parser.add_argument("--start", default="2004-11-18")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--output", default="data/GLD_1D.csv")
    args = parser.parse_args()

    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY are required.")

    client = StockHistoricalDataClient(key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=[args.symbol],
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(args.start),
        end=datetime.fromisoformat(args.end),
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise SystemExit("Alpaca returned no bars.")

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
    else:
        bars = bars.reset_index()

    rename = {"timestamp": "date"}
    bars = bars.rename(columns=rename)
    wanted = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in bars.columns]
    bars = bars[wanted]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output, index=False)

    print(f"Saved {len(bars)} historical bars to {output}.")
    print("A market-data request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
