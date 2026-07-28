"""Download daily historical bars for multiple symbols from Alpaca.

This script uses Alpaca's free IEX stock-data feed.

Safety:
- Creates a historical market-data client only.
- Does not create a trading client.
- Cannot submit orders.
- Continues downloading other symbols if one symbol fails.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


DEFAULT_CONFIG_PATH = Path("config/multi_market_arena_v2.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data")


def load_configuration(config_path: Path) -> dict[str, Any]:
    """Load and validate the multi-market research configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    configuration = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    symbols = configuration.get("symbols")

    if not isinstance(symbols, list) or not symbols:
        raise ValueError(
            "The configuration must contain a non-empty symbols list."
        )

    start_date = configuration.get("start_date")

    if not start_date:
        raise ValueError(
            "The configuration must contain start_date."
        )

    return configuration


def required_environment_variable(name: str) -> str:
    """Return a required environment variable or stop safely."""

    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"{name} is required but is not loaded "
            "in this Command Prompt."
        )

    return value


def build_client() -> StockHistoricalDataClient:
    """Create an Alpaca historical-data client."""

    api_key = required_environment_variable(
        "ALPACA_API_KEY"
    )

    secret_key = required_environment_variable(
        "ALPACA_SECRET_KEY"
    )

    return StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )


def download_symbol(
    client: StockHistoricalDataClient,
    symbol: str,
    start_date: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Download and save one symbol's daily IEX bars."""

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start_date),
        end=datetime.now(timezone.utc),
        feed=DataFeed.IEX,
    )

    try:
        bars = client.get_stock_bars(request).df
    except APIError as error:
        return {
            "symbol": symbol,
            "status": "failed",
            "bar_count": 0,
            "file": None,
            "error": str(error),
        }
    except Exception as error:
        return {
            "symbol": symbol,
            "status": "failed",
            "bar_count": 0,
            "file": None,
            "error": (
                f"{type(error).__name__}: {error}"
            ),
        }

    if bars.empty:
        return {
            "symbol": symbol,
            "status": "no_data",
            "bar_count": 0,
            "file": None,
            "error": None,
        }

    bars = bars.reset_index()

    bars = bars.rename(
        columns={
            "timestamp": "date",
        }
    )

    columns_to_save = [
        column
        for column in [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        if column in bars.columns
    ]

    required_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = (
        required_columns - set(columns_to_save)
    )

    if missing_columns:
        return {
            "symbol": symbol,
            "status": "failed",
            "bar_count": 0,
            "file": None,
            "error": (
                "Downloaded data is missing required "
                f"columns: {sorted(missing_columns)}"
            ),
        }

    output_path = (
        output_directory / f"{symbol}_1D.csv"
    )

    bars[columns_to_save].to_csv(
        output_path,
        index=False,
    )

    return {
        "symbol": symbol,
        "status": "saved",
        "bar_count": int(len(bars)),
        "file": str(output_path),
        "error": None,
    }


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Download multi-market daily research "
            "data from Alpaca's IEX feed."
        )
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Path to the multi-market configuration."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help=(
            "Folder where SYMBOL_1D.csv files "
            "will be saved."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Download all configured symbols."""

    arguments = parse_arguments()

    config_path = Path(arguments.config)
    output_directory = Path(
        arguments.output_dir
    )

    configuration = load_configuration(
        config_path
    )

    symbols = [
        str(symbol).upper().strip()
        for symbol in configuration["symbols"]
        if str(symbol).strip()
    ]

    start_date = str(
        configuration["start_date"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = build_client()

    results: list[dict[str, Any]] = []

    print(
        "Downloading multi-market research data "
        "using Alpaca IEX..."
    )

    for symbol in symbols:
        print(f"Downloading {symbol}...")

        result = download_symbol(
            client=client,
            symbol=symbol,
            start_date=start_date,
            output_directory=output_directory,
        )

        results.append(result)

        if result["status"] == "saved":
            print(
                f"Saved {result['bar_count']} bars "
                f"for {symbol} to {result['file']}."
            )
        elif result["status"] == "no_data":
            print(
                f"No historical bars were returned "
                f"for {symbol}."
            )
        else:
            print(
                f"Could not download {symbol}: "
                f"{result['error']}"
            )

    saved_results = [
        result
        for result in results
        if result["status"] == "saved"
    ]

    failed_results = [
        result
        for result in results
        if result["status"] == "failed"
    ]

    empty_results = [
        result
        for result in results
        if result["status"] == "no_data"
    ]

    summary = {
        "status": (
            "completed"
            if saved_results
            else "failed"
        ),
        "feed": "IEX",
        "start_date": start_date,
        "symbols_requested": symbols,
        "symbols_saved": [
            result["symbol"]
            for result in saved_results
        ],
        "symbols_failed": [
            result["symbol"]
            for result in failed_results
        ],
        "symbols_with_no_data": [
            result["symbol"]
            for result in empty_results
        ],
        "total_bars_saved": sum(
            int(result["bar_count"])
            for result in saved_results
        ),
        "results": results,
        "market_data_request_made": True,
        "trading_client_created": False,
        "order_submitted": False,
    }

    print()
    print("Multi-market research-data download")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )
    print(
        "Historical IEX market-data requests "
        "were made."
    )
    print("No trading client was created.")
    print("No order was submitted.")

    if not saved_results:
        raise SystemExit(
            "No symbol data was downloaded."
        )


if __name__ == "__main__":
    main()