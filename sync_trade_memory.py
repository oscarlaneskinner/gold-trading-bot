"""Manual reconciliation of filled GLD paper orders into trade memory."""

from __future__ import annotations

import json

from broker import create_trading_client
from config import PAPER_TRADING, SYMBOL
from trade_memory import database_summary
from trade_memory_sync import synchronize_filled_trades


def run():
    if not PAPER_TRADING:
        raise RuntimeError(
            "This synchronization is restricted to paper trading."
        )

    client = create_trading_client()

    output = {
        "synchronization": synchronize_filled_trades(
            client,
            SYMBOL,
        ),
        "summary": database_summary(),
        "order_submitted": False,
    }

    print("GLD filled-order trade-memory synchronization")
    print(json.dumps(output, indent=2))
    print("No order was submitted.")


if __name__ == "__main__":
    run()
