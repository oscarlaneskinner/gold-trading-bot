"""Inspect persistent GLD decisions and completed/open trade memory."""

from __future__ import annotations

import json

from trade_memory import (
    database_summary,
    recent_trades,
)


def run():
    output = {
        "summary": database_summary(),
        "recent_trades": recent_trades(limit=10),
        "order_submitted": False,
    }

    print("GLD trade-memory Phase 3 inspection")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
