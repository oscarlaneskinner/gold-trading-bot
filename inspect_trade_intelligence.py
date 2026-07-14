"""Inspect the GLD trade-intelligence research database."""

from __future__ import annotations

import json

from trade_intelligence import intelligence_summary
from trade_memory import database_summary


def run():
    output = {
        "trade_memory": database_summary(),
        "trade_intelligence": intelligence_summary(),
        "order_submitted": False,
    }

    print("GLD trade-intelligence summary")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
