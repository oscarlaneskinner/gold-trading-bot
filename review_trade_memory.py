"""Generate a read-only GLD trade-memory performance review."""

from __future__ import annotations

import json

from broker import create_trading_client, get_position
from config import PAPER_TRADING, REPORTS_DIR, SYMBOL
from logger import write_json
from trade_review import build_review


REPORT_PATH = REPORTS_DIR / "trade_memory_review.json"


def current_glg_price(client) -> float | None:
    position = get_position(
        client,
        SYMBOL,
    )

    if position is None:
        return None

    current_price = getattr(
        position,
        "current_price",
        None,
    )

    if current_price is None:
        market_value = float(
            position.market_value
        )
        quantity = float(
            position.qty
        )

        if quantity == 0:
            return None

        return market_value / quantity

    return float(current_price)


def run():
    if not PAPER_TRADING:
        raise RuntimeError(
            "Trade review is restricted to paper trading."
        )

    client = create_trading_client()
    current_price = current_glg_price(client)

    report = build_review(
        current_price=current_price,
    )

    report["symbol"] = SYMBOL
    report["current_price"] = current_price
    report["paper_trading"] = PAPER_TRADING
    report["order_submitted"] = False

    write_json(
        REPORT_PATH,
        report,
    )

    print("GLD trade-memory review")
    print(
        json.dumps(
            report,
            indent=2,
        )
    )
    print(f"JSON report: {REPORT_PATH}")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
