"""Reconcile filled Alpaca GLD orders with the SQLite trade-memory database."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from trade_memory import (
    close_latest_open_trade,
    latest_buy_decision_for_order,
    latest_open_trade,
    record_trade_entry,
)


def as_iso(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    return str(value)


def filled_orders(client, symbol: str):
    request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        symbols=[symbol],
        limit=500,
        nested=False,
    )

    orders = client.get_orders(filter=request)

    result = [
        order
        for order in orders
        if order.symbol == symbol
        and order.filled_at is not None
        and order.filled_avg_price is not None
        and order.filled_qty is not None
    ]

    return sorted(
        result,
        key=lambda order: order.filled_at,
    )


def decision_context(order_id: str):
    row = latest_buy_decision_for_order(order_id)

    if row is None:
        return None, None, None

    features = None

    if row["features_json"]:
        features = json.loads(row["features_json"])

    return (
        row["probability_up"],
        row["position_percent"],
        features,
    )


def synchronize_filled_trades(
    client,
    symbol: str,
) -> dict[str, int]:
    """
    Synchronize the most recent completed buy/sell cycle.

    The bot permits only one GLD position at a time, so the reconciliation
    deliberately uses a single-open-trade model.
    """

    created_entries = 0
    closed_entries = 0
    skipped_orders = 0

    for order in filled_orders(client, symbol):
        order_id = str(order.id)
        side = order.side
        fill_price = float(order.filled_avg_price)
        quantity = float(order.filled_qty)
        notional = fill_price * quantity

        if side == OrderSide.BUY:
            probability, position_percent, features = (
                decision_context(order_id)
            )

            existing_open = latest_open_trade(symbol)

            if (
                existing_open is not None
                and existing_open["entry_order_id"] != order_id
            ):
                skipped_orders += 1
                continue

            record_trade_entry(
                symbol=symbol,
                entry_order_id=order_id,
                entry_timestamp_utc=as_iso(order.filled_at),
                entry_price=fill_price,
                notional=notional,
                quantity=quantity,
                probability_up=(
                    float(probability)
                    if probability is not None
                    else None
                ),
                position_percent=(
                    float(position_percent)
                    if position_percent is not None
                    else None
                ),
                entry_features=features,
            )
            created_entries += 1

        elif side == OrderSide.SELL:
            closed = close_latest_open_trade(
                symbol=symbol,
                exit_order_id=order_id,
                exit_timestamp_utc=as_iso(order.filled_at),
                exit_price=fill_price,
                exit_reason="filled_sell_order",
            )

            if closed:
                closed_entries += 1
            else:
                skipped_orders += 1

    return {
        "created_or_updated_entries": created_entries,
        "closed_entries": closed_entries,
        "skipped_orders": skipped_orders,
    }
