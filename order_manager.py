"""Duplicate and stale-order protection for GLD v5."""

from datetime import datetime, timezone

from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


def get_open_buy_orders(client, symbol: str):
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
    )

    return [
        order
        for order in client.get_orders(filter=request)
        if order.symbol == symbol and order.side == OrderSide.BUY
    ]


def cancel_stale_buy_orders(client, symbol: str, maximum_age_minutes: int):
    cancelled = []
    now = datetime.now(timezone.utc)

    for order in get_open_buy_orders(client, symbol):
        if order.submitted_at is None:
            continue

        age_minutes = (now - order.submitted_at).total_seconds() / 60

        if age_minutes >= maximum_age_minutes:
            client.cancel_order_by_id(order.id)
            cancelled.append(str(order.id))

    return cancelled
