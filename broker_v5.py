"""Limit-order additions for Alpaca."""

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest


def submit_limit_buy(client, symbol: str, notional: float, limit_price: float):
    quantity = round(notional / limit_price, 6)

    request = LimitOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2),
    )

    return client.submit_order(order_data=request)
