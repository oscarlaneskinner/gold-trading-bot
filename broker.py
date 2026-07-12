"""Thin Alpaca trading wrapper."""

from __future__ import annotations
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING, SYMBOL, validate_configuration

def create_trading_client() -> TradingClient:
    validate_configuration(require_credentials=True)
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)

def get_account_equity(client: TradingClient) -> float:
    return float(client.get_account().equity)

def get_position(client: TradingClient, symbol: str = SYMBOL):
    try:
        return client.get_open_position(symbol)
    except Exception:
        return None

def submit_market_buy(client: TradingClient, symbol: str, notional: float):
    request = MarketOrderRequest(
        symbol=symbol, notional=round(notional, 2),
        side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order_data=request)

def close_position(client: TradingClient, symbol: str = SYMBOL):
    return client.close_position(symbol)

def get_filled_buy_orders(client: TradingClient, symbol: str = SYMBOL):
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol])
    orders = client.get_orders(filter=request)
    return [
        order for order in orders
        if order.symbol == symbol and order.side == OrderSide.BUY
        and order.filled_at is not None
    ]
