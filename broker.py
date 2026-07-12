"""
Broker connection module for Gold AI Trading Bot

Handles Alpaca trading operations.
"""

from alpaca.trading.client import TradingClient

from alpaca.trading.requests import (
    MarketOrderRequest
)

from alpaca.trading.enums import (
    OrderSide,
    TimeInForce
)


from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    PAPER_TRADING,
    SYMBOL
)



# Alpaca connection

trading_client = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=PAPER_TRADING
)



def get_account():

    return trading_client.get_account()



def get_equity():

    account = get_account()

    return float(
        account.equity
    )



def get_position(symbol=SYMBOL):

    try:

        return trading_client.get_open_position(
            symbol
        )

    except Exception:

        return None



def buy_stock(
    symbol,
    dollar_amount
):

    order = MarketOrderRequest(

        symbol=symbol,

        notional=dollar_amount,

        side=OrderSide.BUY,

        time_in_force=TimeInForce.DAY

    )


    response = trading_client.submit_order(
        order
    )


    return response



def sell_stock(
    symbol=SYMBOL
):

    response = trading_client.close_position(
        symbol
    )


    return response



def get_latest_orders():

    return trading_client.get_orders()
