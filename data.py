"""
Market data handler for Gold AI Trading Bot

Downloads historical price data from Alpaca.
"""

from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    SYMBOL
)


# Create Alpaca data connection

data_client = StockHistoricalDataClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY
)


def get_market_data(
    symbol=SYMBOL,
    lookback_days=500
):
    """
    Download daily bars from Alpaca.
    """

    start_date = (
        datetime.now()
        -
        timedelta(days=lookback_days)
    )


    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date
    )


    bars = data_client.get_stock_bars(
        request
    )


    df = bars.df.reset_index()


    # Keep only requested symbol

    df = df[
        df["symbol"] == symbol
    ]


    # Sort oldest to newest

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


    # Rename columns if needed

    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }
    )


    return df



def get_latest_price(symbol=SYMBOL):

    df = get_market_data(
        symbol,
        lookback_days=10
    )

    return float(
        df.iloc[-1]["close"]
    )
