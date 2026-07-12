"""Alpaca historical market-data access."""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_LOOKBACK_DAYS, SYMBOL, validate_configuration

def create_data_client() -> StockHistoricalDataClient:
    validate_configuration(require_credentials=True)
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def get_market_data(symbol: str = SYMBOL, lookback_days: int = DATA_LOOKBACK_DAYS) -> pd.DataFrame:
    client = create_data_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=lookback_days),
    )
    frame = client.get_stock_bars(request).df.reset_index()
    if frame.empty:
        raise RuntimeError(f"No market data was returned for {symbol}.")
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"] == symbol]
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Alpaca response is missing columns: {missing}")
    return frame[required].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
