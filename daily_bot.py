"""
AI-driven daily trading bot for GLD.
Meant to run once per day, shortly after market open (or before close).

Flow:
1. Pull recent daily bars for GLD from Alpaca's free market data API
2. Calculate the same features used in training
3. Load the trained model, get today's prediction
4. If predicting "up" and no position open -> buy
5. If holding a position for 5+ trading days -> close it
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ============ CONFIG ============
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
PAPER_TRADING = True

SYMBOL = "GLD"
DOLLAR_AMOUNT = 1000       # position size per trade - adjust to your comfort level
HOLD_DAYS = 5              # matches the model's training window
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURE_COLS = ['return_1d','return_5d','return_10d','return_20d',
                 'price_vs_ema200','ema9_vs_ema21','ema21_vs_ema50',
                 'rsi_14','rsi_7','atr_pct','volatility_20d',
                 'volume_change','volume_ma_ratio']

# ============ CLIENTS ============
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)


# ============ FEATURE ENGINEERING (same as training) ============
def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length).mean()

def build_features(df):
    df = df.copy()
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['price_vs_ema200'] = (df['close'] - df['ema_200']) / df['ema_200']
    df['ema9_vs_ema21'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
    df['ema21_vs_ema50'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
    df['rsi_14'] = rsi(df['close'], 14)
    df['rsi_7'] = rsi(df['close'], 7)
    df['atr_14'] = atr(df, 14)
    df['atr_pct'] = df['atr_14'] / df['close']
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    return df


def get_recent_bars(symbol, lookback_days=400):
    """Pull enough daily history to compute all features (200 EMA needs ~200+ days)."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=lookback_days),
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
    return df


def get_open_position(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return position
    except Exception:
        return None


def get_position_days_held(symbol):
    """Check how many trading days the current position has been open, based on order history."""
    orders = trading_client.get_orders()
    buy_orders = [o for o in orders if o.symbol == symbol and o.side == OrderSide.BUY and o.status == 'filled']
    if not buy_orders:
        return None
    most_recent_buy = max(buy_orders, key=lambda o: o.filled_at)
    days_held = (datetime.now(most_recent_buy.filled_at.tzinfo) - most_recent_buy.filled_at).days
    return days_held


def run_daily_decision():
    print(f"=== Daily AI bot run: {datetime.now()} ===")

    # 1. Pull recent data
    raw_df = get_recent_bars(SYMBOL)
    print(f"Pulled {len(raw_df)} days of history, latest: {raw_df.iloc[-1]['timestamp']}")

    # 2. Build features
    feat_df = build_features(raw_df)
    latest_row = feat_df.iloc[[-1]][FEATURE_COLS]

    if latest_row.isna().any(axis=1).iloc[0]:
        print("ERROR: Not enough history to compute features. Need ~250+ days of data.")
        return

    # 3. Load model, predict
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    prediction = int(model.predict(latest_row)[0])
    probability_up = float(model.predict_proba(latest_row)[0][1])
    print(f"Model prediction: {'UP' if prediction == 1 else 'DOWN/FLAT'} (confidence: {probability_up:.1%})")

    # 4. Check current position
    position = get_open_position(SYMBOL)

    if position is None:
        # No position currently open
        if prediction == 1:
            print(f"No position open, model says UP -> placing buy order for ${DOLLAR_AMOUNT}")
            order_request = MarketOrderRequest(
                symbol=SYMBOL,
                notional=DOLLAR_AMOUNT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = trading_client.submit_order(order_request)
            print(f"Buy order submitted: {order.id}")
        else:
            print("No position open, model says DOWN/FLAT -> staying in cash, no action.")
    else:
        # Already holding a position - check if it's time to exit
        days_held = get_position_days_held(SYMBOL)
        print(f"Currently holding position, open for {days_held} days.")
        if days_held is not None and days_held >= HOLD_DAYS:
            print(f"Held for {HOLD_DAYS}+ days -> closing position")
            trading_client.close_position(SYMBOL)
            print("Position closed.")
        else:
            print(f"Holding period not yet reached ({days_held}/{HOLD_DAYS} days) -> no action.")

    print("=== Run complete ===\n")


if __name__ == "__main__":
    run_daily_decision()
