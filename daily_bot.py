"""
AI-driven daily trading bot for GLD.
Meant to run once per day, shortly after market open (or before close).

Flow:
1. Pull recent daily bars for GLD from Alpaca's free market data API
2. Calculate the same features used in training
3. Load the trained model, get today's prediction
4. If predicting "up" and no position open -> buy
5. If holding a position, check exit conditions in this order:
   a. Stop-loss hit (price <= entry * (1 - STOP_LOSS_PCT))
   b. Take-profit hit (price >= entry * (1 + TAKE_PROFIT_PCT))
   c. Trailing stop hit (once profitable, price pulled back TRAILING_STOP_PCT from its peak)
   d. Time limit reached (held for HOLD_DAYS trading days) - backstop exit
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
HOLD_DAYS = 20             # backstop exit if no stop/target hit first
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Stop-loss / take-profit settings (tested via backtest - 10% SL / 20% TP showed consistent
# improvement over baseline across all 5 walk-forward periods, roughly matching buy-and-hold)
STOP_LOSS_PCT = 0.10       # exit if price drops 10% from entry
TAKE_PROFIT_PCT = 0.20     # exit if price rises 20% from entry
TRAILING_STOP_PCT = None   # disabled - tight trailing stops tested worse; wide fixed stop/target worked better
TRAILING_ACTIVATION_PCT = 0.01  # unused while TRAILING_STOP_PCT is None

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


def get_position_entry_info(symbol):
    """Get the entry date and entry price of the current position, based on the most recent filled buy order."""
    orders = trading_client.get_orders()
    buy_orders = [o for o in orders if o.symbol == symbol and o.side == OrderSide.BUY and o.status == 'filled']
    if not buy_orders:
        return None, None
    most_recent_buy = max(buy_orders, key=lambda o: o.filled_at)
    entry_price = float(most_recent_buy.filled_avg_price)
    entry_date = most_recent_buy.filled_at
    return entry_date, entry_price


def check_exit_conditions(raw_df, entry_date, entry_price):
    """
    Check price action since entry against stop-loss, take-profit, and trailing stop.
    Uses daily High/Low from Alpaca bars (checked once per day, not intraday).
    Returns: (should_exit: bool, reason: str, days_held: int)
    """
    since_entry = raw_df[raw_df['timestamp'] >= entry_date].copy()
    if len(since_entry) == 0:
        return False, None, 0

    days_held = len(since_entry) - 1  # trading days since entry (today counts as day 0)
    peak_price = max(entry_price, since_entry['high'].max())

    latest_low = since_entry.iloc[-1]['low']
    latest_high = since_entry.iloc[-1]['high']

    # 1. Stop loss check
    if latest_low <= entry_price * (1 - STOP_LOSS_PCT):
        return True, 'stop_loss', days_held

    # 2. Take profit check
    if latest_high >= entry_price * (1 + TAKE_PROFIT_PCT):
        return True, 'take_profit', days_held

    # 3. Trailing stop check (only active if enabled AND once meaningfully profitable)
    if TRAILING_STOP_PCT is not None and peak_price >= entry_price * (1 + TRAILING_ACTIVATION_PCT):
        trail_trigger = peak_price * (1 - TRAILING_STOP_PCT)
        if latest_low <= trail_trigger:
            return True, 'trailing_stop', days_held

    # 4. Time limit backstop
    if days_held >= HOLD_DAYS:
        return True, 'time_limit', days_held

    return False, None, days_held


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
        # Already holding a position - check stop-loss / take-profit / trailing stop / time limit
        entry_date, entry_price = get_position_entry_info(SYMBOL)
        if entry_date is None:
            print("WARNING: Could not determine entry info for open position. Skipping exit check.")
            return

        should_exit, reason, days_held = check_exit_conditions(raw_df, entry_date, entry_price)
        print(f"Currently holding position. Entry price: ${entry_price:.2f}, days held: {days_held}")

        if should_exit:
            print(f"Exit condition triggered: {reason} -> closing position")
            trading_client.close_position(SYMBOL)
            print("Position closed.")
        else:
            print(f"No exit condition met yet (days_held={days_held}/{HOLD_DAYS}) -> holding.")

    print("=== Run complete ===\n")


if __name__ == "__main__":
    run_daily_decision()
