"""
Feature engineering for Gold AI Trading Bot

Converts raw market data into AI model inputs.
"""

import pandas as pd
import numpy as np


def calculate_rsi(series, length=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        min_periods=length
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        min_periods=length
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_atr(df, length=14):

    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] -
        df["close"].shift()
    ).abs()

    low_close = (
        df["low"] -
        df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / length,
        min_periods=length
    ).mean()


def add_features(df):

    df = df.copy()

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)


    # Moving averages
    df["ema_9"] = (
        df["close"]
        .ewm(span=9)
        .mean()
    )

    df["ema_21"] = (
        df["close"]
        .ewm(span=21)
        .mean()
    )

    df["ema_50"] = (
        df["close"]
        .ewm(span=50)
        .mean()
    )

    df["ema_200"] = (
        df["close"]
        .ewm(span=200)
        .mean()
    )


    # EMA relationships
    df["price_vs_ema200"] = (
        df["close"] -
        df["ema_200"]
    ) / df["ema_200"]


    df["ema9_vs_ema21"] = (
        df["ema_9"] -
        df["ema_21"]
    ) / df["ema_21"]


    df["ema21_vs_ema50"] = (
        df["ema_21"] -
        df["ema_50"]
    ) / df["ema_50"]


    # RSI
    df["rsi_14"] = calculate_rsi(
        df["close"],
        14
    )

    df["rsi_7"] = calculate_rsi(
        df["close"],
        7
    )


    # ATR
    df["atr_14"] = calculate_atr(df)

    df["atr_pct"] = (
        df["atr_14"] /
        df["close"]
    )


    # Volatility
    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )


    # Volume
    df["volume_change"] = (
        df["volume"]
        .pct_change()
    )

    df["volume_ma_ratio"] = (
        df["volume"] /
        df["volume"]
        .rolling(20)
        .mean()
    )


    # MACD
    ema12 = (
        df["close"]
        .ewm(span=12)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26)
        .mean()
    )

    df["macd"] = ema12 - ema26


    # Bollinger Bands
    middle = (
        df["close"]
        .rolling(20)
        .mean()
    )

    std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["bollinger_upper"] = middle + (2 * std)

    df["bollinger_lower"] = middle - (2 * std)


    df["bollinger_position"] = (
        df["close"] - df["bollinger_lower"]
    ) / (
        df["bollinger_upper"] -
        df["bollinger_lower"]
    )


    return df


# Features sent into the AI model

MODEL_FEATURES = [

    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",

    "price_vs_ema200",
    "ema9_vs_ema21",
    "ema21_vs_ema50",

    "rsi_14",
    "rsi_7",

    "atr_pct",
    "volatility_20d",

    "volume_change",
    "volume_ma_ratio",

    "macd",

    "bollinger_position"
]
