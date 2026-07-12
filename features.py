"""Feature engineering shared by training, backtesting, and daily inference."""

from __future__ import annotations
import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "price_vs_ema200", "ema9_vs_ema21", "ema21_vs_ema50",
    "rsi_14", "rsi_7", "atr_pct", "volatility_20d",
    "volume_change", "volume_ma_ratio", "macd_pct",
    "macd_signal_pct", "bollinger_position",
]

def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {sorted(missing)}")

    result = df.copy().sort_values("timestamp").reset_index(drop=True)
    for period in (1, 5, 10, 20):
        result[f"return_{period}d"] = result["close"].pct_change(period)
    for span in (9, 21, 50, 200):
        result[f"ema_{span}"] = result["close"].ewm(span=span, adjust=False).mean()

    result["price_vs_ema200"] = (result["close"] - result["ema_200"]) / result["ema_200"]
    result["ema9_vs_ema21"] = (result["ema_9"] - result["ema_21"]) / result["ema_21"]
    result["ema21_vs_ema50"] = (result["ema_21"] - result["ema_50"]) / result["ema_50"]
    result["rsi_14"] = calculate_rsi(result["close"], 14)
    result["rsi_7"] = calculate_rsi(result["close"], 7)
    result["atr_14"] = calculate_atr(result, 14)
    result["atr_pct"] = result["atr_14"] / result["close"]
    result["volatility_20d"] = result["return_1d"].rolling(20).std()
    result["volume_change"] = result["volume"].pct_change()
    result["volume_ma_ratio"] = result["volume"] / result["volume"].rolling(20).mean()

    ema_12 = result["close"].ewm(span=12, adjust=False).mean()
    ema_26 = result["close"].ewm(span=26, adjust=False).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_pct"] = result["macd"] / result["close"]
    result["macd_signal_pct"] = result["macd_signal"] / result["close"]

    middle = result["close"].rolling(20).mean()
    std = result["close"].rolling(20).std()
    upper, lower = middle + 2 * std, middle - 2 * std
    result["bollinger_position"] = (result["close"] - lower) / (upper - lower).replace(0, np.nan)

    result[MODEL_FEATURES] = result[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return result
