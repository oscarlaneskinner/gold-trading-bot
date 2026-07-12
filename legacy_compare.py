"""
Legacy bot comparison runner.

This script compares the original bot's signal logic against the
current v3 bot without submitting any orders.

It is intentionally read-only:
- no TradingClient
- no order submission
- no position closing
"""

from __future__ import annotations

import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import MODEL_PATH, SYMBOL
from data import get_market_data


LEGACY_LOG_PATH = Path("reports/legacy_signals.csv")

LEGACY_FEATURES = [
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
]


def calculate_rsi(
    series: pd.Series,
    length: int = 14,
) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / length,
        min_periods=length,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / length,
        min_periods=length,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss

    return 100 - (
        100 / (1 + relative_strength)
    )


def calculate_atr(
    frame: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    high_low = frame["high"] - frame["low"]

    high_close = (
        frame["high"]
        - frame["close"].shift()
    ).abs()

    low_close = (
        frame["low"]
        - frame["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / length,
        min_periods=length,
        adjust=False,
    ).mean()


def build_legacy_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["return_1d"] = (
        result["close"].pct_change(1)
    )
    result["return_5d"] = (
        result["close"].pct_change(5)
    )
    result["return_10d"] = (
        result["close"].pct_change(10)
    )
    result["return_20d"] = (
        result["close"].pct_change(20)
    )

    result["ema_9"] = result["close"].ewm(
        span=9,
        adjust=False,
    ).mean()

    result["ema_21"] = result["close"].ewm(
        span=21,
        adjust=False,
    ).mean()

    result["ema_50"] = result["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    result["ema_200"] = result["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    result["price_vs_ema200"] = (
        result["close"] - result["ema_200"]
    ) / result["ema_200"]

    result["ema9_vs_ema21"] = (
        result["ema_9"] - result["ema_21"]
    ) / result["ema_21"]

    result["ema21_vs_ema50"] = (
        result["ema_21"] - result["ema_50"]
    ) / result["ema_50"]

    result["rsi_14"] = calculate_rsi(
        result["close"],
        14,
    )

    result["rsi_7"] = calculate_rsi(
        result["close"],
        7,
    )

    result["atr_14"] = calculate_atr(
        result,
        14,
    )

    result["atr_pct"] = (
        result["atr_14"]
        / result["close"]
    )

    result["volatility_20d"] = (
        result["return_1d"]
        .rolling(20)
        .std()
    )

    result["volume_change"] = (
        result["volume"]
        .pct_change()
    )

    result["volume_ma_ratio"] = (
        result["volume"]
        / result["volume"]
        .rolling(20)
        .mean()
    )

    return result


def load_legacy_model():
    """
    Load the legacy model.

    First choice:
        model_old.pkl

    Fallback:
        models/model_old.pkl

    Final fallback:
        current model.pkl
    """

    candidates = [
        Path("model_old.pkl"),
        Path("models/model_old.pkl"),
        Path("model.pkl"),
        Path(MODEL_PATH),
    ]

    for path in candidates:
        if path.exists():
            with path.open("rb") as file:
                return pickle.load(file), path

    raise FileNotFoundError(
        "No legacy model was found."
    )


def append_signal(
    row: dict[str, object],
) -> None:
    LEGACY_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = LEGACY_LOG_PATH.exists()

    with LEGACY_LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_legacy_comparison() -> None:
    frame = get_market_data(
        symbol=SYMBOL,
        lookback_days=500,
    )

    featured = build_legacy_features(
        frame
    )

    latest = (
        featured
        .dropna(subset=LEGACY_FEATURES)
        .iloc[-1]
    )

    model, model_path = load_legacy_model()

    feature_row = pd.DataFrame(
        [latest[LEGACY_FEATURES]],
        columns=LEGACY_FEATURES,
    )

    prediction = int(
        model.predict(feature_row)[0]
    )

    probability_up = float(
        model.predict_proba(feature_row)[0][1]
    )

    hypothetical_action = (
        "BUY"
        if prediction == 1
        else "CASH"
    )

    signal = {
        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "symbol":
            SYMBOL,

        "model_path":
            str(model_path),

        "prediction":
            "UP"
            if prediction == 1
            else "DOWN",

        "probability_up":
            round(probability_up, 6),

        "hypothetical_action":
            hypothetical_action,

        "price":
            float(latest["close"]),
    }

    append_signal(signal)

    print("Legacy comparison signal")
    print(signal)
    print(
        "No order was submitted."
    )


if __name__ == "__main__":
    run_legacy_comparison()
