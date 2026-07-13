"""
Verify the tuned LightGBM candidate and generate one read-only prediction.

This script never imports the broker and cannot place an order.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from config import SYMBOL
from data import get_market_data
from features import MODEL_FEATURES, add_features
from model_loader import load_active_model


LOOKBACK_DAYS = 450


def run() -> dict:
    model, model_info = load_active_model()

    frame = add_features(
        get_market_data(
            symbol=SYMBOL,
            lookback_days=LOOKBACK_DAYS,
        )
    )

    frame[MODEL_FEATURES] = frame[
        MODEL_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = frame.dropna(
        subset=MODEL_FEATURES
    )

    if frame.empty:
        raise RuntimeError(
            "No complete feature row was available."
        )

    latest = frame.iloc[-1]
    features = frame[
        MODEL_FEATURES
    ].iloc[[-1]]

    prediction = int(
        model.predict(features)[0]
    )

    probabilities = model.predict_proba(
        features
    )

    positive_index = list(
        model.classes_
    ).index(1)

    probability_up = float(
        probabilities[
            0,
            positive_index,
        ]
    )

    result = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "symbol": SYMBOL,
        "market_timestamp": str(
            latest["timestamp"]
        ),
        "price": float(
            latest["close"]
        ),
        "model": model_info,
        "prediction": (
            "UP"
            if prediction == 1
            else "DOWN"
        ),
        "probability_up": probability_up,
        "hypothetical_action": (
            "BUY"
            if prediction == 1
            else "HOLD"
        ),
        "order_submitted": False,
    }

    print(
        "LightGBM candidate inference test"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "No order was submitted."
    )

    return result


if __name__ == "__main__":
    run()
