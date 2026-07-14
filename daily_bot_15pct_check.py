"""
Read-only 15% GLD paper-sizing check.

This script:
- loads the verified LightGBM model
- generates the current GLD prediction
- connects to Alpaca Paper Trading
- checks for an existing GLD position and open GLD buy orders
- calculates the proposed fixed 15% notional
- never submits, cancels, replaces, or closes an order
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from broker import (
    create_trading_client,
    get_account_equity,
    get_position,
)
from config import (
    MIN_ORDER_AMOUNT,
    PAPER_TRADING,
    SYMBOL,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from model_loader import load_active_model
from position_sizing import (
    calculate_fixed_notional,
    get_position_percent,
)


def get_open_buy_orders(client, symbol: str):
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
    )

    orders = client.get_orders(
        filter=request
    )

    return [
        order
        for order in orders
        if order.symbol == symbol
        and order.side == OrderSide.BUY
    ]


def run() -> None:
    if not PAPER_TRADING:
        raise RuntimeError(
            "This checker is restricted to PAPER_TRADING=True."
        )

    frame = add_features(
        get_market_data()
    )

    frame[MODEL_FEATURES] = frame[
        MODEL_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    usable = frame.dropna(
        subset=MODEL_FEATURES
    )

    if usable.empty:
        raise RuntimeError(
            "No complete feature row is available."
        )

    latest = usable.iloc[-1]
    model_input = usable[
        MODEL_FEATURES
    ].iloc[[-1]]

    model, model_info = load_active_model()

    prediction = int(
        model.predict(model_input)[0]
    )

    positive_index = list(
        model.classes_
    ).index(1)

    probability_up = float(
        model.predict_proba(model_input)[
            0,
            positive_index,
        ]
    )

    client = create_trading_client()
    account_equity = get_account_equity(client)
    position = get_position(client, SYMBOL)
    open_buy_orders = get_open_buy_orders(
        client,
        SYMBOL,
    )

    position_percent = get_position_percent()

    proposed_notional = calculate_fixed_notional(
        account_equity=account_equity,
        position_percent=position_percent,
        minimum_order_amount=MIN_ORDER_AMOUNT,
    )

    blocked_reasons = []

    if position is not None:
        blocked_reasons.append(
            "An open GLD position already exists."
        )

    if open_buy_orders:
        blocked_reasons.append(
            "An open GLD buy order already exists."
        )

    if prediction != 1:
        blocked_reasons.append(
            "The model is not currently bullish."
        )

    result = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "symbol": SYMBOL,
        "paper_trading": PAPER_TRADING,
        "model_name": model_info[
            "model_name"
        ],
        "model_version": model_info.get(
            "model_version"
        ),
        "prediction": (
            "UP"
            if prediction == 1
            else "DOWN"
        ),
        "probability_up": probability_up,
        "latest_price": float(
            latest["close"]
        ),
        "account_equity": account_equity,
        "position_percent": position_percent,
        "proposed_notional": proposed_notional,
        "existing_position": (
            position is not None
        ),
        "open_buy_order_count": len(
            open_buy_orders
        ),
        "eligible_for_future_15pct_entry": (
            not blocked_reasons
        ),
        "blocked_reasons": blocked_reasons,
        "order_submitted": False,
    }

    print(
        "GLD fixed-15% read-only sizing check"
    )
    print(
        json.dumps(
            result,
            indent=2,
        )
    )
    print("No order was submitted.")


if __name__ == "__main__":
    run()
