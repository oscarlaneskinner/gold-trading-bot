"""Run one paper-trading decision for GLD."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from broker import (
    close_position,
    create_trading_client,
    get_account_equity,
    get_filled_buy_orders,
    get_position,
    submit_market_buy,
)
from config import (
    MINIMUM_ACCOUNT_EQUITY,
    PAPER_TRADING,
    SYMBOL,
    create_project_directories,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from logger import log_decision, log_trade
from model_loader import load_active_model
from risk_manager import calculate_notional, check_exit_conditions
from strategy import evaluate_entry, model_exit_signal


def latest_entry_information(client):
    """Return the latest filled buy order date and average fill price."""

    orders = get_filled_buy_orders(client, SYMBOL)

    if not orders:
        return None, None

    order = max(
        orders,
        key=lambda item: item.filled_at,
    )

    return (
        order.filled_at,
        float(order.filled_avg_price),
    )


def run():
    """Run one GLD paper-trading decision."""

    create_project_directories()

    print(
        f"=== GLD AI bot: "
        f"{datetime.now(timezone.utc).isoformat()} ==="
    )
    print(f"Paper trading: {PAPER_TRADING}")

    frame = add_features(
        get_market_data()
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

    probabilities = model.predict_proba(
        model_input
    )

    positive_class_index = list(
        model.classes_
    ).index(1)

    probability_up = float(
        probabilities[
            0,
            positive_class_index,
        ]
    )

    price = float(
        latest["close"]
    )

    print(
        f"Using model: "
        f"{model_info['model_name']}"
    )
    print(
        f"Version: "
        f"{model_info.get('model_version')}"
    )

    client = create_trading_client()

    position = get_position(
        client,
        SYMBOL,
    )

    action = "HOLD"
    reason = "No action."
    order_id = None

    if position is None:
        decision = evaluate_entry(
            prediction,
            probability_up,
            latest,
        )

        action = decision.action
        reason = decision.reason

        if action == "BUY":
            equity = get_account_equity(
                client
            )

            if equity < MINIMUM_ACCOUNT_EQUITY:
                action = "HOLD"
                reason = (
                    f"Account equity ${equity:.2f} "
                    "is below the configured minimum."
                )

            else:
                notional = calculate_notional(
                    equity,
                    probability_up,
                )

                order = submit_market_buy(
                    client,
                    SYMBOL,
                    notional,
                )

                order_id = str(
                    order.id
                )

                log_trade(
                    symbol=SYMBOL,
                    action="BUY",
                    price=price,
                    notional=notional,
                    probability_up=probability_up,
                    order_id=order_id,
                    paper_trading=PAPER_TRADING,
                )

    else:
        entry_date, entry_price = (
            latest_entry_information(
                client
            )
        )

        if (
            entry_date is None
            or entry_price is None
        ):
            reason = (
                "Open position found, but entry "
                "order could not be identified."
            )

        else:
            since_entry = usable[
                usable["timestamp"]
                >= entry_date
            ]

            days_held = max(
                0,
                len(since_entry) - 1,
            )

            peak_price = max(
                entry_price,
                float(
                    since_entry[
                        "high"
                    ].max()
                ),
            )

            risk_exit = (
                check_exit_conditions(
                    entry_price,
                    price,
                    peak_price,
                    days_held,
                )
            )

            model_exit = model_exit_signal(
                probability_up
            )

            if risk_exit.should_exit:
                action = "SELL"
                reason = (
                    risk_exit.reason
                    or "risk_exit"
                )

            elif model_exit.action == "SELL":
                action = "SELL"
                reason = model_exit.reason

            else:
                action = "HOLD"
                reason = (
                    "Position remains within "
                    "exit rules."
                )

            if action == "SELL":
                order = close_position(
                    client,
                    SYMBOL,
                )

                order_id = str(
                    order.id
                )

                log_trade(
                    symbol=SYMBOL,
                    action="SELL",
                    price=price,
                    probability_up=probability_up,
                    reason=reason,
                    order_id=order_id,
                    paper_trading=PAPER_TRADING,
                )

    log_decision(
        symbol=SYMBOL,
        prediction=prediction,
        probability_up=round(
            probability_up,
            6,
        ),
        price=price,
        action=action,
        reason=reason,
        order_id=order_id,
        paper_trading=PAPER_TRADING,
    )

    output = {
        "symbol": SYMBOL,
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
        "price": price,
        "action": action,
        "reason": reason,
        "order_id": order_id,
        "paper_trading": PAPER_TRADING,
    }

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()