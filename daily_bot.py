"""Run one controlled paper-trading decision for GLD."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

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
    POSITION_PERCENT,
    SYMBOL,
    create_project_directories,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from logger import log_decision, log_trade
from model_loader import load_active_model
from risk_manager import calculate_notional, check_exit_conditions
from strategy import evaluate_entry, model_exit_signal
from trade_intelligence import record_decision_intelligence
from trade_memory import record_decision
from trade_memory_sync import synchronize_filled_trades


def get_open_buy_orders(client, symbol: str):
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
    )
    orders = client.get_orders(filter=request)

    return [
        order
        for order in orders
        if order.symbol == symbol
        and order.side == OrderSide.BUY
    ]


def latest_entry_information(client):
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


def feature_snapshot(row) -> dict[str, float]:
    return {
        feature_name: float(row[feature_name])
        for feature_name in MODEL_FEATURES
    }


def save_memory_safely(
    *,
    latest,
    model_info,
    prediction: int,
    probability_up: float,
    price: float,
    action: str,
    reason: str,
    notional: float | None,
    market_open: bool,
    existing_position: bool,
    open_buy_order_count: int,
    order_id: str | None,
) -> tuple[int | None, dict | None]:
    try:
        market_timestamp = latest.get("timestamp")
        features = feature_snapshot(latest)

        decision_id = record_decision(
            symbol=SYMBOL,
            model_name=model_info["model_name"],
            model_version=model_info.get("model_version"),
            prediction=prediction,
            probability_up=probability_up,
            price=price,
            action=action,
            reason=reason,
            position_percent=POSITION_PERCENT,
            notional=notional,
            market_open=market_open,
            existing_position=existing_position,
            open_buy_order_count=open_buy_order_count,
            order_id=order_id,
            paper_trading=PAPER_TRADING,
            market_timestamp=(
                str(market_timestamp)
                if market_timestamp is not None
                else None
            ),
            features=features,
        )

        intelligence = record_decision_intelligence(
            decision_id=decision_id,
            probability_up=probability_up,
            features=features,
        )

        print(
            f"Trade-memory decision saved with id {decision_id}."
        )
        print(
            "Trade intelligence: "
            f"{json.dumps(intelligence, sort_keys=True)}"
        )

        return decision_id, intelligence

    except Exception as error:
        print(
            "WARNING: The trading decision completed, but "
            f"trade-memory storage failed: {error}"
        )
        return None, None


def synchronize_safely(client, stage: str) -> dict | None:
    try:
        result = synchronize_filled_trades(
            client,
            SYMBOL,
        )
        print(
            f"Trade-memory synchronization ({stage}): "
            f"{json.dumps(result, sort_keys=True)}"
        )
        return result

    except Exception as error:
        print(
            f"WARNING: Trade-memory synchronization "
            f"failed during {stage}: {error}"
        )
        return None


def run():
    create_project_directories()

    if not PAPER_TRADING:
        raise RuntimeError(
            "This bot is restricted to paper trading."
        )

    print(
        "=== GLD AI bot: "
        f"{datetime.now(timezone.utc).isoformat()} ==="
    )
    print(f"Paper trading: {PAPER_TRADING}")
    print(
        "Configured fixed position size: "
        f"{POSITION_PERCENT:.1%}"
    )

    frame = add_features(get_market_data())
    usable = frame.dropna(subset=MODEL_FEATURES)

    if usable.empty:
        raise RuntimeError(
            "No complete feature row is available."
        )

    latest = usable.iloc[-1]
    model_input = usable[MODEL_FEATURES].iloc[[-1]]

    model, model_info = load_active_model()
    prediction = int(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)

    positive_class_index = list(model.classes_).index(1)
    probability_up = float(
        probabilities[0, positive_class_index]
    )

    price = float(latest["close"])

    print(f"Using model: {model_info['model_name']}")
    print(f"Version: {model_info.get('model_version')}")

    client = create_trading_client()
    synchronize_before = synchronize_safely(
        client,
        "before_decision",
    )

    clock = client.get_clock()
    position = get_position(client, SYMBOL)
    open_buy_orders = get_open_buy_orders(client, SYMBOL)

    action = "HOLD"
    reason = "No action."
    order_id = None
    notional = None

    if position is None:
        if open_buy_orders:
            reason = "An open GLD buy order already exists."
        else:
            decision = evaluate_entry(
                prediction,
                probability_up,
                latest,
            )
            action = decision.action
            reason = decision.reason

            if action == "BUY":
                if not clock.is_open:
                    action = "HOLD"
                    reason = (
                        "Bullish signal detected, "
                        "but the market is closed."
                    )
                else:
                    equity = get_account_equity(client)

                    if equity < MINIMUM_ACCOUNT_EQUITY:
                        action = "HOLD"
                        reason = (
                            f"Account equity ${equity:.2f} is below "
                            "the configured minimum."
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
                        order_id = str(order.id)

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
        entry_date, entry_price = latest_entry_information(
            client
        )

        if entry_date is None or entry_price is None:
            reason = (
                "Open position found, but entry "
                "order could not be identified."
            )
        else:
            since_entry = usable[
                usable["timestamp"] >= entry_date
            ]

            days_held = max(0, len(since_entry) - 1)
            peak_price = max(
                entry_price,
                float(since_entry["high"].max()),
            )

            risk_exit = check_exit_conditions(
                entry_price,
                price,
                peak_price,
                days_held,
            )
            model_exit = model_exit_signal(probability_up)

            if risk_exit.should_exit:
                action = "SELL"
                reason = risk_exit.reason or "risk_exit"
            elif model_exit.action == "SELL":
                action = "SELL"
                reason = model_exit.reason
            else:
                action = "HOLD"
                reason = (
                    "Position remains within exit rules."
                )

            if action == "SELL":
                if not clock.is_open:
                    action = "HOLD"
                    reason = (
                        "Exit condition detected, "
                        "but the market is closed."
                    )
                else:
                    order = close_position(
                        client,
                        SYMBOL,
                    )
                    order_id = str(order.id)

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
        probability_up=round(probability_up, 6),
        price=price,
        action=action,
        reason=reason,
        order_id=order_id,
        paper_trading=PAPER_TRADING,
    )

    memory_decision_id, intelligence = save_memory_safely(
        latest=latest,
        model_info=model_info,
        prediction=prediction,
        probability_up=probability_up,
        price=price,
        action=action,
        reason=reason,
        notional=notional,
        market_open=bool(clock.is_open),
        existing_position=position is not None,
        open_buy_order_count=len(open_buy_orders),
        order_id=order_id,
    )

    synchronize_after = synchronize_safely(
        client,
        "after_decision",
    )

    output = {
        "symbol": SYMBOL,
        "model_name": model_info["model_name"],
        "model_version": model_info.get("model_version"),
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "price": price,
        "position_percent": POSITION_PERCENT,
        "notional": notional,
        "market_open": bool(clock.is_open),
        "existing_position": position is not None,
        "open_buy_order_count": len(open_buy_orders),
        "action": action,
        "reason": reason,
        "order_id": order_id,
        "trade_memory_decision_id": memory_decision_id,
        "trade_intelligence": intelligence,
        "trade_sync_before": synchronize_before,
        "trade_sync_after": synchronize_after,
        "paper_trading": PAPER_TRADING,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run()
