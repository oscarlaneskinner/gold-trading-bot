"""Run one adaptive-limit GLD paper decision."""

import json
from datetime import datetime, timezone

from broker import (
    close_position,
    create_trading_client,
    get_account_equity,
    get_filled_buy_orders,
    get_position,
)
from broker_v5 import submit_limit_buy
from config import (
    MINIMUM_ACCOUNT_EQUITY,
    PAPER_TRADING,
    SYMBOL,
    create_project_directories,
)
from data import get_market_data
from execution import build_limit_plan
from features import MODEL_FEATURES, add_features
from logger import log_decision, log_trade
from model_loader import load_active_model
from order_manager import cancel_stale_buy_orders, get_open_buy_orders
from risk_manager import calculate_notional, check_exit_conditions
from strategy import evaluate_entry, model_exit_signal

STALE_ORDER_MINUTES = 120


def latest_entry_information(client):
    orders = get_filled_buy_orders(client, SYMBOL)
    if not orders:
        return None, None

    order = max(orders, key=lambda item: item.filled_at)
    return order.filled_at, float(order.filled_avg_price)


def run():
    create_project_directories()

    print(f"=== GLD AI limit bot v5: {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Paper trading: {PAPER_TRADING}")

    frame = add_features(get_market_data())
    usable = frame.dropna(subset=MODEL_FEATURES)

    if usable.empty:
        raise RuntimeError("No complete feature row is available.")

    latest = usable.iloc[-1]
    X = usable[MODEL_FEATURES].iloc[[-1]]

    model, model_info = load_active_model()
    prediction = int(model.predict(X)[0])
    probability_up = float(
        model.predict_proba(X)[0][list(model.classes_).index(1)]
    )

    price = float(latest["close"])
    atr_percent = float(latest["atr_pct"])

    client = create_trading_client()
    clock = client.get_clock()

    cancelled = cancel_stale_buy_orders(
        client,
        SYMBOL,
        STALE_ORDER_MINUTES,
    )

    position = get_position(client, SYMBOL)
    open_buys = get_open_buy_orders(client, SYMBOL)

    action = "HOLD"
    reason = "No action."
    order_id = None
    limit_price = None

    if position is None:
        if open_buys:
            reason = "An open GLD buy order already exists."
        else:
            decision = evaluate_entry(prediction, probability_up, latest)
            action, reason = decision.action, decision.reason

            if action == "BUY":
                if not clock.is_open:
                    action = "HOLD"
                    reason = "Bullish signal detected, but the market is closed."
                else:
                    equity = get_account_equity(client)

                    if equity < MINIMUM_ACCOUNT_EQUITY:
                        action = "HOLD"
                        reason = "Account equity is below the configured minimum."
                    else:
                        plan = build_limit_plan(
                            price,
                            atr_percent,
                            probability_up,
                        )

                        notional = calculate_notional(
                            equity,
                            probability_up,
                        )

                        order = submit_limit_buy(
                            client,
                            SYMBOL,
                            notional,
                            plan.limit_price,
                        )

                        order_id = str(order.id)
                        limit_price = plan.limit_price
                        reason = (
                            f"{decision.reason} Adaptive limit "
                            f"${limit_price:.2f}; band {plan.confidence_band}."
                        )

                        log_trade(
                            symbol=SYMBOL,
                            action="BUY_LIMIT",
                            price=price,
                            notional=notional,
                            probability_up=probability_up,
                            order_id=order_id,
                            paper_trading=PAPER_TRADING,
                        )

    else:
        entry_date, entry_price = latest_entry_information(client)

        if entry_date is None or entry_price is None:
            reason = "Open position found, but entry order could not be identified."
        else:
            since_entry = usable[usable["timestamp"] >= entry_date]
            days_held = max(0, len(since_entry) - 1)
            peak_price = max(entry_price, float(since_entry["high"].max()))

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
                reason = "Position remains within exit rules."

            if action == "SELL":
                if not clock.is_open:
                    action = "HOLD"
                    reason = "Exit condition detected, but market is closed."
                else:
                    order = close_position(client, SYMBOL)
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

    print(json.dumps({
        "symbol": SYMBOL,
        "model_name": model_info["model_name"],
        "model_version": model_info.get("model_version"),
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "price": price,
        "atr_percent": atr_percent,
        "market_open": bool(clock.is_open),
        "action": action,
        "reason": reason,
        "limit_price": limit_price,
        "order_id": order_id,
        "cancelled_stale_orders": cancelled,
        "paper_trading": PAPER_TRADING,
    }, indent=2))


if __name__ == "__main__":
    run()
