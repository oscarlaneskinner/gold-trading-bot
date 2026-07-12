"""Run one paper-trading decision for GLD."""

from __future__ import annotations
import json
import pickle
from datetime import datetime, timezone
from broker import (
    close_position, create_trading_client, get_account_equity,
    get_filled_buy_orders, get_position, submit_market_buy,
)
from config import (
    MINIMUM_ACCOUNT_EQUITY, MODEL_PATH, PAPER_TRADING, SYMBOL,
    create_project_directories,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from logger import log_decision, log_trade
from risk_manager import calculate_notional, check_exit_conditions
from strategy import evaluate_entry, model_exit_signal

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    expected = list(getattr(model, "feature_names_in_", []))
    if expected and expected != MODEL_FEATURES:
        raise RuntimeError("The model feature list does not match features.py.")
    return model

def latest_entry_information(client):
    orders = get_filled_buy_orders(client, SYMBOL)
    if not orders:
        return None, None
    order = max(orders, key=lambda item: item.filled_at)
    return order.filled_at, float(order.filled_avg_price)

def run():
    create_project_directories()
    print(f"=== GLD AI bot: {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Paper trading: {PAPER_TRADING}")

    frame = add_features(get_market_data())
    usable = frame.dropna(subset=MODEL_FEATURES)
    if usable.empty:
        raise RuntimeError("No complete feature row is available.")

    latest = usable.iloc[-1]
    X = usable[MODEL_FEATURES].iloc[[-1]]
    model = load_model()
    prediction = int(model.predict(X)[0])
    probability_up = float(model.predict_proba(X)[0][list(model.classes_).index(1)])
    price = float(latest["close"])

    client = create_trading_client()
    position = get_position(client, SYMBOL)
    action, reason, order_id = "HOLD", "No action.", None

    if position is None:
        decision = evaluate_entry(prediction, probability_up, latest)
        action, reason = decision.action, decision.reason
        if action == "BUY":
            equity = get_account_equity(client)
            if equity < MINIMUM_ACCOUNT_EQUITY:
                action = "HOLD"
                reason = f"Account equity ${equity:.2f} is below the configured minimum."
            else:
                notional = calculate_notional(equity, probability_up)
                order = submit_market_buy(client, SYMBOL, notional)
                order_id = str(order.id)
                log_trade(
                    symbol=SYMBOL, action="BUY", price=price, notional=notional,
                    probability_up=probability_up, order_id=order_id,
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
            risk_exit = check_exit_conditions(entry_price, price, peak_price, days_held)
            model_exit = model_exit_signal(probability_up)

            if risk_exit.should_exit:
                action, reason = "SELL", risk_exit.reason or "risk_exit"
            elif model_exit.action == "SELL":
                action, reason = "SELL", model_exit.reason
            else:
                action, reason = "HOLD", "Position remains within exit rules."

            if action == "SELL":
                order = close_position(client, SYMBOL)
                order_id = str(order.id)
                log_trade(
                    symbol=SYMBOL, action="SELL", price=price,
                    probability_up=probability_up, reason=reason,
                    order_id=order_id, paper_trading=PAPER_TRADING,
                )

    log_decision(
        symbol=SYMBOL, prediction=prediction,
        probability_up=round(probability_up, 6), price=price,
        action=action, reason=reason, order_id=order_id,
        paper_trading=PAPER_TRADING,
    )

    print(json.dumps({
        "symbol": SYMBOL,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "price": price,
        "action": action,
        "reason": reason,
        "order_id": order_id,
    }, indent=2))

if __name__ == "__main__":
    run()
