"""Simple chronological backtest for the production model and strategy filters."""

from __future__ import annotations
import json
import pickle
import numpy as np
from config import (
    BACKTEST_INITIAL_CAPITAL, BACKTEST_POSITION_PERCENT,
    BACKTEST_REPORT_PATH, BACKTEST_SLIPPAGE_PERCENT,
    HOLD_DAYS, MODEL_PATH, SYMBOL, TRAINING_LOOKBACK_DAYS,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from logger import write_json
from strategy import evaluate_entry

def run_backtest():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/model.pkl does not exist. Train a model first.")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    frame = add_features(get_market_data(SYMBOL, TRAINING_LOOKBACK_DAYS))
    frame = frame.dropna(subset=MODEL_FEATURES).reset_index(drop=True)

    cash = BACKTEST_INITIAL_CAPITAL
    shares = 0.0
    entry_index = None
    entry_value = None
    completed_returns = []
    equity_curve = []

    for index, row in frame.iterrows():
        price = float(row["close"])
        X = frame[MODEL_FEATURES].iloc[[index]]
        prediction = int(model.predict(X)[0])
        probability_up = float(model.predict_proba(X)[0][list(model.classes_).index(1)])

        if shares == 0:
            decision = evaluate_entry(prediction, probability_up, row)
            if decision.action == "BUY":
                allocation = cash * BACKTEST_POSITION_PERCENT
                execution_price = price * (1 + BACKTEST_SLIPPAGE_PERCENT)
                shares = allocation / execution_price
                cash -= allocation
                entry_index = index
                entry_value = allocation
        elif entry_index is not None and index - entry_index >= HOLD_DAYS:
            execution_price = price * (1 - BACKTEST_SLIPPAGE_PERCENT)
            proceeds = shares * execution_price
            cash += proceeds
            completed_returns.append((proceeds - entry_value) / entry_value)
            shares = 0.0
            entry_index = None
            entry_value = None

        equity_curve.append(cash + shares * price)

    final_value = cash + shares * float(frame.iloc[-1]["close"])
    curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    drawdowns = curve / running_max - 1

    metrics = {
        "symbol": SYMBOL,
        "starting_capital": BACKTEST_INITIAL_CAPITAL,
        "final_value": float(final_value),
        "total_return": float(final_value / BACKTEST_INITIAL_CAPITAL - 1),
        "completed_trades": len(completed_returns),
        "win_rate": float(np.mean(np.asarray(completed_returns) > 0)) if completed_returns else 0.0,
        "average_trade_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "maximum_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
    }
    write_json(BACKTEST_REPORT_PATH, metrics)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    run_backtest()
