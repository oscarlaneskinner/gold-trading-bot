"""Compare market entries with adaptive limit entries."""

import json
import numpy as np

from config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_POSITION_PERCENT,
    BACKTEST_SLIPPAGE_PERCENT,
    HOLD_DAYS,
    SYMBOL,
    TRAINING_LOOKBACK_DAYS,
)
from data import get_market_data
from execution import build_limit_plan
from features import MODEL_FEATURES, add_features
from model_loader import load_active_model
from strategy import evaluate_entry


def simulate(frame, model, mode: str):
    cash = BACKTEST_INITIAL_CAPITAL
    shares = 0.0
    entry_index = None
    entry_value = None
    trades = []
    missed_entries = 0
    equity_curve = []

    for index, row in frame.iterrows():
        close_price = float(row["close"])
        X = frame[MODEL_FEATURES].iloc[[index]]
        prediction = int(model.predict(X)[0])
        probability_up = float(
            model.predict_proba(X)[0][list(model.classes_).index(1)]
        )

        if shares == 0:
            decision = evaluate_entry(prediction, probability_up, row)

            if decision.action == "BUY":
                allocation = cash * BACKTEST_POSITION_PERCENT

                if mode == "market":
                    execution_price = close_price * (1 + BACKTEST_SLIPPAGE_PERCENT)
                else:
                    if index + 1 >= len(frame):
                        missed_entries += 1
                        equity_curve.append(cash)
                        continue

                    plan = build_limit_plan(
                        close_price,
                        float(row["atr_pct"]),
                        probability_up,
                    )

                    next_low = float(frame.iloc[index + 1]["low"])

                    if next_low > plan.limit_price:
                        missed_entries += 1
                        equity_curve.append(cash)
                        continue

                    execution_price = plan.limit_price

                shares = allocation / execution_price
                cash -= allocation
                entry_index = index
                entry_value = allocation

        elif entry_index is not None and index - entry_index >= HOLD_DAYS:
            execution_price = close_price * (1 - BACKTEST_SLIPPAGE_PERCENT)
            proceeds = shares * execution_price
            cash += proceeds
            trades.append((proceeds - entry_value) / entry_value)
            shares = 0.0
            entry_index = None
            entry_value = None

        equity_curve.append(cash + shares * close_price)

    final_value = cash + shares * float(frame.iloc[-1]["close"])
    curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    drawdowns = curve / running_max - 1

    return {
        "mode": mode,
        "final_value": float(final_value),
        "total_return": float(final_value / BACKTEST_INITIAL_CAPITAL - 1),
        "completed_trades": len(trades),
        "missed_entries": missed_entries,
        "win_rate": float(np.mean(np.asarray(trades) > 0)) if trades else 0.0,
        "maximum_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
    }


def run():
    model, model_info = load_active_model()

    frame = add_features(
        get_market_data(SYMBOL, TRAINING_LOOKBACK_DAYS)
    ).dropna(subset=MODEL_FEATURES).reset_index(drop=True)

    market = simulate(frame, model, "market")
    adaptive_limit = simulate(frame, model, "adaptive_limit")

    print(json.dumps({
        "model": model_info,
        "market": market,
        "adaptive_limit": adaptive_limit,
        "difference": {
            "return_difference":
                adaptive_limit["total_return"] - market["total_return"],
            "trade_difference":
                adaptive_limit["completed_trades"] - market["completed_trades"],
        },
    }, indent=2))


if __name__ == "__main__":
    run()
