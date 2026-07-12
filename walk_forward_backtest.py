"""
Proper walk-forward backtest: trains on data BEFORE each split date, tests only
on data AFTER it (never seen during training). Matches the actual live exit
logic in risk_manager.py (stop-loss, take-profit, trailing stop if enabled,
max hold days), unlike the original backtest.py which tested in-sample and
ignored the risk-manager exits entirely.

Run this any time you change config.py, features.py, or strategy.py, BEFORE
deploying the change to the live bot. Compare against previous runs' output
(saved to reports/walk_forward_metrics.json) to see if a change actually helped.
"""

from __future__ import annotations
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import (
    BACKTEST_REPORT_PATH, ENABLE_TRAILING_STOP, HOLD_DAYS, MAX_HOLD_DAYS,
    MIN_BUY_CONFIDENCE, RANDOM_FOREST_ESTIMATORS, RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MIN_SAMPLES_LEAF, RANDOM_STATE, RSI_MAXIMUM, RSI_MINIMUM,
    STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, TRAILING_ACTIVATION_PERCENT,
    TRAILING_STOP_PERCENT, USE_RSI_FILTER, USE_TREND_FILTER,
    USE_VOLATILITY_FILTER, MAX_ATR_PERCENT, SYMBOL, TRAINING_LOOKBACK_DAYS,
    create_project_directories,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from logger import write_json

# Realistic per-trade cost assumption (spread + modest slippage)
COST_PER_SIDE = 0.0005

# Walk-forward split dates - each trains only on data before this date,
# tests only on data from this date forward (genuinely unseen by that model)
SPLIT_DATES = ["2015-01-01", "2017-01-01", "2019-01-01", "2021-01-01", "2023-01-01"]


def entry_ok(row, prediction, probability_up):
    if prediction != 1 or probability_up < MIN_BUY_CONFIDENCE:
        return False
    if USE_TREND_FILTER:
        if not (row["close"] > row["ema_200"] and row["ema_9"] > row["ema_21"] > row["ema_50"]):
            return False
    if USE_RSI_FILTER:
        if not (RSI_MINIMUM <= row["rsi_14"] < RSI_MAXIMUM):
            return False
    if USE_VOLATILITY_FILTER:
        if row["atr_pct"] > MAX_ATR_PERCENT:
            return False
    return True


def simulate(test_df):
    """Simulate trading using the SAME exit rules as risk_manager.py / daily_bot.py."""
    capital = 100_000.0
    n_trades, wins = 0, 0
    exit_reasons = {"stop_loss": 0, "take_profit": 0, "trailing_stop": 0, "max_hold_days": 0}

    i, n = 0, len(test_df)
    while i < n - 1:
        row = test_df.iloc[i]
        if entry_ok(row, row["prediction"], row["probability_up"]):
            entry_price = row["close"]
            peak_price = entry_price
            exit_price, exit_reason, j = None, None, i

            for j in range(i + 1, min(i + 1 + MAX_HOLD_DAYS, n)):
                day = test_df.iloc[j]
                peak_price = max(peak_price, day["high"])

                if day["low"] <= entry_price * (1 - STOP_LOSS_PERCENT):
                    exit_price, exit_reason = entry_price * (1 - STOP_LOSS_PERCENT), "stop_loss"
                    break
                if day["high"] >= entry_price * (1 + TAKE_PROFIT_PERCENT):
                    exit_price, exit_reason = entry_price * (1 + TAKE_PROFIT_PERCENT), "take_profit"
                    break
                if ENABLE_TRAILING_STOP and peak_price >= entry_price * (1 + TRAILING_ACTIVATION_PERCENT):
                    trail_trigger = peak_price * (1 - TRAILING_STOP_PERCENT)
                    if day["low"] <= trail_trigger:
                        exit_price, exit_reason = trail_trigger, "trailing_stop"
                        break

            if exit_price is None:
                j = min(i + MAX_HOLD_DAYS, n - 1)
                exit_price, exit_reason = test_df.iloc[j]["close"], "max_hold_days"

            trade_return = (exit_price - entry_price) / entry_price - (COST_PER_SIDE * 2)
            capital *= (1 + trade_return)
            n_trades += 1
            exit_reasons[exit_reason] += 1
            if trade_return > 0:
                wins += 1
            i = j
        else:
            i += 1

    win_rate = (wins / n_trades * 100) if n_trades else 0.0
    total_return_pct = (capital / 100_000.0 - 1) * 100
    return total_return_pct, n_trades, win_rate, exit_reasons


def run():
    create_project_directories()
    raw = get_market_data(SYMBOL, TRAINING_LOOKBACK_DAYS + 400)
    full = add_features(raw)
    full["future_close"] = full["close"].shift(-HOLD_DAYS)
    full["target"] = (full["future_close"] > full["close"]).astype(int)
    clean = full.dropna(subset=MODEL_FEATURES + ["target"]).reset_index(drop=True)
    clean["timestamp"] = clean["timestamp"].dt.tz_localize(None)

    results, bh_results = [], []
    per_split = []

    for split_date in SPLIT_DATES:
        train = clean[clean["timestamp"] < split_date]
        test = clean[clean["timestamp"] >= split_date].copy()
        if len(train) < 500 or len(test) < 100:
            continue

        model = RandomForestClassifier(
            n_estimators=RANDOM_FOREST_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[MODEL_FEATURES], train["target"])

        test["prediction"] = model.predict(test[MODEL_FEATURES])
        test["probability_up"] = model.predict_proba(test[MODEL_FEATURES])[:, list(model.classes_).index(1)]

        ret, trades, win_rate, exit_reasons = simulate(test)
        bh_return = (test.iloc[-1]["close"] - test.iloc[0]["close"]) / test.iloc[0]["close"] * 100

        results.append(ret)
        bh_results.append(bh_return)
        per_split.append({
            "split_date": split_date, "strategy_return_pct": round(ret, 1),
            "buy_hold_return_pct": round(bh_return, 1), "trades": trades,
            "win_rate_pct": round(win_rate, 1), "exit_reasons": exit_reasons,
        })
        print(f"{split_date}: strategy {ret:>7.1f}% | buy&hold {bh_return:>7.1f}% | "
              f"{trades} trades | {win_rate:.0f}% win rate | exits={exit_reasons}")

    avg_strategy = float(np.mean(results)) if results else 0.0
    avg_bh = float(np.mean(bh_results)) if bh_results else 0.0
    print(f"\nAVERAGE: strategy {avg_strategy:.1f}% | buy&hold {avg_bh:.1f}%")
    if avg_strategy < avg_bh:
        print(f"NOTE: strategy underperformed buy-and-hold by {avg_bh - avg_strategy:.1f} points on average.")

    report = {
        "per_split": per_split,
        "average_strategy_return_pct": round(avg_strategy, 1),
        "average_buy_hold_return_pct": round(avg_bh, 1),
        "config_snapshot": {
            "hold_days": HOLD_DAYS, "max_hold_days": MAX_HOLD_DAYS,
            "min_buy_confidence": MIN_BUY_CONFIDENCE,
            "use_trend_filter": USE_TREND_FILTER, "use_rsi_filter": USE_RSI_FILTER,
            "use_volatility_filter": USE_VOLATILITY_FILTER,
            "stop_loss_percent": STOP_LOSS_PERCENT, "take_profit_percent": TAKE_PROFIT_PERCENT,
            "enable_trailing_stop": ENABLE_TRAILING_STOP,
        },
    }
    write_json(BACKTEST_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    run()
