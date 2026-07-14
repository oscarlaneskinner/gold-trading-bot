"""Research-only GLD LightGBM/XGBoost ensemble comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config import HOLD_DAYS, SYMBOL
from data import get_market_data
from features import MODEL_FEATURES, add_features

LOOKBACK_DAYS = 5000
FOLDS = 6
MIN_TRAIN = 750
INITIAL_CAPITAL = 10_000.0
POSITION_PERCENT = 0.10
THRESHOLD = 0.50
SLIPPAGE = 0.0005
STOP = 0.10
TARGET = 0.20
MAX_HOLD = 20
RANDOM_STATE = 42

REPORT_DIR = Path("reports")
JSON_PATH = REPORT_DIR / "ensemble_comparison.json"
CSV_PATH = REPORT_DIR / "ensemble_comparison_folds.csv"

WEIGHTS = {
    "lightgbm": (1.0, 0.0),
    "xgboost": (0.0, 1.0),
    "ensemble_50_50": (0.5, 0.5),
    "ensemble_60_40": (0.6, 0.4),
    "ensemble_70_30": (0.7, 0.3),
}


def prepare():
    frame = add_features(
        get_market_data(symbol=SYMBOL, lookback_days=LOOKBACK_DAYS)
    )
    frame["future_close"] = frame["close"].shift(-HOLD_DAYS)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    return (
        frame.dropna(
            subset=MODEL_FEATURES
            + ["target", "timestamp", "open", "high", "low", "close"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def build_folds(frame):
    size = (len(frame) - MIN_TRAIN) // FOLDS
    result = []
    for n in range(FOLDS):
        start = MIN_TRAIN + n * size
        end = len(frame) if n == FOLDS - 1 else start + size
        train = frame.iloc[:start].copy()
        test = frame.iloc[start:end].copy()
        if train["target"].nunique() == 2 and test["target"].nunique() == 2:
            result.append((train, test))
    return result


def build_lightgbm():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def build_xgboost():
    return XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=10,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def positive_probability(model, X):
    classes = list(model.classes_)
    return np.asarray(
        model.predict_proba(X)[:, classes.index(1)],
        dtype=float,
    )


def simulate(test, probability_up):
    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = entry_value = entry_index = None
    trades = []
    curve = []
    invested = 0

    for i, ((_, row), probability) in enumerate(
        zip(test.iterrows(), probability_up)
    ):
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])

        if shares == 0:
            if probability >= THRESHOLD:
                allocation = cash * POSITION_PERCENT
                execution = close * (1 + SLIPPAGE)
                shares = allocation / execution
                cash -= allocation
                entry_price = execution
                entry_value = allocation
                entry_index = i
        else:
            invested += 1
            held = i - int(entry_index)
            stop_price = float(entry_price) * (1 - STOP)
            target_price = float(entry_price) * (1 + TARGET)
            exit_price = None

            if low <= stop_price:
                exit_price = stop_price
            elif high >= target_price:
                exit_price = target_price
            elif held >= MAX_HOLD:
                exit_price = close

            if exit_price is not None:
                proceeds = shares * exit_price * (1 - SLIPPAGE)
                cash += proceeds
                trades.append(
                    (proceeds - float(entry_value)) / float(entry_value)
                )
                shares = 0.0
                entry_price = entry_value = entry_index = None

        curve.append(cash + shares * close)

    if shares > 0:
        proceeds = shares * float(test.iloc[-1]["close"]) * (1 - SLIPPAGE)
        cash += proceeds
        trades.append(
            (proceeds - float(entry_value)) / float(entry_value)
        )
        curve[-1] = cash

    arr = np.asarray(curve, dtype=float)
    max_dd = (
        float((arr / np.maximum.accumulate(arr) - 1).min())
        if arr.size
        else 0.0
    )

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (math.inf if gross_profit > 0 else 0.0)
    )

    return {
        "total_return": float(cash / INITIAL_CAPITAL - 1),
        "maximum_drawdown": max_dd,
        "trades": len(trades),
        "win_rate": float(np.mean(np.asarray(trades) > 0)) if trades else 0.0,
        "profit_factor": float(profit_factor),
        "exposure": float(invested / max(1, len(test))),
    }


def summarize(rows):
    returns = np.asarray([r["total_return"] for r in rows], dtype=float)
    drawdowns = np.asarray([r["maximum_drawdown"] for r in rows], dtype=float)
    finite_pf = [
        min(float(r["profit_factor"]), 10.0)
        for r in rows
        if math.isfinite(float(r["profit_factor"]))
    ]

    return {
        "folds": len(rows),
        "positive_folds": int((returns > 0).sum()),
        "median_fold_return": float(np.median(returns)),
        "mean_fold_return": float(np.mean(returns)),
        "worst_fold_return": float(np.min(returns)),
        "best_fold_return": float(np.max(returns)),
        "total_trades": int(sum(r["trades"] for r in rows)),
        "mean_win_rate": float(np.mean([r["win_rate"] for r in rows])),
        "mean_profit_factor": float(np.mean(finite_pf)) if finite_pf else 0.0,
        "mean_exposure": float(np.mean([r["exposure"] for r in rows])),
        "mean_maximum_drawdown": float(np.mean(drawdowns)),
        "worst_maximum_drawdown": float(np.min(drawdowns)),
        "mean_model_accuracy": float(
            np.mean([r["model_accuracy"] for r in rows])
        ),
        "mean_model_roc_auc": float(
            np.mean([r["model_roc_auc"] for r in rows])
        ),
    }


def score(s):
    return float(
        4 * s["median_fold_return"]
        + 2 * s["mean_fold_return"]
        + 0.015 * s["positive_folds"]
        + 0.010 * min(3.0, s["mean_profit_factor"])
        + 0.10 * (s["mean_model_roc_auc"] - 0.50)
        + 0.015 * min(1.0, s["total_trades"] / 50.0)
        + 1.5 * s["worst_maximum_drawdown"]
    )


def run():
    frame = prepare()
    rows = []

    for fold_number, (train, test) in enumerate(build_folds(frame), start=1):
        print(
            f"\nFold {fold_number}: "
            f"{test.iloc[0]['timestamp']} through {test.iloc[-1]['timestamp']}"
        )

        lightgbm = build_lightgbm()
        xgboost = build_xgboost()

        lightgbm.fit(train[MODEL_FEATURES], train["target"])
        xgboost.fit(train[MODEL_FEATURES], train["target"])

        X_test = test[MODEL_FEATURES]
        y_test = test["target"]

        p_lgbm = positive_probability(lightgbm, X_test)
        p_xgb = positive_probability(xgboost, X_test)

        for name, (w_lgbm, w_xgb) in WEIGHTS.items():
            probability = w_lgbm * p_lgbm + w_xgb * p_xgb
            prediction = (probability >= THRESHOLD).astype(int)
            result = simulate(test, probability)

            row = {
                "fold": fold_number,
                "model": name,
                "lightgbm_weight": w_lgbm,
                "xgboost_weight": w_xgb,
                "model_accuracy": float(
                    accuracy_score(y_test, prediction)
                ),
                "model_roc_auc": float(
                    roc_auc_score(y_test, probability)
                ),
                **result,
            }
            rows.append(row)

            print(
                f"  {name}: "
                f"return={result['total_return']:.2%}, "
                f"trades={result['trades']}, "
                f"auc={row['model_roc_auc']:.3f}, "
                f"max_dd={result['maximum_drawdown']:.2%}"
            )

    summaries = {}

    for name in WEIGHTS:
        summary = summarize([r for r in rows if r["model"] == name])
        summary["research_score"] = score(summary)
        summary["weights"] = {
            "lightgbm": WEIGHTS[name][0],
            "xgboost": WEIGHTS[name][1],
        }
        summaries[name] = summary

    ranking = sorted(
        summaries,
        key=lambda name: summaries[name]["research_score"],
        reverse=True,
    )

    lightgbm = summaries["lightgbm"]
    winner = summaries[ranking[0]]

    rules = {
        "winner_is_ensemble": ranking[0].startswith("ensemble_"),
        "median_return_not_worse":
            winner["median_fold_return"] >= lightgbm["median_fold_return"],
        "mean_return_not_worse":
            winner["mean_fold_return"] >= lightgbm["mean_fold_return"],
        "positive_folds_not_worse":
            winner["positive_folds"] >= lightgbm["positive_folds"],
        "worst_drawdown_not_worse":
            winner["worst_maximum_drawdown"]
            >= lightgbm["worst_maximum_drawdown"],
        "trade_count_at_least_90_percent":
            winner["total_trades"] >= 0.90 * lightgbm["total_trades"],
    }

    passed = all(rules.values())

    report = {
        "features": MODEL_FEATURES,
        "summaries": summaries,
        "recommendation": {
            "research_winner": ranking[0],
            "ranking": ranking,
            "promotion_rules": rules,
            "promotion_passed": passed,
            "decision": (
                "Promote ensemble for further paper research."
                if passed
                else "Keep tuned LightGBM as the paper-trading model."
            ),
            "note": "Research only. This workflow cannot place orders.",
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    print("\n=== ENSEMBLE RESULT ===")
    print(json.dumps(report["recommendation"], indent=2))
    print(f"\nJSON report: {JSON_PATH}")
    print(f"CSV report: {CSV_PATH}")


if __name__ == "__main__":
    run()
