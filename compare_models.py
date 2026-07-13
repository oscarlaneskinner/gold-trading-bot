"""Research-only walk-forward comparison: Random Forest, XGBoost, LightGBM, CatBoost."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from data import get_market_data
from features import MODEL_FEATURES, add_features

SYMBOL = "GLD"
LOOKBACK_DAYS = 5000
PREDICTION_HORIZON = 20
NUMBER_OF_TEST_FOLDS = 6
MINIMUM_TRAINING_ROWS = 750

INITIAL_CAPITAL = 10_000.0
POSITION_PERCENT = 0.10
CONFIDENCE_THRESHOLD = 0.50
SLIPPAGE_PERCENT = 0.0005
STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20
MAX_HOLD_DAYS = 20
RANDOM_STATE = 42

REPORTS_DIR = Path("reports")
JSON_REPORT_PATH = REPORTS_DIR / "model_comparison.json"
CSV_REPORT_PATH = REPORTS_DIR / "model_comparison_folds.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], Any]


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "xgboost",
            lambda: XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "lightgbm",
            lambda: LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=2.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        ModelSpec(
            "catboost",
            lambda: CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=5.0,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                thread_count=-1,
            ),
        ),
    ]


def prepare_dataset() -> pd.DataFrame:
    frame = add_features(get_market_data(SYMBOL, LOOKBACK_DAYS))
    frame["future_close"] = frame["close"].shift(-PREDICTION_HORIZON)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    frame = (
        frame.dropna(
            subset=MODEL_FEATURES + ["target", "timestamp", "open", "high", "low", "close"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return frame


def build_folds(frame: pd.DataFrame):
    remaining = len(frame) - MINIMUM_TRAINING_ROWS
    fold_size = remaining // NUMBER_OF_TEST_FOLDS
    folds = []

    for fold_number in range(NUMBER_OF_TEST_FOLDS):
        test_start = MINIMUM_TRAINING_ROWS + fold_number * fold_size
        test_end = len(frame) if fold_number == NUMBER_OF_TEST_FOLDS - 1 else test_start + fold_size
        train = frame.iloc[:test_start].copy()
        test = frame.iloc[test_start:test_end].copy()

        if train["target"].nunique() >= 2 and test["target"].nunique() >= 2:
            folds.append((train, test))

    if len(folds) < 4:
        raise RuntimeError("Fewer than four usable folds were created.")

    return folds


def positive_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def simulate(model, testing: pd.DataFrame) -> dict[str, Any]:
    X = testing[MODEL_FEATURES]
    predictions = np.asarray(model.predict(X), dtype=int)
    probabilities = positive_probabilities(model, X)

    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = None
    entry_value = None
    entry_index = None
    trade_returns = []
    equity_curve = []
    invested_days = 0

    for i, ((_, row), prediction, probability_up) in enumerate(
        zip(testing.iterrows(), predictions, probabilities)
    ):
        close_price = float(row["close"])
        low_price = float(row["low"])
        high_price = float(row["high"])

        if shares == 0:
            if int(prediction) == 1 and float(probability_up) >= CONFIDENCE_THRESHOLD:
                allocation = cash * POSITION_PERCENT
                execution_price = close_price * (1 + SLIPPAGE_PERCENT)
                shares = allocation / execution_price
                cash -= allocation
                entry_price = execution_price
                entry_value = allocation
                entry_index = i
        else:
            invested_days += 1
            days_held = i - int(entry_index)
            stop_level = float(entry_price) * (1 - STOP_LOSS_PERCENT)
            target_level = float(entry_price) * (1 + TAKE_PROFIT_PERCENT)
            exit_price = None

            if low_price <= stop_level:
                exit_price = stop_level
            elif high_price >= target_level:
                exit_price = target_level
            elif days_held >= MAX_HOLD_DAYS:
                exit_price = close_price

            if exit_price is not None:
                proceeds = shares * exit_price * (1 - SLIPPAGE_PERCENT)
                cash += proceeds
                trade_returns.append((proceeds - float(entry_value)) / float(entry_value))
                shares = 0.0
                entry_price = None
                entry_value = None
                entry_index = None

        equity_curve.append(cash + shares * close_price)

    if shares > 0:
        proceeds = shares * float(testing.iloc[-1]["close"]) * (1 - SLIPPAGE_PERCENT)
        cash += proceeds
        trade_returns.append((proceeds - float(entry_value)) / float(entry_value))
        if equity_curve:
            equity_curve[-1] = cash

    curve = np.asarray(equity_curve, dtype=float)
    max_drawdown = float((curve / np.maximum.accumulate(curve) - 1).min()) if curve.size else 0.0

    wins = [x for x in trade_returns if x > 0]
    losses = [x for x in trade_returns if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)

    return {
        "final_value": float(cash),
        "total_return": float(cash / INITIAL_CAPITAL - 1),
        "maximum_drawdown": max_drawdown,
        "trades": len(trade_returns),
        "win_rate": float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0,
        "average_trade_return": float(np.mean(trade_returns)) if trade_returns else 0.0,
        "profit_factor": float(profit_factor),
        "exposure": float(invested_days / max(1, len(testing))),
    }


def summarize(rows):
    returns = [float(r["total_return"]) for r in rows]
    drawdowns = [float(r["maximum_drawdown"]) for r in rows]
    finite_pf = [float(r["profit_factor"]) for r in rows if math.isfinite(float(r["profit_factor"]))]

    return {
        "folds": len(rows),
        "positive_folds": sum(x > 0 for x in returns),
        "median_fold_return": float(np.median(returns)),
        "mean_fold_return": float(np.mean(returns)),
        "worst_fold_return": min(returns),
        "best_fold_return": max(returns),
        "total_trades": sum(int(r["trades"]) for r in rows),
        "mean_win_rate": float(np.mean([float(r["win_rate"]) for r in rows])),
        "mean_profit_factor": float(np.mean(finite_pf)) if finite_pf else 0.0,
        "mean_exposure": float(np.mean([float(r["exposure"]) for r in rows])),
        "mean_maximum_drawdown": float(np.mean(drawdowns)),
        "worst_maximum_drawdown": min(drawdowns),
        "mean_model_accuracy": float(np.mean([float(r["model_accuracy"]) for r in rows])),
        "mean_model_roc_auc": float(np.mean([float(r["model_roc_auc"]) for r in rows])),
    }


def score(summary):
    trade_score = min(1.0, summary["total_trades"] / 50.0)
    return float(
        4.0 * summary["median_fold_return"]
        + 2.0 * summary["mean_fold_return"]
        + 0.015 * summary["positive_folds"]
        + 0.010 * min(3.0, summary["mean_profit_factor"])
        + 0.10 * (summary["mean_model_roc_auc"] - 0.50)
        + 0.015 * trade_score
        + 1.5 * summary["worst_maximum_drawdown"]
    )


def run_comparison():
    frame = prepare_dataset()
    folds = build_folds(frame)
    rows = []

    for fold_index, (training, testing) in enumerate(folds, start=1):
        print(f"\nFold {fold_index}: {testing.iloc[0]['timestamp']} through {testing.iloc[-1]['timestamp']}")

        for spec in model_specs():
            print(f"  Training {spec.name}...")
            model = spec.builder()
            model.fit(training[MODEL_FEATURES], training["target"])

            prediction = np.asarray(model.predict(testing[MODEL_FEATURES]), dtype=int)
            probability = positive_probabilities(model, testing[MODEL_FEATURES])

            validation = {
                "accuracy": float(accuracy_score(testing["target"], prediction)),
                "roc_auc": float(roc_auc_score(testing["target"], probability)),
            }
            strategy = simulate(model, testing)

            row = {
                "fold": fold_index,
                "model": spec.name,
                "test_start": str(testing.iloc[0]["timestamp"]),
                "test_end": str(testing.iloc[-1]["timestamp"]),
                "training_rows": len(training),
                "testing_rows": len(testing),
                "model_accuracy": validation["accuracy"],
                "model_roc_auc": validation["roc_auc"],
                **strategy,
            }
            rows.append(row)

            print(
                f"    return={strategy['total_return']:.2%}, "
                f"trades={strategy['trades']}, "
                f"auc={validation['roc_auc']:.3f}, "
                f"max_dd={strategy['maximum_drawdown']:.2%}"
            )

    summaries = {}
    for spec in model_specs():
        model_rows = [r for r in rows if r["model"] == spec.name]
        summary = summarize(model_rows)
        summary["research_score"] = score(summary)
        summaries[spec.name] = summary

    ranking = sorted(summaries, key=lambda name: summaries[name]["research_score"], reverse=True)

    report = {
        "symbol": SYMBOL,
        "prediction_horizon_days": PREDICTION_HORIZON,
        "features": MODEL_FEATURES,
        "fold_results": rows,
        "summaries": summaries,
        "recommendation": {
            "research_winner": ranking[0],
            "ranking": ranking,
            "note": "Research ranking only; do not deploy without reviewing trade count, fold consistency, and drawdown.",
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(CSV_REPORT_PATH, index=False)

    print("\n=== MODEL SUMMARIES ===")
    print(json.dumps(summaries, indent=2))
    print("\n=== RESEARCH RANKING ===")
    print(json.dumps(report["recommendation"], indent=2))
    print(f"\nJSON report: {JSON_REPORT_PATH}")
    print(f"CSV report: {CSV_REPORT_PATH}")


if __name__ == "__main__":
    run_comparison()
