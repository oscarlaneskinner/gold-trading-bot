"""
Research-only GLD trade-quality filter comparison.

Keeps the tuned LightGBM model and the existing 13 production features.
Tests whether a small decision layer can improve trading results without
changing the model itself.

Candidate policies:
- baseline: probability >= 0.50
- confidence_55: probability >= 0.55
- confidence_60: probability >= 0.60
- trend: probability >= 0.50 and close > EMA 200
- rsi_band: probability >= 0.50 and 40 <= RSI 14 <= 70
- volatility_cap: probability >= 0.50 and ATR percent <= 3.5%
- quality_balanced:
    probability >= 0.55
    close > EMA 200
    RSI 14 between 40 and 72
    ATR percent <= 4.0%
- quality_selective:
    probability >= 0.60
    close > EMA 200
    RSI 14 between 45 and 68
    ATR percent <= 3.5%

Uses six walk-forward folds and the same exit assumptions used in prior
research. This script cannot place orders.

Outputs:
- reports/trade_quality_comparison.json
- reports/trade_quality_comparison_folds.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from config import HOLD_DAYS, SYMBOL
from data import get_market_data
from features import MODEL_FEATURES, add_features


LOOKBACK_DAYS = 5000
NUMBER_OF_FOLDS = 6
MINIMUM_TRAINING_ROWS = 750

INITIAL_CAPITAL = 10_000.0
POSITION_PERCENT = 0.10
SLIPPAGE_PERCENT = 0.0005
STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20
MAX_HOLD_DAYS = 20
RANDOM_STATE = 42

REPORTS_DIR = Path("reports")
JSON_PATH = REPORTS_DIR / "trade_quality_comparison.json"
CSV_PATH = REPORTS_DIR / "trade_quality_comparison_folds.csv"


def baseline(row: pd.Series, probability: float) -> bool:
    return probability >= 0.50


def confidence_55(row: pd.Series, probability: float) -> bool:
    return probability >= 0.55


def confidence_60(row: pd.Series, probability: float) -> bool:
    return probability >= 0.60


def trend(row: pd.Series, probability: float) -> bool:
    return probability >= 0.50 and float(row["price_vs_ema200"]) > 0


def rsi_band(row: pd.Series, probability: float) -> bool:
    rsi = float(row["rsi_14"])
    return probability >= 0.50 and 40 <= rsi <= 70


def volatility_cap(row: pd.Series, probability: float) -> bool:
    return probability >= 0.50 and float(row["atr_pct"]) <= 0.035


def quality_balanced(row: pd.Series, probability: float) -> bool:
    return (
        probability >= 0.55
        and float(row["price_vs_ema200"]) > 0
        and 40 <= float(row["rsi_14"]) <= 72
        and float(row["atr_pct"]) <= 0.040
    )


def quality_selective(row: pd.Series, probability: float) -> bool:
    return (
        probability >= 0.60
        and float(row["price_vs_ema200"]) > 0
        and 45 <= float(row["rsi_14"]) <= 68
        and float(row["atr_pct"]) <= 0.035
    )


POLICIES: dict[str, Callable[[pd.Series, float], bool]] = {
    "baseline": baseline,
    "confidence_55": confidence_55,
    "confidence_60": confidence_60,
    "trend": trend,
    "rsi_band": rsi_band,
    "volatility_cap": volatility_cap,
    "quality_balanced": quality_balanced,
    "quality_selective": quality_selective,
}


def prepare_dataset() -> pd.DataFrame:
    frame = add_features(
        get_market_data(
            symbol=SYMBOL,
            lookback_days=LOOKBACK_DAYS,
        )
    )

    frame["future_close"] = frame["close"].shift(-HOLD_DAYS)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (
        frame["future_close"] > frame["close"]
    ).astype(int)

    required_columns = list(
        dict.fromkeys(
            MODEL_FEATURES
            + [
                "target",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "price_vs_ema200",
                "rsi_14",
                "atr_pct",
            ]
        )
    )

    frame[required_columns] = frame[required_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = (
        frame.dropna(subset=required_columns)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(frame) < MINIMUM_TRAINING_ROWS + 240:
        raise RuntimeError(
            f"Only {len(frame)} usable rows were returned."
        )

    return frame


def build_folds(
    frame: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    remaining = len(frame) - MINIMUM_TRAINING_ROWS
    fold_size = remaining // NUMBER_OF_FOLDS

    result = []

    for fold_number in range(NUMBER_OF_FOLDS):
        test_start = MINIMUM_TRAINING_ROWS + fold_number * fold_size
        test_end = (
            len(frame)
            if fold_number == NUMBER_OF_FOLDS - 1
            else test_start + fold_size
        )

        training = frame.iloc[:test_start].copy()
        testing = frame.iloc[test_start:test_end].copy()

        if (
            training["target"].nunique() >= 2
            and testing["target"].nunique() >= 2
        ):
            result.append((training, testing))

    if len(result) < 4:
        raise RuntimeError("Fewer than four usable folds were created.")

    return result


def build_model() -> LGBMClassifier:
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


def positive_probability(
    model: LGBMClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)
    return np.asarray(
        probabilities[:, positive_index],
        dtype=float,
    )


def simulate(
    testing: pd.DataFrame,
    probability_up: np.ndarray,
    policy: Callable[[pd.Series, float], bool],
) -> dict:
    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = None
    entry_value = None
    entry_index = None

    trade_returns = []
    equity_curve = []
    invested_days = 0
    accepted_signals = 0
    rejected_signals = 0

    for index, ((_, row), probability) in enumerate(
        zip(testing.iterrows(), probability_up)
    ):
        close_price = float(row["close"])
        low_price = float(row["low"])
        high_price = float(row["high"])

        if shares == 0:
            model_is_bullish = probability >= 0.50
            should_enter = policy(row, float(probability))

            if model_is_bullish and should_enter:
                accepted_signals += 1

                allocation = cash * POSITION_PERCENT
                execution_price = close_price * (1 + SLIPPAGE_PERCENT)

                shares = allocation / execution_price
                cash -= allocation

                entry_price = execution_price
                entry_value = allocation
                entry_index = index

            elif model_is_bullish and not should_enter:
                rejected_signals += 1

        else:
            invested_days += 1
            days_held = index - int(entry_index)

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
                proceeds = (
                    shares
                    * exit_price
                    * (1 - SLIPPAGE_PERCENT)
                )

                cash += proceeds

                trade_returns.append(
                    (proceeds - float(entry_value))
                    / float(entry_value)
                )

                shares = 0.0
                entry_price = None
                entry_value = None
                entry_index = None

        equity_curve.append(cash + shares * close_price)

    if shares > 0:
        final_close = float(testing.iloc[-1]["close"])
        proceeds = (
            shares
            * final_close
            * (1 - SLIPPAGE_PERCENT)
        )

        cash += proceeds

        trade_returns.append(
            (proceeds - float(entry_value))
            / float(entry_value)
        )

        if equity_curve:
            equity_curve[-1] = cash

    curve = np.asarray(equity_curve, dtype=float)

    maximum_drawdown = (
        float(
            (
                curve
                / np.maximum.accumulate(curve)
                - 1
            ).min()
        )
        if curve.size
        else 0.0
    )

    wins = [value for value in trade_returns if value > 0]
    losses = [value for value in trade_returns if value < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    return {
        "final_value": float(cash),
        "total_return": float(cash / INITIAL_CAPITAL - 1),
        "maximum_drawdown": maximum_drawdown,
        "trades": len(trade_returns),
        "win_rate": (
            float(np.mean(np.asarray(trade_returns) > 0))
            if trade_returns
            else 0.0
        ),
        "profit_factor": float(profit_factor),
        "exposure": float(
            invested_days / max(1, len(testing))
        ),
        "accepted_signals": accepted_signals,
        "rejected_signals": rejected_signals,
    }


def summarize(rows: list[dict]) -> dict:
    returns = np.asarray(
        [row["total_return"] for row in rows],
        dtype=float,
    )

    drawdowns = np.asarray(
        [row["maximum_drawdown"] for row in rows],
        dtype=float,
    )

    finite_profit_factors = [
        min(float(row["profit_factor"]), 10.0)
        for row in rows
        if math.isfinite(float(row["profit_factor"]))
    ]

    return {
        "folds": len(rows),
        "positive_folds": int((returns > 0).sum()),
        "median_fold_return": float(np.median(returns)),
        "mean_fold_return": float(np.mean(returns)),
        "worst_fold_return": float(np.min(returns)),
        "best_fold_return": float(np.max(returns)),
        "total_trades": int(sum(row["trades"] for row in rows)),
        "total_accepted_signals": int(
            sum(row["accepted_signals"] for row in rows)
        ),
        "total_rejected_signals": int(
            sum(row["rejected_signals"] for row in rows)
        ),
        "mean_win_rate": float(
            np.mean([row["win_rate"] for row in rows])
        ),
        "mean_profit_factor": (
            float(np.mean(finite_profit_factors))
            if finite_profit_factors
            else 0.0
        ),
        "mean_exposure": float(
            np.mean([row["exposure"] for row in rows])
        ),
        "mean_maximum_drawdown": float(np.mean(drawdowns)),
        "worst_maximum_drawdown": float(np.min(drawdowns)),
        "mean_model_accuracy": float(
            np.mean([row["model_accuracy"] for row in rows])
        ),
        "mean_model_roc_auc": float(
            np.mean([row["model_roc_auc"] for row in rows])
        ),
    }


def research_score(summary: dict) -> float:
    trade_score = min(
        1.0,
        summary["total_trades"] / 50.0,
    )

    return float(
        4.0 * summary["median_fold_return"]
        + 2.0 * summary["mean_fold_return"]
        + 0.015 * summary["positive_folds"]
        + 0.010 * min(3.0, summary["mean_profit_factor"])
        + 0.10 * (summary["mean_model_roc_auc"] - 0.50)
        + 0.015 * trade_score
        + 1.5 * summary["worst_maximum_drawdown"]
    )


def run() -> None:
    frame = prepare_dataset()
    fold_rows = []

    for fold_number, (training, testing) in enumerate(
        build_folds(frame),
        start=1,
    ):
        print(
            f"\nFold {fold_number}: "
            f"{testing.iloc[0]['timestamp']} through "
            f"{testing.iloc[-1]['timestamp']}"
        )

        model = build_model()
        model.fit(
            training[MODEL_FEATURES],
            training["target"],
        )

        X_test = testing[MODEL_FEATURES]
        y_test = testing["target"]

        probability_up = positive_probability(
            model,
            X_test,
        )

        prediction = (
            probability_up >= 0.50
        ).astype(int)

        validation = {
            "model_accuracy": float(
                accuracy_score(y_test, prediction)
            ),
            "model_roc_auc": float(
                roc_auc_score(y_test, probability_up)
            ),
        }

        for policy_name, policy in POLICIES.items():
            strategy = simulate(
                testing,
                probability_up,
                policy,
            )

            row = {
                "fold": fold_number,
                "policy": policy_name,
                **validation,
                **strategy,
            }

            fold_rows.append(row)

            print(
                f"  {policy_name}: "
                f"return={strategy['total_return']:.2%}, "
                f"trades={strategy['trades']}, "
                f"rejected={strategy['rejected_signals']}, "
                f"max_dd={strategy['maximum_drawdown']:.2%}"
            )

    summaries = {}

    for policy_name in POLICIES:
        rows = [
            row
            for row in fold_rows
            if row["policy"] == policy_name
        ]

        summary = summarize(rows)
        summary["research_score"] = research_score(summary)
        summaries[policy_name] = summary

    ranking = sorted(
        summaries,
        key=lambda name: summaries[name]["research_score"],
        reverse=True,
    )

    baseline_summary = summaries["baseline"]
    winner_summary = summaries[ranking[0]]

    promotion_rules = {
        "winner_is_filter": ranking[0] != "baseline",
        "median_return_improves": (
            winner_summary["median_fold_return"]
            > baseline_summary["median_fold_return"]
        ),
        "mean_return_not_worse": (
            winner_summary["mean_fold_return"]
            >= baseline_summary["mean_fold_return"]
        ),
        "positive_folds_not_worse": (
            winner_summary["positive_folds"]
            >= baseline_summary["positive_folds"]
        ),
        "worst_drawdown_not_worse": (
            winner_summary["worst_maximum_drawdown"]
            >= baseline_summary["worst_maximum_drawdown"]
        ),
        "trade_count_at_least_80_percent": (
            winner_summary["total_trades"]
            >= 0.80 * baseline_summary["total_trades"]
        ),
    }

    promotion_passed = all(promotion_rules.values())

    report = {
        "symbol": SYMBOL,
        "features": MODEL_FEATURES,
        "fold_results": fold_rows,
        "summaries": summaries,
        "recommendation": {
            "research_winner": ranking[0],
            "ranking": ranking,
            "promotion_rules": promotion_rules,
            "promotion_passed": promotion_passed,
            "decision": (
                "Promote the winning filter for further paper research."
                if promotion_passed
                else "Keep the unfiltered LightGBM entry policy."
            ),
            "note": (
                "Research only. This workflow cannot place orders."
            ),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    JSON_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(fold_rows).to_csv(
        CSV_PATH,
        index=False,
    )

    print("\n=== TRADE QUALITY RESULT ===")
    print(
        json.dumps(
            report["recommendation"],
            indent=2,
        )
    )
    print(f"\nJSON report: {JSON_PATH}")
    print(f"CSV report: {CSV_PATH}")


if __name__ == "__main__":
    run()
