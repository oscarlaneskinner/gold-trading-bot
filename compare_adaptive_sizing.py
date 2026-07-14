"""
Research-only GLD adaptive position-sizing comparison.

Keeps the tuned LightGBM model and the existing 13 production features.
Compares fixed and adaptive position-sizing policies:

- fixed_10
- fixed_15
- fixed_20
- confidence_5_15
- volatility_5_15
- confidence_volatility_5_15
- confidence_volatility_drawdown_5_15

The model decides whether to enter. The sizing policy decides how much
capital to allocate.

Uses six walk-forward folds and the same stop-loss, take-profit, hold period,
and slippage assumptions used in prior research.

This script cannot place orders.

Outputs:
- reports/adaptive_sizing_comparison.json
- reports/adaptive_sizing_comparison_folds.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import HOLD_DAYS, SYMBOL
from data import get_market_data
from features import MODEL_FEATURES, add_features


LOOKBACK_DAYS = 5000
NUMBER_OF_FOLDS = 6
MINIMUM_TRAINING_ROWS = 750

INITIAL_CAPITAL = 10_000.0
ENTRY_THRESHOLD = 0.50
SLIPPAGE_PERCENT = 0.0005
STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20
MAX_HOLD_DAYS = 20
RANDOM_STATE = 42

MIN_POSITION_PERCENT = 0.05
NORMAL_POSITION_PERCENT = 0.10
MAX_POSITION_PERCENT = 0.15

REPORTS_DIR = Path("reports")
JSON_PATH = REPORTS_DIR / "adaptive_sizing_comparison.json"
CSV_PATH = REPORTS_DIR / "adaptive_sizing_comparison_folds.csv"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def fixed_10(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    return 0.10


def fixed_15(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    return 0.15


def fixed_20(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    return 0.20


def confidence_5_15(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    scaled = (
        MIN_POSITION_PERCENT
        + (probability_up - 0.50) / 0.50
        * (MAX_POSITION_PERCENT - MIN_POSITION_PERCENT)
    )

    return clamp(
        scaled,
        MIN_POSITION_PERCENT,
        MAX_POSITION_PERCENT,
    )


def volatility_5_15(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    atr_percent = float(row["atr_pct"])

    if atr_percent <= 0.015:
        return 0.15

    if atr_percent >= 0.040:
        return 0.05

    scaled = (
        0.15
        - (atr_percent - 0.015)
        / (0.040 - 0.015)
        * 0.10
    )

    return clamp(scaled, 0.05, 0.15)


def confidence_volatility_5_15(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    confidence_size = confidence_5_15(
        row,
        probability_up,
        current_drawdown,
    )

    volatility_size = volatility_5_15(
        row,
        probability_up,
        current_drawdown,
    )

    combined = (
        0.60 * confidence_size
        + 0.40 * volatility_size
    )

    return clamp(combined, 0.05, 0.15)


def confidence_volatility_drawdown_5_15(
    row: pd.Series,
    probability_up: float,
    current_drawdown: float,
) -> float:
    base_size = confidence_volatility_5_15(
        row,
        probability_up,
        current_drawdown,
    )

    if current_drawdown <= -0.05:
        multiplier = 0.60
    elif current_drawdown <= -0.03:
        multiplier = 0.75
    elif current_drawdown <= -0.015:
        multiplier = 0.90
    else:
        multiplier = 1.00

    return clamp(
        base_size * multiplier,
        MIN_POSITION_PERCENT,
        MAX_POSITION_PERCENT,
    )


SIZING_POLICIES: dict[
    str,
    Callable[[pd.Series, float, float], float],
] = {
    "fixed_10": fixed_10,
    "fixed_15": fixed_15,
    "fixed_20": fixed_20,
    "confidence_5_15": confidence_5_15,
    "volatility_5_15": volatility_5_15,
    "confidence_volatility_5_15": confidence_volatility_5_15,
    "confidence_volatility_drawdown_5_15":
        confidence_volatility_drawdown_5_15,
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

    folds = []

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
            folds.append((training, testing))

    if len(folds) < 4:
        raise RuntimeError(
            "Fewer than four usable folds were created."
        )

    return folds


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
    sizing_policy: Callable[
        [pd.Series, float, float],
        float,
    ],
) -> dict:
    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = None
    entry_value = None
    entry_index = None

    trade_returns = []
    trade_position_sizes = []
    equity_curve = []

    for index, ((_, row), probability) in enumerate(
        zip(testing.iterrows(), probability_up)
    ):
        close_price = float(row["close"])
        low_price = float(row["low"])
        high_price = float(row["high"])

        current_equity = cash + shares * close_price

        if equity_curve:
            running_peak = max(equity_curve)
            current_drawdown = (
                current_equity / running_peak - 1
                if running_peak > 0
                else 0.0
            )
        else:
            current_drawdown = 0.0

        if shares == 0:
            if float(probability) >= ENTRY_THRESHOLD:
                position_percent = sizing_policy(
                    row,
                    float(probability),
                    float(current_drawdown),
                )

                position_percent = clamp(
                    position_percent,
                    0.01,
                    0.25,
                )

                allocation = cash * position_percent
                execution_price = close_price * (1 + SLIPPAGE_PERCENT)

                shares = allocation / execution_price
                cash -= allocation

                entry_price = execution_price
                entry_value = allocation
                entry_index = index

                trade_position_sizes.append(
                    position_percent
                )

        else:
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

        equity_curve.append(
            cash + shares * close_price
        )

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
        "total_return": float(
            cash / INITIAL_CAPITAL - 1
        ),
        "maximum_drawdown": maximum_drawdown,
        "trades": len(trade_returns),
        "win_rate": (
            float(
                np.mean(
                    np.asarray(trade_returns) > 0
                )
            )
            if trade_returns
            else 0.0
        ),
        "profit_factor": float(profit_factor),
        "mean_position_percent": (
            float(np.mean(trade_position_sizes))
            if trade_position_sizes
            else 0.0
        ),
        "minimum_position_percent": (
            float(np.min(trade_position_sizes))
            if trade_position_sizes
            else 0.0
        ),
        "maximum_position_percent": (
            float(np.max(trade_position_sizes))
            if trade_position_sizes
            else 0.0
        ),
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
        if math.isfinite(
            float(row["profit_factor"])
        )
    ]

    return {
        "folds": len(rows),
        "positive_folds": int(
            (returns > 0).sum()
        ),
        "median_fold_return": float(
            np.median(returns)
        ),
        "mean_fold_return": float(
            np.mean(returns)
        ),
        "worst_fold_return": float(
            np.min(returns)
        ),
        "best_fold_return": float(
            np.max(returns)
        ),
        "total_trades": int(
            sum(row["trades"] for row in rows)
        ),
        "mean_win_rate": float(
            np.mean(
                [row["win_rate"] for row in rows]
            )
        ),
        "mean_profit_factor": (
            float(np.mean(finite_profit_factors))
            if finite_profit_factors
            else 0.0
        ),
        "mean_position_percent": float(
            np.mean(
                [
                    row["mean_position_percent"]
                    for row in rows
                ]
            )
        ),
        "mean_maximum_drawdown": float(
            np.mean(drawdowns)
        ),
        "worst_maximum_drawdown": float(
            np.min(drawdowns)
        ),
    }


def risk_adjusted_score(summary: dict) -> float:
    drawdown_penalty = abs(
        summary["worst_maximum_drawdown"]
    )

    return float(
        4.0 * summary["median_fold_return"]
        + 2.0 * summary["mean_fold_return"]
        + 0.015 * summary["positive_folds"]
        + 0.010 * min(
            3.0,
            summary["mean_profit_factor"],
        )
        - 2.5 * drawdown_penalty
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

        probability_up = positive_probability(
            model,
            testing[MODEL_FEATURES],
        )

        for policy_name, policy in SIZING_POLICIES.items():
            result = simulate(
                testing,
                probability_up,
                policy,
            )

            fold_rows.append(
                {
                    "fold": fold_number,
                    "policy": policy_name,
                    **result,
                }
            )

            print(
                f"  {policy_name}: "
                f"return={result['total_return']:.2%}, "
                f"trades={result['trades']}, "
                f"avg_size="
                f"{result['mean_position_percent']:.2%}, "
                f"max_dd="
                f"{result['maximum_drawdown']:.2%}"
            )

    summaries = {}

    for policy_name in SIZING_POLICIES:
        rows = [
            row
            for row in fold_rows
            if row["policy"] == policy_name
        ]

        summary = summarize(rows)
        summary["risk_adjusted_score"] = (
            risk_adjusted_score(summary)
        )

        summaries[policy_name] = summary

    ranking = sorted(
        summaries,
        key=lambda name:
            summaries[name]["risk_adjusted_score"],
        reverse=True,
    )

    baseline = summaries["fixed_10"]
    winner = summaries[ranking[0]]

    promotion_rules = {
        "winner_is_not_baseline":
            ranking[0] != "fixed_10",
        "mean_return_improves":
            winner["mean_fold_return"]
            > baseline["mean_fold_return"],
        "median_return_not_worse":
            winner["median_fold_return"]
            >= baseline["median_fold_return"],
        "positive_folds_not_worse":
            winner["positive_folds"]
            >= baseline["positive_folds"],
        "worst_drawdown_increase_within_50_percent":
            abs(winner["worst_maximum_drawdown"])
            <= 1.50
            * abs(baseline["worst_maximum_drawdown"]),
        "profit_factor_not_worse":
            winner["mean_profit_factor"]
            >= baseline["mean_profit_factor"],
    }

    promotion_passed = all(
        promotion_rules.values()
    )

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
                "Promote the winning sizing policy "
                "for further paper research."
                if promotion_passed
                else "Keep fixed 10% production sizing."
            ),
            "note": (
                "Research only. This workflow "
                "cannot place orders."
            ),
        },
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        CSV_PATH,
        index=False,
    )

    print(
        "\n=== ADAPTIVE SIZING RESULT ==="
    )

    print(
        json.dumps(
            report["recommendation"],
            indent=2,
        )
    )

    print(
        f"\nJSON report: {JSON_PATH}"
    )

    print(
        f"CSV report: {CSV_PATH}"
    )


if __name__ == "__main__":
    run()
