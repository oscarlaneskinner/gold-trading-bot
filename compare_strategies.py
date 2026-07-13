"""
Side-by-side walk-forward comparison for the GLD trading bot.

This script is research-only. It does not connect to the trading client
and cannot place orders.

It compares:

A. baseline
   - 20-day prediction horizon
   - 50% confidence threshold
   - no entry filters

B. ema200_challenger
   - 20-day prediction horizon
   - 55% confidence threshold
   - price must be above EMA 200

C. inverted_diagnostic
   - diagnostic only
   - buys when the model predicts DOWN with at least 55% confidence
   - never intended for live deployment without strong evidence

The test uses:
- expanding-window training
- non-overlapping test folds
- 10% position sizing
- slippage
- stop-loss, take-profit, and maximum holding period
- corrected forward targets
- drawdown, win rate, profit factor, exposure, and fold statistics

Run:
    python compare_strategies.py
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from data import get_market_data
from features import MODEL_FEATURES, add_features


# ============================================================
# RESEARCH SETTINGS
# ============================================================

SYMBOL = "GLD"
LOOKBACK_DAYS = 5000

PREDICTION_HORIZON = 20
NUMBER_OF_TEST_FOLDS = 6
MINIMUM_TRAINING_ROWS = 750

INITIAL_CAPITAL = 10_000.00
POSITION_PERCENT = 0.10
SLIPPAGE_PERCENT = 0.0005

STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20
MAX_HOLD_DAYS = 20

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 20

REPORTS_DIR = Path("reports")
JSON_REPORT_PATH = REPORTS_DIR / "strategy_comparison.json"
CSV_REPORT_PATH = REPORTS_DIR / "strategy_comparison_folds.csv"


@dataclass(frozen=True)
class StrategyVariant:
    name: str
    confidence_threshold: float
    require_above_ema200: bool
    invert_prediction: bool


VARIANTS = [
    StrategyVariant(
        name="baseline",
        confidence_threshold=0.50,
        require_above_ema200=False,
        invert_prediction=False,
    ),
    StrategyVariant(
        name="ema200_challenger",
        confidence_threshold=0.55,
        require_above_ema200=True,
        invert_prediction=False,
    ),
    StrategyVariant(
        name="inverted_diagnostic",
        confidence_threshold=0.55,
        require_above_ema200=False,
        invert_prediction=True,
    ),
]


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset() -> pd.DataFrame:
    """Download data, build features, and create a corrected forward target."""

    raw = get_market_data(
        symbol=SYMBOL,
        lookback_days=LOOKBACK_DAYS,
    )

    frame = add_features(raw)

    frame["future_close"] = frame["close"].shift(
        -PREDICTION_HORIZON
    )

    # Remove rows that do not yet have a known future close.
    frame = frame[
        frame["future_close"].notna()
    ].copy()

    frame["target"] = (
        frame["future_close"] > frame["close"]
    ).astype(int)

    frame[MODEL_FEATURES] = frame[
        MODEL_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = (
        frame.dropna(
            subset=MODEL_FEATURES + [
                "target",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(frame) < (
        MINIMUM_TRAINING_ROWS
        + NUMBER_OF_TEST_FOLDS * 40
    ):
        raise RuntimeError(
            f"Only {len(frame)} usable rows were returned. "
            "More history is required for a reliable comparison."
        )

    return frame


def build_non_overlapping_folds(
    frame: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding training windows and non-overlapping test windows.
    """

    remaining_rows = len(frame) - MINIMUM_TRAINING_ROWS

    fold_size = remaining_rows // NUMBER_OF_TEST_FOLDS

    if fold_size < 40:
        raise RuntimeError(
            "The calculated test folds are too small."
        )

    folds: list[
        tuple[pd.DataFrame, pd.DataFrame]
    ] = []

    for fold_number in range(
        NUMBER_OF_TEST_FOLDS
    ):
        test_start = (
            MINIMUM_TRAINING_ROWS
            + fold_number * fold_size
        )

        if fold_number == (
            NUMBER_OF_TEST_FOLDS - 1
        ):
            test_end = len(frame)
        else:
            test_end = test_start + fold_size

        training = frame.iloc[
            :test_start
        ].copy()

        testing = frame.iloc[
            test_start:test_end
        ].copy()

        if (
            training["target"].nunique() < 2
            or testing["target"].nunique() < 2
        ):
            continue

        folds.append(
            (
                training,
                testing,
            )
        )

    if len(folds) < 4:
        raise RuntimeError(
            "Fewer than four usable folds were created."
        )

    return folds


# ============================================================
# MODEL
# ============================================================

def train_model(
    training: pd.DataFrame,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        training[MODEL_FEATURES],
        training["target"],
    )

    return model


def model_validation_metrics(
    model: RandomForestClassifier,
    testing: pd.DataFrame,
) -> dict[str, float]:
    prediction = model.predict(
        testing[MODEL_FEATURES]
    )

    positive_index = list(
        model.classes_
    ).index(1)

    probability_up = model.predict_proba(
        testing[MODEL_FEATURES]
    )[:, positive_index]

    metrics = {
        "accuracy": float(
            accuracy_score(
                testing["target"],
                prediction,
            )
        ),
        "roc_auc": math.nan,
    }

    if testing["target"].nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                testing["target"],
                probability_up,
            )
        )

    return metrics


# ============================================================
# STRATEGY SIMULATION
# ============================================================

def entry_signal(
    row: pd.Series,
    prediction: int,
    probability_up: float,
    variant: StrategyVariant,
) -> bool:
    if variant.invert_prediction:
        signal_is_correct_direction = (
            prediction == 0
        )

        confidence = 1.0 - probability_up
    else:
        signal_is_correct_direction = (
            prediction == 1
        )

        confidence = probability_up

    if not signal_is_correct_direction:
        return False

    if confidence < variant.confidence_threshold:
        return False

    if (
        variant.require_above_ema200
        and not (
            float(row["close"])
            > float(row["ema_200"])
        )
    ):
        return False

    return True


def simulate_fold(
    model: RandomForestClassifier,
    testing: pd.DataFrame,
    variant: StrategyVariant,
) -> dict[str, Any]:
    """
    Simulate one test fold with one position at a time.
    """

    cash = INITIAL_CAPITAL
    shares = 0.0

    entry_price: float | None = None
    entry_value: float | None = None
    entry_index: int | None = None
    peak_price: float | None = None

    trade_returns: list[float] = []
    equity_curve: list[float] = []
    invested_days = 0

    positive_index = list(
        model.classes_
    ).index(1)

    features = testing[
        MODEL_FEATURES
    ]

    predictions = model.predict(
        features
    )

    probabilities_up = model.predict_proba(
        features
    )[:, positive_index]

    for local_index, (
        (_, row),
        prediction,
        probability_up,
    ) in enumerate(
        zip(
            testing.iterrows(),
            predictions,
            probabilities_up,
        )
    ):
        close_price = float(
            row["close"]
        )

        low_price = float(
            row["low"]
        )

        high_price = float(
            row["high"]
        )

        if shares == 0.0:
            if entry_signal(
                row=row,
                prediction=int(prediction),
                probability_up=float(
                    probability_up
                ),
                variant=variant,
            ):
                allocation = (
                    cash * POSITION_PERCENT
                )

                execution_price = (
                    close_price
                    * (
                        1
                        + SLIPPAGE_PERCENT
                    )
                )

                shares = (
                    allocation
                    / execution_price
                )

                cash -= allocation

                entry_price = execution_price
                entry_value = allocation
                entry_index = local_index
                peak_price = high_price

        else:
            invested_days += 1

            peak_price = max(
                float(peak_price),
                high_price,
            )

            days_held = (
                local_index
                - int(entry_index)
            )

            stop_level = (
                float(entry_price)
                * (
                    1
                    - STOP_LOSS_PERCENT
                )
            )

            target_level = (
                float(entry_price)
                * (
                    1
                    + TAKE_PROFIT_PERCENT
                )
            )

            exit_price: float | None = None

            if low_price <= stop_level:
                exit_price = stop_level

            elif high_price >= target_level:
                exit_price = target_level

            elif days_held >= MAX_HOLD_DAYS:
                exit_price = close_price

            if exit_price is not None:
                execution_price = (
                    exit_price
                    * (
                        1
                        - SLIPPAGE_PERCENT
                    )
                )

                proceeds = (
                    shares
                    * execution_price
                )

                cash += proceeds

                trade_return = (
                    proceeds
                    - float(entry_value)
                ) / float(entry_value)

                trade_returns.append(
                    trade_return
                )

                shares = 0.0
                entry_price = None
                entry_value = None
                entry_index = None
                peak_price = None

        marked_equity = (
            cash
            + shares * close_price
        )

        equity_curve.append(
            marked_equity
        )

    # Close an unfinished position at the final fold close.
    if shares > 0:
        final_close = float(
            testing.iloc[-1]["close"]
        )

        execution_price = (
            final_close
            * (
                1
                - SLIPPAGE_PERCENT
            )
        )

        proceeds = (
            shares
            * execution_price
        )

        cash += proceeds

        trade_returns.append(
            (
                proceeds
                - float(entry_value)
            ) / float(entry_value)
        )

        shares = 0.0

        if equity_curve:
            equity_curve[-1] = cash

    curve = np.asarray(
        equity_curve,
        dtype=float,
    )

    if curve.size:
        running_peak = np.maximum.accumulate(
            curve
        )

        drawdowns = (
            curve / running_peak
        ) - 1.0

        maximum_drawdown = float(
            drawdowns.min()
        )
    else:
        maximum_drawdown = 0.0

    wins = [
        value
        for value in trade_returns
        if value > 0
    ]

    losses = [
        value
        for value in trade_returns
        if value < 0
    ]

    gross_profit = sum(wins)

    gross_loss = abs(
        sum(losses)
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    final_value = float(cash)

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1.0
    )

    return {
        "final_value": final_value,
        "total_return": float(
            total_return
        ),
        "maximum_drawdown": (
            maximum_drawdown
        ),
        "trades": len(
            trade_returns
        ),
        "win_rate": (
            float(
                np.mean(
                    np.asarray(
                        trade_returns
                    ) > 0
                )
            )
            if trade_returns
            else 0.0
        ),
        "average_trade_return": (
            float(
                np.mean(
                    trade_returns
                )
            )
            if trade_returns
            else 0.0
        ),
        "profit_factor": float(
            profit_factor
        ),
        "exposure": float(
            invested_days
            / max(
                1,
                len(testing),
            )
        ),
    }


def buy_and_hold_fold(
    testing: pd.DataFrame,
) -> float:
    first_price = float(
        testing.iloc[0]["close"]
    )

    final_price = float(
        testing.iloc[-1]["close"]
    )

    return (
        final_price
        / first_price
        - 1.0
    )


# ============================================================
# SUMMARY
# ============================================================

def safe_median(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.median(
            values
        )
    )


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(
            values
        )
    )


def summarize_variant(
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    returns = [
        float(row["total_return"])
        for row in fold_rows
    ]

    drawdowns = [
        float(row["maximum_drawdown"])
        for row in fold_rows
    ]

    trade_counts = [
        int(row["trades"])
        for row in fold_rows
    ]

    profit_factors = [
        float(row["profit_factor"])
        for row in fold_rows
        if math.isfinite(
            float(
                row["profit_factor"]
            )
        )
    ]

    return {
        "folds": len(
            fold_rows
        ),
        "positive_folds": sum(
            value > 0
            for value in returns
        ),
        "median_fold_return": safe_median(
            returns
        ),
        "mean_fold_return": safe_mean(
            returns
        ),
        "worst_fold_return": (
            min(returns)
            if returns
            else 0.0
        ),
        "best_fold_return": (
            max(returns)
            if returns
            else 0.0
        ),
        "mean_maximum_drawdown": safe_mean(
            drawdowns
        ),
        "worst_maximum_drawdown": (
            min(drawdowns)
            if drawdowns
            else 0.0
        ),
        "total_trades": sum(
            trade_counts
        ),
        "mean_profit_factor": safe_mean(
            profit_factors
        ),
        "mean_win_rate": safe_mean(
            [
                float(row["win_rate"])
                for row in fold_rows
            ]
        ),
        "mean_exposure": safe_mean(
            [
                float(row["exposure"])
                for row in fold_rows
            ]
        ),
    }


def choose_recommendation(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, str]:
    baseline = summaries[
        "baseline"
    ]

    challenger = summaries[
        "ema200_challenger"
    ]

    minimum_total_trades = 20

    challenger_passes = all(
        [
            challenger["total_trades"]
            >= minimum_total_trades,

            challenger[
                "positive_folds"
            ] >= 4,

            challenger[
                "median_fold_return"
            ] > baseline[
                "median_fold_return"
            ],

            challenger[
                "worst_maximum_drawdown"
            ] >= baseline[
                "worst_maximum_drawdown"
            ] - 0.01,
        ]
    )

    if challenger_passes:
        return {
            "recommended_variant":
                "ema200_challenger",

            "reason":
                "The challenger passed the predefined "
                "trade-count, fold-consistency, return, "
                "and drawdown checks.",
        }

    return {
        "recommended_variant":
            "baseline",

        "reason":
            "The EMA-200 challenger did not pass every "
            "predefined acceptance rule. Keep the "
            "baseline for paper-trading research.",
    }


# ============================================================
# MAIN
# ============================================================

def run_comparison() -> dict[str, Any]:
    print(
        "Preparing historical data..."
    )

    frame = prepare_dataset()

    folds = build_non_overlapping_folds(
        frame
    )

    print(
        f"Usable rows: {len(frame)}"
    )

    print(
        f"Test folds: {len(folds)}"
    )

    fold_rows: list[
        dict[str, Any]
    ] = []

    validation_rows: list[
        dict[str, Any]
    ] = []

    for fold_index, (
        training,
        testing,
    ) in enumerate(
        folds,
        start=1,
    ):
        print(
            f"\nFold {fold_index}: "
            f"train through "
            f"{training.iloc[-1]['timestamp']}; "
            f"test "
            f"{testing.iloc[0]['timestamp']} "
            f"through "
            f"{testing.iloc[-1]['timestamp']}"
        )

        model = train_model(
            training
        )

        validation = (
            model_validation_metrics(
                model,
                testing,
            )
        )

        validation_rows.append(
            {
                "fold": fold_index,
                **validation,
            }
        )

        benchmark_return = (
            buy_and_hold_fold(
                testing
            )
        )

        for variant in VARIANTS:
            result = simulate_fold(
                model=model,
                testing=testing,
                variant=variant,
            )

            row = {
                "fold": fold_index,
                "variant": variant.name,
                "test_start": str(
                    testing.iloc[0][
                        "timestamp"
                    ]
                ),
                "test_end": str(
                    testing.iloc[-1][
                        "timestamp"
                    ]
                ),
                "training_rows": len(
                    training
                ),
                "testing_rows": len(
                    testing
                ),
                "model_accuracy": (
                    validation["accuracy"]
                ),
                "model_roc_auc": (
                    validation["roc_auc"]
                ),
                "buy_and_hold_return": (
                    benchmark_return
                ),
                **result,
            }

            fold_rows.append(
                row
            )

            print(
                f"  {variant.name}: "
                f"return={result['total_return']:.2%}, "
                f"trades={result['trades']}, "
                f"max_dd={result['maximum_drawdown']:.2%}"
            )

    summaries: dict[
        str,
        dict[str, Any]
    ] = {}

    for variant in VARIANTS:
        variant_rows = [
            row
            for row in fold_rows
            if row["variant"]
            == variant.name
        ]

        summaries[
            variant.name
        ] = summarize_variant(
            variant_rows
        )

    recommendation = (
        choose_recommendation(
            summaries
        )
    )

    report = {
        "symbol": SYMBOL,
        "prediction_horizon_days":
            PREDICTION_HORIZON,
        "position_percent":
            POSITION_PERCENT,
        "slippage_percent":
            SLIPPAGE_PERCENT,
        "stop_loss_percent":
            STOP_LOSS_PERCENT,
        "take_profit_percent":
            TAKE_PROFIT_PERCENT,
        "maximum_hold_days":
            MAX_HOLD_DAYS,
        "model_features":
            MODEL_FEATURES,
        "model_parameters": {
            "n_estimators":
                N_ESTIMATORS,
            "max_depth":
                MAX_DEPTH,
            "min_samples_leaf":
                MIN_SAMPLES_LEAF,
            "random_state":
                RANDOM_STATE,
        },
        "variants": [
            asdict(
                variant
            )
            for variant in VARIANTS
        ],
        "validation_folds":
            validation_rows,
        "fold_results":
            fold_rows,
        "summaries":
            summaries,
        "recommendation":
            recommendation,
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_REPORT_PATH.write_text(
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
        CSV_REPORT_PATH,
        index=False,
    )

    print(
        "\n=== STRATEGY SUMMARIES ==="
    )

    for name, summary in (
        summaries.items()
    ):
        print(
            f"\n{name}"
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

    print(
        "\n=== RECOMMENDATION ==="
    )

    print(
        json.dumps(
            recommendation,
            indent=2,
        )
    )

    print(
        f"\nJSON report: "
        f"{JSON_REPORT_PATH}"
    )

    print(
        f"CSV report: "
        f"{CSV_REPORT_PATH}"
    )

    return report


if __name__ == "__main__":
    run_comparison()
