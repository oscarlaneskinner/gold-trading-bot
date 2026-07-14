"""
Research-only focused sizing study for GLD.

Compares:
- fixed 10%
- fixed 12.5%
- fixed 15%

Adds stronger risk analysis:
- continuous-period return
- continuous maximum drawdown
- return-to-drawdown ratio
- downside deviation
- longest losing streak
- average losing trade
- worst losing trade
- doubled-slippage stress test

Uses the tuned LightGBM model and the existing 13 production features.
This script cannot place orders.

Outputs:
- reports/focused_sizing_study.json
- reports/focused_sizing_study_folds.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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
BASE_SLIPPAGE = 0.0005
STRESS_SLIPPAGE = 0.0010
STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20
MAX_HOLD_DAYS = 20
RANDOM_STATE = 42

SIZING_POLICIES = {
    "fixed_10": 0.10,
    "fixed_12_5": 0.125,
    "fixed_15": 0.15,
}

REPORTS_DIR = Path("reports")
JSON_PATH = REPORTS_DIR / "focused_sizing_study.json"
CSV_PATH = REPORTS_DIR / "focused_sizing_study_folds.csv"


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

    required = list(
        dict.fromkeys(
            MODEL_FEATURES
            + [
                "target",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        )
    )

    frame[required] = frame[required].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = (
        frame.dropna(subset=required)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(frame) < MINIMUM_TRAINING_ROWS + 240:
        raise RuntimeError(
            f"Only {len(frame)} usable rows were returned."
        )

    return frame


def build_folds(frame: pd.DataFrame):
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

    return result


def build_model():
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


def probability_up(model, X):
    classes = list(model.classes_)
    return np.asarray(
        model.predict_proba(X)[:, classes.index(1)],
        dtype=float,
    )


def longest_losing_streak(trade_returns):
    longest = 0
    current = 0

    for value in trade_returns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def downside_deviation(equity_curve):
    if len(equity_curve) < 2:
        return 0.0

    returns = pd.Series(
        equity_curve,
        dtype=float,
    ).pct_change().dropna()

    downside = returns[returns < 0]

    if downside.empty:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(downside)
            )
        )
    )


def simulate(
    testing: pd.DataFrame,
    probabilities: np.ndarray,
    position_percent: float,
    slippage_percent: float,
):
    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = None
    entry_value = None
    entry_index = None

    trade_returns = []
    equity_curve = []

    for index, ((_, row), probability) in enumerate(
        zip(testing.iterrows(), probabilities)
    ):
        close_price = float(row["close"])
        low_price = float(row["low"])
        high_price = float(row["high"])

        if shares == 0:
            if probability >= ENTRY_THRESHOLD:
                allocation = cash * position_percent
                execution_price = (
                    close_price
                    * (1 + slippage_percent)
                )

                shares = allocation / execution_price
                cash -= allocation

                entry_price = execution_price
                entry_value = allocation
                entry_index = index

        else:
            days_held = index - int(entry_index)

            stop_level = (
                float(entry_price)
                * (1 - STOP_LOSS_PERCENT)
            )

            target_level = (
                float(entry_price)
                * (1 + TAKE_PROFIT_PERCENT)
            )

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
                    * (1 - slippage_percent)
                )

                cash += proceeds

                trade_returns.append(
                    (
                        proceeds
                        - float(entry_value)
                    )
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
        final_close = float(
            testing.iloc[-1]["close"]
        )

        proceeds = (
            shares
            * final_close
            * (1 - slippage_percent)
        )

        cash += proceeds

        trade_returns.append(
            (
                proceeds
                - float(entry_value)
            )
            / float(entry_value)
        )

        if equity_curve:
            equity_curve[-1] = cash

    curve = np.asarray(
        equity_curve,
        dtype=float,
    )

    max_drawdown = (
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

    losses = [
        value
        for value in trade_returns
        if value < 0
    ]

    wins = [
        value
        for value in trade_returns
        if value > 0
    ]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    total_return = float(
        cash / INITIAL_CAPITAL - 1
    )

    return {
        "final_value": float(cash),
        "total_return": total_return,
        "maximum_drawdown": max_drawdown,
        "return_to_drawdown": (
            total_return / abs(max_drawdown)
            if max_drawdown < 0
            else 0.0
        ),
        "downside_deviation": downside_deviation(
            equity_curve
        ),
        "trades": len(trade_returns),
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
        "profit_factor": float(
            profit_factor
        ),
        "longest_losing_streak":
            longest_losing_streak(
                trade_returns
            ),
        "average_losing_trade": (
            float(np.mean(losses))
            if losses
            else 0.0
        ),
        "worst_losing_trade": (
            float(np.min(losses))
            if losses
            else 0.0
        ),
    }


def summarize(rows):
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
        "mean_maximum_drawdown": float(
            np.mean(drawdowns)
        ),
        "worst_maximum_drawdown": float(
            np.min(drawdowns)
        ),
        "mean_return_to_drawdown": float(
            np.mean(
                [
                    row["return_to_drawdown"]
                    for row in rows
                ]
            )
        ),
        "mean_downside_deviation": float(
            np.mean(
                [
                    row["downside_deviation"]
                    for row in rows
                ]
            )
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
        "worst_longest_losing_streak": int(
            max(
                row["longest_losing_streak"]
                for row in rows
            )
        ),
        "mean_average_losing_trade": float(
            np.mean(
                [
                    row["average_losing_trade"]
                    for row in rows
                ]
            )
        ),
        "worst_losing_trade": float(
            min(
                row["worst_losing_trade"]
                for row in rows
            )
        ),
    }


def continuous_simulation(
    frame: pd.DataFrame,
    position_percent: float,
    slippage_percent: float,
):
    folds = build_folds(frame)
    all_test_rows = []
    all_probabilities = []

    for training, testing in folds:
        model = build_model()

        model.fit(
            training[MODEL_FEATURES],
            training["target"],
        )

        probabilities = probability_up(
            model,
            testing[MODEL_FEATURES],
        )

        all_test_rows.append(testing)
        all_probabilities.append(probabilities)

    continuous_frame = pd.concat(
        all_test_rows,
        ignore_index=True,
    )

    continuous_probabilities = np.concatenate(
        all_probabilities
    )

    return simulate(
        continuous_frame,
        continuous_probabilities,
        position_percent,
        slippage_percent,
    )


def run():
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

        probabilities = probability_up(
            model,
            testing[MODEL_FEATURES],
        )

        for policy_name, position_percent in (
            SIZING_POLICIES.items()
        ):
            for scenario_name, slippage in {
                "base": BASE_SLIPPAGE,
                "double_slippage": STRESS_SLIPPAGE,
            }.items():
                result = simulate(
                    testing,
                    probabilities,
                    position_percent,
                    slippage,
                )

                fold_rows.append(
                    {
                        "fold": fold_number,
                        "policy": policy_name,
                        "position_percent":
                            position_percent,
                        "scenario":
                            scenario_name,
                        **result,
                    }
                )

                print(
                    f"  {policy_name} "
                    f"[{scenario_name}]: "
                    f"return="
                    f"{result['total_return']:.2%}, "
                    f"max_dd="
                    f"{result['maximum_drawdown']:.2%}, "
                    f"r/dd="
                    f"{result['return_to_drawdown']:.2f}"
                )

    summaries = {}

    for policy_name, position_percent in (
        SIZING_POLICIES.items()
    ):
        base_rows = [
            row
            for row in fold_rows
            if row["policy"] == policy_name
            and row["scenario"] == "base"
        ]

        stress_rows = [
            row
            for row in fold_rows
            if row["policy"] == policy_name
            and row["scenario"]
            == "double_slippage"
        ]

        base_summary = summarize(base_rows)
        stress_summary = summarize(stress_rows)

        continuous_base = continuous_simulation(
            frame,
            position_percent,
            BASE_SLIPPAGE,
        )

        continuous_stress = continuous_simulation(
            frame,
            position_percent,
            STRESS_SLIPPAGE,
        )

        summaries[policy_name] = {
            "position_percent":
                position_percent,
            "base":
                base_summary,
            "double_slippage":
                stress_summary,
            "continuous_base":
                continuous_base,
            "continuous_double_slippage":
                continuous_stress,
        }

    baseline = summaries["fixed_10"]

    decisions = {}

    for candidate in (
        "fixed_12_5",
        "fixed_15",
    ):
        candidate_summary = summaries[
            candidate
        ]

        rules = {
            "continuous_return_improves":
                candidate_summary[
                    "continuous_base"
                ]["total_return"]
                > baseline[
                    "continuous_base"
                ]["total_return"],
            "continuous_return_to_drawdown_not_worse":
                candidate_summary[
                    "continuous_base"
                ]["return_to_drawdown"]
                >= 0.95
                * baseline[
                    "continuous_base"
                ]["return_to_drawdown"],
            "continuous_drawdown_within_limit":
                abs(
                    candidate_summary[
                        "continuous_base"
                    ]["maximum_drawdown"]
                )
                <= 1.50
                * abs(
                    baseline[
                        "continuous_base"
                    ]["maximum_drawdown"]
                ),
            "stress_return_positive":
                candidate_summary[
                    "continuous_double_slippage"
                ]["total_return"] > 0,
            "stress_return_to_drawdown_not_worse":
                candidate_summary[
                    "continuous_double_slippage"
                ]["return_to_drawdown"]
                >= 0.90
                * baseline[
                    "continuous_double_slippage"
                ]["return_to_drawdown"],
            "worst_losing_streak_not_worse":
                candidate_summary[
                    "base"
                ]["worst_longest_losing_streak"]
                <= baseline[
                    "base"
                ]["worst_longest_losing_streak"],
            "profit_factor_not_worse":
                candidate_summary[
                    "base"
                ]["mean_profit_factor"]
                >= 0.95
                * baseline[
                    "base"
                ]["mean_profit_factor"],
        }

        decisions[candidate] = {
            "rules": rules,
            "passed": all(rules.values()),
        }

    passed_candidates = [
        candidate
        for candidate, decision
        in decisions.items()
        if decision["passed"]
    ]

    recommendation = (
        passed_candidates[-1]
        if passed_candidates
        else "fixed_10"
    )

    report = {
        "symbol": SYMBOL,
        "features": MODEL_FEATURES,
        "fold_results": fold_rows,
        "summaries": summaries,
        "candidate_decisions": decisions,
        "recommendation": {
            "recommended_policy":
                recommendation,
            "passed_candidates":
                passed_candidates,
            "decision": (
                f"Advance {recommendation} "
                "for limited paper research."
                if recommendation != "fixed_10"
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
        "\n=== FOCUSED SIZING RESULT ==="
    )

    print(
        json.dumps(
            report["recommendation"],
            indent=2,
        )
    )

    print(
        "\n=== CANDIDATE DECISIONS ==="
    )

    print(
        json.dumps(
            decisions,
            indent=2,
        )
    )

    print(f"\nJSON report: {JSON_PATH}")
    print(f"CSV report: {CSV_PATH}")


if __name__ == "__main__":
    run()
