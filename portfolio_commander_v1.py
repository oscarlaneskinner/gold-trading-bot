"""Portfolio Commander v1.

Reads the Strategy Hall of Fame and creates a research-only capital allocation
plan for GLD, long, and short strategy roles.

No orders are submitted and no production configuration is changed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DATABASE_PATH = Path("data/strategy_hall_of_fame.sqlite3")
REPORT_DIR = Path("reports/portfolio")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_strategies() -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "Hall of Fame database does not exist. Run strategy_hall_of_fame_v1.py first."
        )
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query("SELECT * FROM strategies", connection)


def normalize(values: pd.Series, reverse: bool = False) -> pd.Series:
    if values.empty:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        result = pd.Series(1.0, index=values.index)
    else:
        result = (values - minimum) / (maximum - minimum)
    return 1.0 - result if reverse else result


def choose_candidates(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    eligible = frame[
        (frame["score"] >= float(config["minimum_score"]))
        & (frame["profit_factor"] >= float(config["minimum_profit_factor"]))
        & (frame["drawdown_percent"] <= float(config["maximum_drawdown_percent"]))
        & (frame["consistency_percent"] >= float(config["minimum_consistency_percent"]))
    ].copy()

    if eligible.empty:
        return eligible

    weights = config["allocation_weights"]
    eligible["quality_score"] = (
        normalize(eligible["score"]) * float(weights["score"])
        + normalize(eligible["profit_factor"]) * float(weights["profit_factor"])
        + normalize(eligible["consistency_percent"]) * float(weights["consistency"])
        + normalize(eligible["drawdown_percent"], reverse=True) * float(weights["drawdown"])
    )

    selected_rows = []
    for role in config["preferred_roles"]:
        role_rows = eligible[eligible["role"] == role].sort_values(
            ["quality_score", "score"],
            ascending=False,
        )
        if not role_rows.empty:
            selected_rows.append(role_rows.iloc[0])

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        selected = eligible.sort_values("quality_score", ascending=False).head(
            int(config["maximum_active_strategies"])
        )

    return selected.head(int(config["maximum_active_strategies"]))


def allocate(selected: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    if selected.empty:
        return []

    total_exposure = min(
        100.0 - float(config["cash_reserve_percent"]),
        float(config["maximum_total_exposure_percent"]),
    )

    quality = selected["quality_score"].clip(lower=0.001)
    raw_weights = quality / quality.sum()

    maximum = float(config["maximum_strategy_allocation_percent"])
    minimum = float(config["minimum_strategy_allocation_percent"])

    allocations = raw_weights * total_exposure
    allocations = allocations.clip(lower=minimum, upper=maximum)

    if allocations.sum() > total_exposure:
        allocations = allocations * (total_exposure / allocations.sum())

    capital = float(config["starting_capital"])
    results = []

    for (_, row), allocation in zip(selected.iterrows(), allocations):
        results.append(
            {
                "strategy_name": row["strategy_name"],
                "role": row["role"],
                "symbol": row["symbol"],
                "allocation_percent": round(float(allocation), 2),
                "allocation_dollars": round(capital * float(allocation) / 100, 2),
                "score": round(float(row["score"]), 4),
                "profit_factor": round(float(row["profit_factor"]), 4),
                "drawdown_percent": round(float(row["drawdown_percent"]), 4),
                "consistency_percent": round(float(row["consistency_percent"]), 4),
                "status": "RESEARCH_ALLOCATION_ONLY",
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/portfolio_commander_v1.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    strategies = load_strategies()
    selected = choose_candidates(strategies, config)
    allocations = allocate(selected, config)

    allocated_percent = sum(item["allocation_percent"] for item in allocations)
    cash_percent = round(100.0 - allocated_percent, 2)
    capital = float(config["starting_capital"])

    output = {
        "starting_capital": capital,
        "allocations": allocations,
        "allocated_percent": round(allocated_percent, 2),
        "cash_reserve_percent": cash_percent,
        "cash_reserve_dollars": round(capital * cash_percent / 100, 2),
        "eligible_strategy_count": int(len(selected)),
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "portfolio_commander_v1.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(allocations).to_csv(
        REPORT_DIR / "portfolio_allocations.csv",
        index=False,
    )

    print("Portfolio Commander v1")
    print(json.dumps(output, indent=2))
    print("Research allocation only.")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
