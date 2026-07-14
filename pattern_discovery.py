"""Read-only pattern discovery for the GLD research engine."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from trade_memory import DATABASE_PATH
from trade_review import grade_return


MINIMUM_CLOSED_TRADES_FOR_RECOMMENDATION = 10
MINIMUM_GROUP_TRADES = 3


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def load_joined_records() -> list[dict[str, Any]]:
    if not Path(DATABASE_PATH).exists():
        return []

    connection = connect()

    try:
        rows = connection.execute(
            """
            SELECT
                t.id AS trade_id,
                t.status,
                t.entry_order_id,
                t.exit_order_id,
                t.entry_timestamp_utc,
                t.exit_timestamp_utc,
                t.entry_price,
                t.exit_price,
                t.notional,
                t.quantity,
                t.entry_probability_up,
                t.position_percent,
                t.exit_reason,
                t.gross_return,
                t.gross_profit_loss,
                ti.market_regime_at_entry,
                ti.trend_state_at_entry,
                ti.volatility_state_at_entry,
                ti.momentum_state_at_entry,
                ti.confidence_bucket_at_entry,
                di.market_regime AS decision_market_regime,
                di.trend_state AS decision_trend_state,
                di.volatility_state AS decision_volatility_state,
                di.momentum_state AS decision_momentum_state,
                di.confidence_bucket AS decision_confidence_bucket,
                d.probability_up AS decision_probability_up
            FROM trades t
            LEFT JOIN trade_intelligence ti
                ON ti.trade_id = t.id
            LEFT JOIN decisions d
                ON d.order_id = t.entry_order_id
                AND d.action = 'BUY'
            LEFT JOIN decision_intelligence di
                ON di.decision_id = d.id
            ORDER BY t.id
            """
        ).fetchall()

        records = []

        for row in rows:
            item = dict(row)

            item["market_regime"] = (
                item["market_regime_at_entry"]
                or item["decision_market_regime"]
                or "unknown"
            )

            item["trend_state"] = (
                item["trend_state_at_entry"]
                or item["decision_trend_state"]
                or "unknown"
            )

            item["volatility_state"] = (
                item["volatility_state_at_entry"]
                or item["decision_volatility_state"]
                or "unknown"
            )

            item["momentum_state"] = (
                item["momentum_state_at_entry"]
                or item["decision_momentum_state"]
                or "unknown"
            )

            item["confidence_bucket"] = (
                item["confidence_bucket_at_entry"]
                or item["decision_confidence_bucket"]
                or "unknown"
            )

            item["probability_up"] = (
                item["entry_probability_up"]
                if item["entry_probability_up"] is not None
                else item["decision_probability_up"]
            )

            records.append(item)

        return records

    finally:
        connection.close()


def profit_factor(profit_losses: list[float]) -> float:
    gross_profit = sum(
        value for value in profit_losses if value > 0
    )

    gross_loss = abs(
        sum(value for value in profit_losses if value < 0)
    )

    if gross_loss > 0:
        return gross_profit / gross_loss

    if gross_profit > 0:
        return math.inf

    return 0.0


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [
        record
        for record in records
        if record["status"] == "CLOSED"
        and record["gross_return"] is not None
    ]

    returns = [
        float(record["gross_return"])
        for record in closed
    ]

    profit_losses = [
        float(record["gross_profit_loss"])
        for record in closed
        if record["gross_profit_loss"] is not None
    ]

    wins = [
        value
        for value in returns
        if value > 0
    ]

    losses = [
        value
        for value in returns
        if value < 0
    ]

    return {
        "total_trades": len(records),
        "open_trades": sum(
            record["status"] == "OPEN"
            for record in records
        ),
        "closed_trades": len(closed),
        "win_rate": (
            len(wins) / len(returns)
            if returns
            else 0.0
        ),
        "average_return": (
            sum(returns) / len(returns)
            if returns
            else 0.0
        ),
        "median_return": (
            sorted(returns)[len(returns) // 2]
            if returns
            else 0.0
        ),
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "profit_factor": float(
            profit_factor(profit_losses)
        ),
        "realized_profit_loss": float(
            sum(profit_losses)
        ),
        "grade_counts": {
            grade: sum(
                grade_return(record["gross_return"]) == grade
                for record in closed
            )
            for grade in (
                "A+",
                "A",
                "B",
                "C",
                "D",
                "F",
            )
        },
    }


def group_by(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        key = str(record.get(field) or "unknown")
        grouped[key].append(record)

    return {
        key: summarize_group(group)
        for key, group in sorted(grouped.items())
    }


def build_observations(
    records: list[dict[str, Any]],
    grouped_results: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    observations: list[str] = []

    closed_count = sum(
        record["status"] == "CLOSED"
        for record in records
    )

    open_count = sum(
        record["status"] == "OPEN"
        for record in records
    )

    if open_count:
        observations.append(
            f"{open_count} trade(s) are still open, so their final outcomes "
            "are not yet available for pattern analysis."
        )

    if closed_count == 0:
        observations.append(
            "No completed trades are available yet. Current results describe "
            "exposure and market conditions only."
        )

    for dimension, groups in grouped_results.items():
        populated = [
            (name, metrics)
            for name, metrics in groups.items()
            if metrics["closed_trades"] >= MINIMUM_GROUP_TRADES
        ]

        if not populated:
            continue

        best_name, best_metrics = max(
            populated,
            key=lambda item: item[1]["average_return"],
        )

        observations.append(
            f"Among sufficiently populated {dimension} groups, "
            f"{best_name} currently has the highest average return "
            f"({best_metrics['average_return']:.2%})."
        )

    return observations


def build_recommendations(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    closed = [
        record
        for record in records
        if record["status"] == "CLOSED"
    ]

    if len(closed) < MINIMUM_CLOSED_TRADES_FOR_RECOMMENDATION:
        return [
            {
                "recommendation": "Keep the current production strategy unchanged.",
                "status": "INSUFFICIENT_DATA",
                "reason": (
                    f"Only {len(closed)} completed trade(s) are available. "
                    f"At least {MINIMUM_CLOSED_TRADES_FOR_RECOMMENDATION} "
                    "completed trades are required before the research engine "
                    "will suggest a strategy change."
                ),
            }
        ]

    return [
        {
            "recommendation": "Continue collecting data and review the strongest groups.",
            "status": "REVIEW_REQUIRED",
            "reason": (
                "Enough completed trades are now available for deeper analysis, "
                "but no production change should be made without a separate "
                "walk-forward validation test."
            ),
        }
    ]


def discover_patterns() -> dict[str, Any]:
    records = load_joined_records()

    grouped_results = {
        "confidence_bucket": group_by(
            records,
            "confidence_bucket",
        ),
        "market_regime": group_by(
            records,
            "market_regime",
        ),
        "trend_state": group_by(
            records,
            "trend_state",
        ),
        "volatility_state": group_by(
            records,
            "volatility_state",
        ),
        "momentum_state": group_by(
            records,
            "momentum_state",
        ),
    }

    return {
        "overall": summarize_group(records),
        "groups": grouped_results,
        "observations": build_observations(
            records,
            grouped_results,
        ),
        "recommendations": build_recommendations(records),
        "thresholds": {
            "minimum_closed_trades_for_recommendation":
                MINIMUM_CLOSED_TRADES_FOR_RECOMMENDATION,
            "minimum_group_trades":
                MINIMUM_GROUP_TRADES,
        },
        "order_submitted": False,
    }
