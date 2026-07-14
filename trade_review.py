"""Read-only trade grading and performance summaries for GLD trade memory."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from trade_memory import database_summary, recent_trades


def grade_return(return_value: float | None) -> str:
    """
    Grade a completed trade by realized return.

    Open trades remain PENDING and are never graded as winners or losers.
    """

    if return_value is None:
        return "PENDING"

    if return_value >= 0.10:
        return "A+"
    if return_value >= 0.05:
        return "A"
    if return_value >= 0.02:
        return "B"
    if return_value >= 0:
        return "C"
    if return_value > -0.03:
        return "D"
    return "F"


def holding_days(
    entry_timestamp: str | None,
    exit_timestamp: str | None,
) -> float | None:
    if not entry_timestamp or not exit_timestamp:
        return None

    entry = datetime.fromisoformat(entry_timestamp)
    exit_time = datetime.fromisoformat(exit_timestamp)

    return (
        exit_time - entry
    ).total_seconds() / 86_400


def review_trade(
    trade: dict[str, Any],
    current_price: float | None = None,
) -> dict[str, Any]:
    status = trade["status"]
    entry_price = trade["entry_price"]
    exit_price = trade["exit_price"]
    quantity = trade["quantity"]

    unrealized_return = None
    unrealized_profit_loss = None

    if (
        status == "OPEN"
        and current_price is not None
        and entry_price is not None
    ):
        unrealized_return = (
            float(current_price)
            / float(entry_price)
            - 1
        )

        if quantity is not None:
            unrealized_profit_loss = (
                float(quantity)
                * (
                    float(current_price)
                    - float(entry_price)
                )
            )

    realized_return = trade["gross_return"]
    realized_profit_loss = trade["gross_profit_loss"]

    return {
        "trade_id": trade["id"],
        "symbol": trade["symbol"],
        "status": status,
        "grade": grade_return(realized_return),
        "classification": (
            "OPEN"
            if status == "OPEN"
            else (
                "WIN"
                if float(realized_return or 0) > 0
                else (
                    "LOSS"
                    if float(realized_return or 0) < 0
                    else "BREAKEVEN"
                )
            )
        ),
        "entry_order_id": trade["entry_order_id"],
        "exit_order_id": trade["exit_order_id"],
        "entry_timestamp_utc": trade["entry_timestamp_utc"],
        "exit_timestamp_utc": trade["exit_timestamp_utc"],
        "holding_days": holding_days(
            trade["entry_timestamp_utc"],
            trade["exit_timestamp_utc"],
        ),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "current_price": current_price if status == "OPEN" else None,
        "quantity": quantity,
        "notional": trade["notional"],
        "entry_probability_up": trade["entry_probability_up"],
        "position_percent": trade["position_percent"],
        "exit_reason": trade["exit_reason"],
        "realized_return": realized_return,
        "realized_profit_loss": realized_profit_loss,
        "unrealized_return": unrealized_return,
        "unrealized_profit_loss": unrealized_profit_loss,
    }


def performance_summary(
    reviewed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    closed = [
        trade
        for trade in reviewed_trades
        if trade["status"] == "CLOSED"
    ]

    open_trades = [
        trade
        for trade in reviewed_trades
        if trade["status"] == "OPEN"
    ]

    realized_returns = [
        float(trade["realized_return"])
        for trade in closed
        if trade["realized_return"] is not None
    ]

    realized_profit_loss = [
        float(trade["realized_profit_loss"])
        for trade in closed
        if trade["realized_profit_loss"] is not None
    ]

    wins = [
        value
        for value in realized_returns
        if value > 0
    ]

    losses = [
        value
        for value in realized_returns
        if value < 0
    ]

    gross_profit = sum(
        value
        for value in realized_profit_loss
        if value > 0
    )

    gross_loss = abs(
        sum(
            value
            for value in realized_profit_loss
            if value < 0
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            math.inf
            if gross_profit > 0
            else 0.0
        )
    )

    holding_periods = [
        float(trade["holding_days"])
        for trade in closed
        if trade["holding_days"] is not None
    ]

    open_unrealized = [
        float(trade["unrealized_profit_loss"])
        for trade in open_trades
        if trade["unrealized_profit_loss"] is not None
    ]

    return {
        "total_trades": len(reviewed_trades),
        "open_trades": len(open_trades),
        "closed_trades": len(closed),
        "win_rate": (
            len(wins) / len(realized_returns)
            if realized_returns
            else 0.0
        ),
        "average_realized_return": (
            float(np.mean(realized_returns))
            if realized_returns
            else 0.0
        ),
        "median_realized_return": (
            float(np.median(realized_returns))
            if realized_returns
            else 0.0
        ),
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "profit_factor": float(profit_factor),
        "realized_profit_loss": float(
            sum(realized_profit_loss)
        ),
        "open_unrealized_profit_loss": float(
            sum(open_unrealized)
        ),
        "average_holding_days": (
            float(np.mean(holding_periods))
            if holding_periods
            else 0.0
        ),
        "grade_counts": {
            grade: sum(
                trade["grade"] == grade
                for trade in reviewed_trades
            )
            for grade in (
                "A+",
                "A",
                "B",
                "C",
                "D",
                "F",
                "PENDING",
            )
        },
    }


def build_review(
    current_price: float | None = None,
) -> dict[str, Any]:
    trades = recent_trades(limit=10_000)

    reviewed = [
        review_trade(
            trade,
            current_price=current_price,
        )
        for trade in trades
    ]

    return {
        "database_summary": database_summary(),
        "performance_summary": performance_summary(reviewed),
        "trades": reviewed,
    }
