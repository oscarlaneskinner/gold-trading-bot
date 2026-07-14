"""Position sizing and exit-condition calculations."""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    DEFAULT_TRADE_AMOUNT,
    ENABLE_TRAILING_STOP,
    MAX_HOLD_DAYS,
    MIN_ORDER_AMOUNT,
    POSITION_PERCENT,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    TRAILING_ACTIVATION_PERCENT,
    TRAILING_STOP_PERCENT,
)


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str | None


def calculate_notional(
    account_equity: float,
    confidence: float,
) -> float:
    """
    Return a fixed percentage of account equity.

    Confidence remains an input for API compatibility and logging, but the
    focused research validated fixed sizing rather than confidence-scaled
    sizing.
    """

    if account_equity <= 0:
        return DEFAULT_TRADE_AMOUNT

    notional = (
        account_equity
        * POSITION_PERCENT
    )

    return round(
        max(
            MIN_ORDER_AMOUNT,
            min(
                notional,
                account_equity,
            ),
        ),
        2,
    )


def check_exit_conditions(
    entry_price: float,
    latest_price: float,
    peak_price: float,
    days_held: int,
) -> ExitDecision:
    if latest_price <= (
        entry_price
        * (1 - STOP_LOSS_PERCENT)
    ):
        return ExitDecision(
            True,
            "stop_loss",
        )

    if latest_price >= (
        entry_price
        * (1 + TAKE_PROFIT_PERCENT)
    ):
        return ExitDecision(
            True,
            "take_profit",
        )

    if (
        ENABLE_TRAILING_STOP
        and peak_price
        >= entry_price
        * (
            1
            + TRAILING_ACTIVATION_PERCENT
        )
    ):
        if latest_price <= (
            peak_price
            * (
                1
                - TRAILING_STOP_PERCENT
            )
        ):
            return ExitDecision(
                True,
                "trailing_stop",
            )

    if days_held >= MAX_HOLD_DAYS:
        return ExitDecision(
            True,
            "maximum_holding_period",
        )

    return ExitDecision(
        False,
        None,
    )
