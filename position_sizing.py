"""Configuration-driven fixed position sizing for controlled paper research."""

from __future__ import annotations

import os


def get_position_percent() -> float:
    raw_value = os.getenv("POSITION_PERCENT", "0.10")

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"POSITION_PERCENT must be numeric, received {raw_value!r}."
        ) from error

    if not 0.01 <= value <= 0.15:
        raise ValueError(
            "POSITION_PERCENT must be between 0.01 and 0.15 "
            "for the current paper-research phase."
        )

    return value


def calculate_fixed_notional(
    account_equity: float,
    position_percent: float,
    minimum_order_amount: float,
) -> float:
    if account_equity <= 0:
        raise ValueError("account_equity must be positive.")

    notional = account_equity * position_percent

    return round(
        max(minimum_order_amount, notional),
        2,
    )
