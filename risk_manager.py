"""Position sizing and exit-condition calculations."""

from __future__ import annotations
from dataclasses import dataclass
from config import (
    DEFAULT_TRADE_AMOUNT, ENABLE_TRAILING_STOP, MAX_HOLD_DAYS,
    MAX_POSITION_PERCENT, MIN_ORDER_AMOUNT, STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT, TRAILING_ACTIVATION_PERCENT, TRAILING_STOP_PERCENT,
)

@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str | None

def calculate_notional(account_equity: float, confidence: float) -> float:
    if account_equity <= 0:
        return DEFAULT_TRADE_AMOUNT
    maximum = account_equity * MAX_POSITION_PERCENT
    notional = maximum * max(0.25, min(1.0, confidence))
    return round(max(MIN_ORDER_AMOUNT, min(notional, maximum)), 2)

def check_exit_conditions(entry_price: float, latest_price: float, peak_price: float, days_held: int) -> ExitDecision:
    if latest_price <= entry_price * (1 - STOP_LOSS_PERCENT):
        return ExitDecision(True, "stop_loss")
    if latest_price >= entry_price * (1 + TAKE_PROFIT_PERCENT):
        return ExitDecision(True, "take_profit")
    if ENABLE_TRAILING_STOP and peak_price >= entry_price * (1 + TRAILING_ACTIVATION_PERCENT):
        if latest_price <= peak_price * (1 - TRAILING_STOP_PERCENT):
            return ExitDecision(True, "trailing_stop")
    if days_held >= MAX_HOLD_DAYS:
        return ExitDecision(True, "maximum_holding_period")
    return ExitDecision(False, None)
