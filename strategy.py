"""Trading signal and filter logic."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from config import (
    BEARISH_EXIT_PROBABILITY, MAX_ATR_PERCENT, MIN_BUY_CONFIDENCE,
    RSI_MAXIMUM, RSI_MINIMUM, USE_RSI_FILTER, USE_TREND_FILTER,
    USE_VOLATILITY_FILTER,
)

@dataclass(frozen=True)
class StrategyDecision:
    action: str
    reason: str
    checks: dict[str, Any]

def evaluate_entry(prediction: int, probability_up: float, row) -> StrategyDecision:
    checks = {
        "prediction_up": prediction == 1,
        "confidence_ok": probability_up >= MIN_BUY_CONFIDENCE,
        "trend_ok": True, "rsi_ok": True, "volatility_ok": True,
    }
    if USE_TREND_FILTER:
        checks["trend_ok"] = bool(
            row["close"] > row["ema_200"]
            and row["ema_9"] > row["ema_21"] > row["ema_50"]
        )
    if USE_RSI_FILTER:
        checks["rsi_ok"] = bool(RSI_MINIMUM <= row["rsi_14"] < RSI_MAXIMUM)
    if USE_VOLATILITY_FILTER:
        checks["volatility_ok"] = bool(row["atr_pct"] <= MAX_ATR_PERCENT)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return StrategyDecision("HOLD", "Entry blocked: " + ", ".join(failed), checks)
    return StrategyDecision("BUY", f"Bullish AI signal at {probability_up:.1%} confidence.", checks)

def model_exit_signal(probability_up: float) -> StrategyDecision:
    if probability_up <= BEARISH_EXIT_PROBABILITY:
        return StrategyDecision("SELL", f"AI probability fell to {probability_up:.1%}.", {"bearish_model_exit": True})
    return StrategyDecision("HOLD", "AI exit threshold not reached.", {"bearish_model_exit": False})
