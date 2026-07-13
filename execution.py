"""Adaptive limit-price calculation for GLD v5."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LimitPlan:
    limit_price: float
    confidence_band: str


def build_limit_plan(current_price: float, atr_percent: float, probability_up: float) -> LimitPlan:
    if probability_up >= 0.85:
        atr_multiplier, premium_cap, band = 0.25, 0.0030, "very_high"
    elif probability_up >= 0.70:
        atr_multiplier, premium_cap, band = 0.18, 0.0020, "high"
    else:
        atr_multiplier, premium_cap, band = 0.10, 0.0010, "standard"

    atr_price = current_price + current_price * atr_percent * atr_multiplier
    capped_price = current_price * (1 + premium_cap)

    return LimitPlan(
        limit_price=round(min(atr_price, capped_price), 2),
        confidence_band=band,
    )
