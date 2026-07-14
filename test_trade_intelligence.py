"""Read-only test of the GLD trade-intelligence schema and classifier."""

from __future__ import annotations

import json

from trade_intelligence import (
    intelligence_summary,
    record_decision_intelligence,
)
from trade_memory import record_decision


def run():
    features = {
        "return_1d": 0.002,
        "return_5d": 0.012,
        "return_10d": 0.020,
        "return_20d": 0.035,
        "price_vs_ema200": 0.080,
        "ema9_vs_ema21": 0.010,
        "ema21_vs_ema50": 0.015,
        "rsi_14": 61.0,
        "rsi_7": 63.0,
        "atr_pct": 0.018,
        "volatility_20d": 0.012,
        "volume_change": 0.050,
        "volume_ma_ratio": 1.10,
    }

    decision_id = record_decision(
        symbol="GLD",
        model_name="lightgbm",
        model_version="lgbm_d",
        prediction=1,
        probability_up=0.86,
        price=375.00,
        action="HOLD",
        reason="Synthetic research-engine test only.",
        position_percent=0.15,
        notional=None,
        market_open=False,
        existing_position=True,
        open_buy_order_count=0,
        order_id=None,
        paper_trading=True,
        market_timestamp="2026-07-14T00:00:00+00:00",
        features=features,
    )

    intelligence = record_decision_intelligence(
        decision_id=decision_id,
        probability_up=0.86,
        features=features,
    )

    print("GLD research-engine Package 6.1 test")
    print(
        json.dumps(
            {
                "status": "passed",
                "decision_id": decision_id,
                "intelligence": intelligence,
                "summary": intelligence_summary(),
                "market_request_made": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
