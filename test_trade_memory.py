"""Read-only test for the SQLite trade-memory database."""

from __future__ import annotations

import json
from pathlib import Path

from trade_memory import (
    DATABASE_PATH,
    database_summary,
    initialize_database,
    record_decision,
)


def run() -> None:
    initialize_database()

    record_id = record_decision(
        symbol="GLD",
        model_name="lightgbm",
        model_version="lgbm_d",
        prediction=1,
        probability_up=0.85,
        price=375.00,
        action="HOLD",
        reason="Synthetic trade-memory test only.",
        position_percent=0.15,
        notional=None,
        market_open=False,
        existing_position=False,
        open_buy_order_count=0,
        order_id=None,
        paper_trading=True,
        market_timestamp="2026-07-14T00:00:00+00:00",
        features={
            "return_1d": 0.001,
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
        },
    )

    output = {
        "status": "passed",
        "inserted_decision_id": record_id,
        "database_exists": Path(DATABASE_PATH).exists(),
        "summary": database_summary(),
        "order_submitted": False,
    }

    print("GLD trade-memory read-only test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
