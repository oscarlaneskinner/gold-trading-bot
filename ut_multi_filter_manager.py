"""Summarize multi-filter championship competitors."""

from __future__ import annotations

import json
from pathlib import Path

from research_tournament_db import load_experiments


REPORT_PATH = Path("reports/ut_multi_filter_championship.json")


def run() -> None:
    records = [
        item
        for item in load_experiments()
        if item["strategy_name"] == "UT Multi-Filter Championship"
    ]

    records.sort(
        key=lambda item: (
            item["score"],
            item["net_profit_percent"],
            -abs(item["max_drawdown_percent"]),
        ),
        reverse=True,
    )

    for rank, item in enumerate(records, start=1):
        item["rank"] = rank

    report = {
        "competitor_count": len(records),
        "champion": records[0] if records else None,
        "leaderboard": records,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("UT Multi-Filter Championship")
    print(json.dumps(report, indent=2))
    print(f"JSON report: {REPORT_PATH}")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
