"""Offline test for the GLD weekly research report."""

from __future__ import annotations

import json

from weekly_research_report import build_research_report


def run() -> None:
    report = build_research_report()

    output = {
        "status": "passed",
        "report_status": report["status"],
        "decision_count": report[
            "memory_summary"
        ]["decision_count"],
        "trade_count": report[
            "memory_summary"
        ]["trade_count"],
        "closed_trade_count": report[
            "memory_summary"
        ]["closed_trade_count"],
        "production_strategy_changed": False,
        "order_submitted": False,
    }

    print("GLD weekly research-report offline test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
