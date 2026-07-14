"""Generate a conservative weekly GLD research report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import REPORTS_DIR
from logger import write_json
from pattern_discovery import discover_patterns
from trade_intelligence import intelligence_summary
from trade_memory import database_summary
from trade_review import build_review


JSON_PATH = REPORTS_DIR / "weekly_research_report.json"
TEXT_PATH = REPORTS_DIR / "weekly_research_report.txt"


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def build_research_report() -> dict[str, Any]:
    patterns = discover_patterns()
    memory = database_summary()
    intelligence = intelligence_summary()
    review = build_review(current_price=None)

    performance = review["performance_summary"]

    observations = list(patterns["observations"])

    if memory["closed_trade_count"] < 10:
        observations.append(
            "The completed-trade sample remains too small for dependable "
            "strategy conclusions."
        )

    recommendations = list(patterns["recommendations"])

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "INSUFFICIENT_DATA"
            if memory["closed_trade_count"] < 10
            else "REVIEW_REQUIRED"
        ),
        "memory_summary": memory,
        "intelligence_summary": intelligence,
        "performance_summary": performance,
        "pattern_summary": {
            "groups": patterns["groups"],
            "observations": observations,
        },
        "recommendations": recommendations,
        "safety": {
            "production_strategy_changed": False,
            "market_request_made": False,
            "order_submitted": False,
        },
    }

    return report


def render_text(report: dict[str, Any]) -> str:
    memory = report["memory_summary"]
    intelligence = report["intelligence_summary"]
    performance = report["performance_summary"]

    lines = [
        "GLD WEEKLY RESEARCH REPORT",
        "=" * 30,
        "",
        f"Generated: {report['generated_at_utc']}",
        f"Status: {report['status']}",
        "",
        "TRADE MEMORY",
        f"- Decisions: {memory['decision_count']}",
        f"- Trades: {memory['trade_count']}",
        f"- Open trades: {memory['open_trade_count']}",
        f"- Closed trades: {memory['closed_trade_count']}",
        f"- Realized P/L: {format_money(memory['realized_profit_loss'])}",
        "",
        "PERFORMANCE",
        f"- Win rate: {format_percent(performance['win_rate'])}",
        f"- Average realized return: "
        f"{format_percent(performance['average_realized_return'])}",
        f"- Profit factor: {performance['profit_factor']:.2f}",
        f"- Open unrealized P/L: "
        f"{format_money(performance['open_unrealized_profit_loss'])}",
        "",
        "INTELLIGENCE",
        f"- Intelligence records: "
        f"{intelligence['intelligence_record_count']}",
        f"- Market regimes: "
        f"{json.dumps(intelligence['market_regime_counts'], sort_keys=True)}",
        f"- Confidence buckets: "
        f"{json.dumps(intelligence['confidence_bucket_counts'], sort_keys=True)}",
        "",
        "OBSERVATIONS",
    ]

    observations = report["pattern_summary"]["observations"]

    if observations:
        lines.extend(
            f"- {item}"
            for item in observations
        )
    else:
        lines.append("- No observations available yet.")

    lines.extend(
        [
            "",
            "RECOMMENDATIONS",
        ]
    )

    for item in report["recommendations"]:
        lines.append(
            f"- [{item['status']}] "
            f"{item['recommendation']} "
            f"Reason: {item['reason']}"
        )

    lines.extend(
        [
            "",
            "SAFETY",
            "- Production strategy changed: No",
            "- Market request made: No",
            "- Order submitted: No",
        ]
    )

    return "\n".join(lines) + "\n"


def run() -> None:
    report = build_research_report()

    write_json(
        JSON_PATH,
        report,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEXT_PATH.write_text(
        render_text(report),
        encoding="utf-8",
    )

    print("GLD weekly research report generated")
    print(f"JSON report: {JSON_PATH}")
    print(f"Text report: {TEXT_PATH}")
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision_count": report[
                    "memory_summary"
                ]["decision_count"],
                "trade_count": report[
                    "memory_summary"
                ]["trade_count"],
                "closed_trade_count": report[
                    "memory_summary"
                ]["closed_trade_count"],
                "intelligence_record_count": report[
                    "intelligence_summary"
                ]["intelligence_record_count"],
                "production_strategy_changed": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
