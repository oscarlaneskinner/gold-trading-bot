"""Guided registration for alternative-indicator championship results."""

from __future__ import annotations

import json

from research_command_center import run as refresh_command_center
from research_tournament_db import build_leaderboard, register_experiment
from ut_championship_manager import run as refresh_championship


def ask(prompt: str, default: str = "") -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def ask_float(prompt: str, default: str = "") -> float:
    while True:
        try:
            return float(ask(prompt, default))
        except ValueError:
            print("Enter a valid number.")


def ask_int(prompt: str, default: str = "") -> int:
    while True:
        try:
            return int(ask(prompt, default))
        except ValueError:
            print("Enter a whole number.")


def run() -> None:
    print("UT ALTERNATIVE INDICATORS CHAMPIONSHIP WIZARD")
    print("=" * 47)

    payload = {
        "asset": ask("Asset", "GLD"),
        "venue": ask("Venue", "NYSE Arca"),
        "strategy_name": ask(
            "Strategy name",
            "UT Alternative Indicators Championship",
        ),
        "strategy_variant": ask(
            "Competitor",
            "UT + B-Xtrender Style",
        ),
        "timeframe": ask("Timeframe", "1D"),
        "date_start": ask("Start date", "2004-11-18"),
        "date_end": ask("End date", "2026-07-15"),
        "starting_capital": ask_float("Starting capital", "100000"),
        "commission_percent": ask_float("Commission percent", "0.01"),
        "slippage_units": ask_float("Slippage units", "1"),
        "position_percent": ask_float("Position size percent", "15"),
        "net_profit_amount": ask_float("Net profit amount"),
        "net_profit_percent": ask_float("Net profit percent"),
        "max_drawdown_amount": ask_float("Maximum drawdown amount"),
        "max_drawdown_percent": ask_float("Maximum drawdown percent"),
        "profitable_trades_percent": ask_float("Profitable trades percent"),
        "profit_factor": ask_float("Profit factor"),
        "closed_trades": ask_int("Closed trades"),
        "market_regime": ask("Market regime", ""),
        "source_platform": ask("Source platform", "TradingView"),
        "strategy_version": ask("Strategy version", "5.0"),
        "notes": ask("Notes", ""),
    }

    print(json.dumps(payload, indent=2))

    if ask("Save this result? yes/no", "yes").lower() not in {"yes", "y"}:
        print("Result not saved.")
        return

    record = register_experiment(payload)
    leaderboard = build_leaderboard(
        asset=record["asset"],
        timeframe=record["timeframe"],
        minimum_trades=30,
    )

    refresh_championship()
    refresh_command_center()

    rank = next(
        (
            item["rank"]
            for item in leaderboard["leaderboard"]
            if item["experiment_code"] == record["experiment_code"]
        ),
        None,
    )

    print(
        json.dumps(
            {
                "status": "saved",
                "experiment_code": record["experiment_code"],
                "score": record["score"],
                "rank": rank,
                "leaderboard_count": leaderboard["experiment_count"],
                "reports_refreshed": True,
                "market_request_made": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
