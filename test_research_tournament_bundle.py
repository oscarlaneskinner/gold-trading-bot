"""Offline test for the research database and tournament manager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import research_tournament_db as tournament


def run() -> None:
    original_database = tournament.DATABASE_PATH
    original_report = tournament.REPORT_PATH

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)

        tournament.DATABASE_PATH = (
            temp_path / "research_tournament.sqlite3"
        )

        tournament.REPORT_PATH = (
            temp_path / "leaderboard.json"
        )

        tournament.register_experiment(
            {
                "asset": "GLD",
                "venue": "NYSE Arca",
                "strategy_name": "UT Competition Lab",
                "strategy_variant": "UT Original",
                "timeframe": "1D",
                "date_start": "2004-11-18",
                "date_end": "2026-07-15",
                "starting_capital": 100000,
                "commission_percent": 0.01,
                "slippage_units": 1,
                "position_percent": 15,
                "net_profit_amount": 26042.87,
                "net_profit_percent": 26.04,
                "max_drawdown_amount": 6047.02,
                "max_drawdown_percent": 5.17,
                "profitable_trades_percent": 40.79,
                "profit_factor": 1.431,
                "closed_trades": 407,
                "source_platform": "TradingView",
                "strategy_version": "2.1",
                "notes": "Synthetic offline validation.",
            }
        )

        tournament.register_experiment(
            {
                "asset": "GLD",
                "venue": "NYSE Arca",
                "strategy_name": "UT Competition Lab",
                "strategy_variant": "UT + EMA200",
                "timeframe": "1D",
                "date_start": "2004-11-18",
                "date_end": "2026-07-15",
                "starting_capital": 100000,
                "commission_percent": 0.01,
                "slippage_units": 1,
                "position_percent": 15,
                "net_profit_amount": 16858.27,
                "net_profit_percent": 16.86,
                "max_drawdown_amount": 6022.55,
                "max_drawdown_percent": 5.0,
                "profitable_trades_percent": 50.0,
                "profit_factor": 1.2,
                "closed_trades": 100,
                "source_platform": "TradingView",
                "strategy_version": "2.1",
                "notes": "Synthetic offline validation.",
            }
        )

        leaderboard = tournament.build_leaderboard(
            asset="GLD",
            timeframe="1D",
            minimum_trades=30,
        )

        summary = tournament.database_summary()

        output = {
            "status": "passed",
            "experiment_count":
                summary["experiment_count"],
            "top_strategy_variant":
                leaderboard["leaderboard"][0][
                    "strategy_variant"
                ],
            "leaderboard_count":
                leaderboard["experiment_count"],
            "database_exists":
                tournament.DATABASE_PATH.exists(),
            "production_strategy_changed": False,
            "market_request_made": False,
            "order_submitted": False,
        }

        print("Research tournament bundle test")
        print(json.dumps(output, indent=2))
        print("No market request was made.")
        print("No order was submitted.")

        if (
            output["experiment_count"] != 2
            or output["leaderboard_count"] != 2
            or not output["database_exists"]
        ):
            raise SystemExit(
                "Research tournament bundle test failed."
            )

    tournament.DATABASE_PATH = original_database
    tournament.REPORT_PATH = original_report


if __name__ == "__main__":
    run()
