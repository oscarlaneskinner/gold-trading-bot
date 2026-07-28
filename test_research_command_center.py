"""Offline test of command-center report generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import research_command_center as command_center
import research_tournament_db as tournament


def run() -> None:
    original_database = tournament.DATABASE_PATH
    original_report = tournament.REPORT_PATH
    original_json = command_center.JSON_REPORT_PATH
    original_html = command_center.HTML_REPORT_PATH
    original_text = command_center.TEXT_REPORT_PATH

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)

        tournament.DATABASE_PATH = (
            directory / "research.sqlite3"
        )

        tournament.REPORT_PATH = (
            directory / "leaderboard.json"
        )

        command_center.JSON_REPORT_PATH = (
            directory / "daily.json"
        )

        command_center.HTML_REPORT_PATH = (
            directory / "daily.html"
        )

        command_center.TEXT_REPORT_PATH = (
            directory / "daily.txt"
        )

        tournament.register_experiment(
            {
                "asset": "GLD",
                "venue": "NYSE Arca",
                "strategy_name":
                    "UT Competition Lab",
                "strategy_variant":
                    "UT Original",
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
            }
        )

        report = command_center.build_report()
        text_report = command_center.build_text(
            report
        )
        html_report = command_center.build_html(
            report
        )

        command_center.JSON_REPORT_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        command_center.TEXT_REPORT_PATH.write_text(
            text_report,
            encoding="utf-8",
        )

        command_center.HTML_REPORT_PATH.write_text(
            html_report,
            encoding="utf-8",
        )

        output = {
            "status": "passed",
            "champion":
                report["top_strategy"][
                    "strategy_variant"
                ],
            "experiment_count":
                report["database_summary"][
                    "experiment_count"
                ],
            "json_created":
                command_center.JSON_REPORT_PATH.exists(),
            "html_created":
                command_center.HTML_REPORT_PATH.exists(),
            "text_created":
                command_center.TEXT_REPORT_PATH.exists(),
            "email_sent": False,
            "market_request_made": False,
            "order_submitted": False,
        }

        print(
            "Research command-center test"
        )
        print(json.dumps(output, indent=2))
        print("No email was sent.")
        print("No market request was made.")
        print("No order was submitted.")

        if not all(
            [
                output["json_created"],
                output["html_created"],
                output["text_created"],
            ]
        ):
            raise SystemExit(
                "Command-center report test failed."
            )

    tournament.DATABASE_PATH = original_database
    tournament.REPORT_PATH = original_report
    command_center.JSON_REPORT_PATH = original_json
    command_center.HTML_REPORT_PATH = original_html
    command_center.TEXT_REPORT_PATH = original_text


if __name__ == "__main__":
    run()
