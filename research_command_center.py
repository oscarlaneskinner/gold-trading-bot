"""Generate a daily research-command-center report from tournament data."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_tournament_db import (
    build_leaderboard,
    database_summary,
    load_experiments,
)


JSON_REPORT_PATH = Path(
    "reports/research_command_center_daily.json"
)
HTML_REPORT_PATH = Path(
    "reports/research_command_center_daily.html"
)
TEXT_REPORT_PATH = Path(
    "reports/research_command_center_daily.txt"
)


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"

    return f"{float(value):,.{digits}f}"


def build_report() -> dict[str, Any]:
    summary = database_summary()
    all_records = load_experiments()

    overall = build_leaderboard(
        minimum_trades=30,
    )

    asset_leaderboards: dict[str, Any] = {}

    for asset in sorted(
        summary.get("asset_counts", {})
    ):
        asset_leaderboards[asset] = (
            build_leaderboard(
                asset=asset,
                minimum_trades=30,
            )
        )

    top_strategy = (
        overall["leaderboard"][0]
        if overall["leaderboard"]
        else None
    )

    recent = sorted(
        all_records,
        key=lambda item: item[
            "created_at_utc"
        ],
        reverse=True,
    )[:10]

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database_summary": summary,
        "top_strategy": top_strategy,
        "overall_leaderboard":
            overall["leaderboard"],
        "asset_leaderboards":
            asset_leaderboards,
        "recent_experiments": recent,
        "production_strategy_changed":
            False,
        "market_request_made": False,
        "order_submitted": False,
    }


def build_text(report: dict[str, Any]) -> str:
    lines = [
        "DAILY TRADING RESEARCH COMPETITION",
        "=" * 38,
        "",
        (
            "Generated: "
            f"{report['generated_at_utc']}"
        ),
        (
            "Experiments stored: "
            f"{report['database_summary']['experiment_count']}"
        ),
        "",
    ]

    champion = report["top_strategy"]

    if champion:
        lines.extend(
            [
                "CURRENT CHAMPION",
                "-" * 16,
                (
                    f"{champion['asset']} "
                    f"{champion['timeframe']} | "
                    f"{champion['strategy_name']} — "
                    f"{champion['strategy_variant']}"
                ),
                (
                    "Tournament score: "
                    f"{format_number(champion['score'], 3)}"
                ),
                (
                    "Net profit: "
                    f"{format_number(champion['net_profit_percent'])}%"
                ),
                (
                    "Max drawdown: "
                    f"{format_number(champion['max_drawdown_percent'])}%"
                ),
                (
                    "Profit factor: "
                    f"{format_number(champion['profit_factor'], 3)}"
                ),
                (
                    "Profitable trades: "
                    f"{format_number(champion['profitable_trades_percent'])}%"
                ),
                (
                    "Closed trades: "
                    f"{champion['closed_trades']}"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No experiment currently satisfies "
                "the 30-trade leaderboard minimum.",
                "",
            ]
        )

    lines.extend(
        [
            "OVERALL LEADERBOARD",
            "-" * 19,
        ]
    )

    for item in report[
        "overall_leaderboard"
    ][:10]:
        lines.append(
            (
                f"#{item['rank']} "
                f"{item['asset']} "
                f"{item['timeframe']} — "
                f"{item['strategy_variant']} | "
                f"Score {format_number(item['score'], 3)} | "
                f"Return {format_number(item['net_profit_percent'])}% | "
                f"DD {format_number(item['max_drawdown_percent'])}% | "
                f"PF {format_number(item['profit_factor'], 3)} | "
                f"Trades {item['closed_trades']}"
            )
        )

    if not report["overall_leaderboard"]:
        lines.append("No qualifying results.")

    lines.extend(
        [
            "",
            "RECENT EXPERIMENTS",
            "-" * 18,
        ]
    )

    for item in report[
        "recent_experiments"
    ][:10]:
        lines.append(
            (
                f"{item['experiment_code']} | "
                f"{item['asset']} "
                f"{item['timeframe']} | "
                f"{item['strategy_variant']} | "
                f"Return "
                f"{format_number(item['net_profit_percent'])}% | "
                f"Trades {item['closed_trades']}"
            )
        )

    lines.extend(
        [
            "",
            "Safety: research only; no market "
            "request and no order submission.",
        ]
    )

    return "\n".join(lines)


def build_html(report: dict[str, Any]) -> str:
    champion = report["top_strategy"]

    champion_html = (
        "<p>No qualifying champion yet.</p>"
    )

    if champion:
        champion_html = f"""
        <div class="card">
          <h2>Current Champion</h2>
          <p><strong>{html.escape(champion['asset'])}
          {html.escape(champion['timeframe'])}</strong></p>
          <p>{html.escape(champion['strategy_name'])}
          — {html.escape(champion['strategy_variant'])}</p>
          <p>Score: {format_number(champion['score'], 3)}</p>
          <p>Net profit: {format_number(champion['net_profit_percent'])}%</p>
          <p>Max drawdown: {format_number(champion['max_drawdown_percent'])}%</p>
          <p>Profit factor: {format_number(champion['profit_factor'], 3)}</p>
          <p>Closed trades: {champion['closed_trades']}</p>
        </div>
        """

    rows = []

    for item in report[
        "overall_leaderboard"
    ]:
        rows.append(
            f"""
            <tr>
              <td>{item['rank']}</td>
              <td>{html.escape(item['asset'])}</td>
              <td>{html.escape(item['timeframe'])}</td>
              <td>{html.escape(item['strategy_variant'])}</td>
              <td>{format_number(item['score'], 3)}</td>
              <td>{format_number(item['net_profit_percent'])}%</td>
              <td>{format_number(item['max_drawdown_percent'])}%</td>
              <td>{format_number(item['profit_factor'], 3)}</td>
              <td>{item['closed_trades']}</td>
            </tr>
            """
        )

    table_rows = (
        "".join(rows)
        if rows
        else (
            '<tr><td colspan="9">'
            "No qualifying results.</td></tr>"
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Daily Trading Research Competition</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 32px;
  background: #f5f7fa;
  color: #18212b;
}}
.card {{
  background: white;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: white;
}}
th, td {{
  padding: 10px;
  border-bottom: 1px solid #ddd;
  text-align: left;
}}
th {{
  background: #eef2f6;
}}
.small {{
  color: #5d6875;
  font-size: 0.9rem;
}}
</style>
</head>
<body>
<h1>Daily Trading Research Competition</h1>
<p class="small">Generated {html.escape(report['generated_at_utc'])}</p>
<p>Experiments stored:
<strong>{report['database_summary']['experiment_count']}</strong></p>
{champion_html}
<h2>Overall Leaderboard</h2>
<table>
<thead>
<tr>
<th>Rank</th>
<th>Asset</th>
<th>Timeframe</th>
<th>Strategy</th>
<th>Score</th>
<th>Return</th>
<th>Drawdown</th>
<th>Profit Factor</th>
<th>Trades</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
<p class="small">
Research only. No market request or order submission.
</p>
</body>
</html>
"""


def run() -> None:
    report = build_report()
    text_report = build_text(report)
    html_report = build_html(report)

    JSON_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    TEXT_REPORT_PATH.write_text(
        text_report,
        encoding="utf-8",
    )

    HTML_REPORT_PATH.write_text(
        html_report,
        encoding="utf-8",
    )

    print(text_report)
    print()
    print(
        f"JSON report: {JSON_REPORT_PATH}"
    )
    print(
        f"Text report: {TEXT_REPORT_PATH}"
    )
    print(
        f"HTML report: {HTML_REPORT_PATH}"
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
