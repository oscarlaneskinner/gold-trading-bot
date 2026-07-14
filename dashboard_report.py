"""Generate a read-only HTML dashboard for the GLD paper-trading system."""

from __future__ import annotations

import html
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from broker import create_trading_client, get_account_equity, get_position
from config import REPORTS_DIR, SYMBOL
from trade_memory import DATABASE_PATH, database_summary, recent_trades
from trade_review import build_review


HTML_PATH = REPORTS_DIR / "gld_dashboard.html"
JSON_PATH = REPORTS_DIR / "gld_dashboard.json"


def safe(value: Any) -> str:
    if value is None:
        return "—"
    return html.escape(str(value))


def money(value: Any) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2%}"


def load_recent_decisions(limit: int = 20) -> list[dict[str, Any]]:
    if not Path(DATABASE_PATH).exists():
        return []

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                timestamp_utc,
                market_timestamp,
                symbol,
                model_name,
                model_version,
                prediction,
                probability_up,
                price,
                action,
                reason,
                position_percent,
                notional,
                order_id
            FROM decisions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def account_snapshot(client) -> dict[str, Any]:
    account = client.get_account()
    position = get_position(client, SYMBOL)

    result = {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "position": None,
    }

    if position is not None:
        result["position"] = {
            "symbol": position.symbol,
            "quantity": float(position.qty),
            "market_value": float(position.market_value),
            "average_entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "unrealized_profit_loss": float(position.unrealized_pl),
            "unrealized_return": float(position.unrealized_plpc),
        }

    return result


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_html = "".join(
        f"<th>{html.escape(header)}</th>"
        for header in headers
    )

    body_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{safe(cell)}</td>"
            for cell in row
        )
        + "</tr>"
        for row in rows
    )

    if not rows:
        body_html = (
            f'<tr><td colspan="{len(headers)}">'
            "No records available."
            "</td></tr>"
        )

    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def build_html(payload: dict[str, Any]) -> str:
    account = payload["account"]
    position = account["position"]
    performance = payload["review"]["performance_summary"]
    decisions = payload["recent_decisions"]
    trades = payload["review"]["trades"]

    decision_rows = [
        [
            row["timestamp_utc"],
            "UP" if row["prediction"] == 1 else "DOWN",
            percent(row["probability_up"]),
            money(row["price"]),
            row["action"],
            row["reason"],
        ]
        for row in decisions
    ]

    trade_rows = [
        [
            row["trade_id"],
            row["status"],
            row["grade"],
            money(row["entry_price"]),
            money(row["current_price"] or row["exit_price"]),
            percent(
                row["unrealized_return"]
                if row["status"] == "OPEN"
                else row["realized_return"]
            ),
            money(
                row["unrealized_profit_loss"]
                if row["status"] == "OPEN"
                else row["realized_profit_loss"]
            ),
        ]
        for row in trades
    ]

    position_html = (
        """
        <div class="card">
          <h3>Open GLD Position</h3>
          <p>No open GLD position.</p>
        </div>
        """
        if position is None
        else f"""
        <div class="card">
          <h3>Open GLD Position</h3>
          <div class="metrics">
            <div><span>Quantity</span><strong>{position["quantity"]:.6f}</strong></div>
            <div><span>Market value</span><strong>{money(position["market_value"])}</strong></div>
            <div><span>Average entry</span><strong>{money(position["average_entry_price"])}</strong></div>
            <div><span>Current price</span><strong>{money(position["current_price"])}</strong></div>
            <div><span>Unrealized P/L</span><strong>{money(position["unrealized_profit_loss"])}</strong></div>
            <div><span>Unrealized return</span><strong>{percent(position["unrealized_return"])}</strong></div>
          </div>
        </div>
        """
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLD AI Trading Dashboard</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  background: #f4f6f8;
  color: #1f2937;
}}
.container {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}}
h1, h2, h3 {{ margin-top: 0; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}}
.card {{
  background: white;
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 2px 10px rgba(0,0,0,.06);
  margin-bottom: 18px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}}
.metrics div {{
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
}}
.metrics span {{
  display: block;
  font-size: 12px;
  color: #64748b;
}}
.metrics strong {{
  display: block;
  margin-top: 4px;
  font-size: 20px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}}
th, td {{
  text-align: left;
  padding: 10px;
  border-bottom: 1px solid #e5e7eb;
}}
th {{ background: #f8fafc; }}
.small {{ color: #64748b; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>GLD AI Trading Dashboard</h1>
  <p class="small">Generated {safe(payload["generated_at_utc"])} · Paper trading · Read-only</p>

  <div class="grid">
    <div class="card"><h3>Account equity</h3><strong>{money(account["equity"])}</strong></div>
    <div class="card"><h3>Buying power</h3><strong>{money(account["buying_power"])}</strong></div>
    <div class="card"><h3>Open trades</h3><strong>{performance["open_trades"]}</strong></div>
    <div class="card"><h3>Closed trades</h3><strong>{performance["closed_trades"]}</strong></div>
    <div class="card"><h3>Realized P/L</h3><strong>{money(performance["realized_profit_loss"])}</strong></div>
    <div class="card"><h3>Win rate</h3><strong>{percent(performance["win_rate"])}</strong></div>
  </div>

  {position_html}

  <div class="card">
    <h2>Performance Summary</h2>
    <div class="metrics">
      <div><span>Profit factor</span><strong>{performance["profit_factor"]:.2f}</strong></div>
      <div><span>Average return</span><strong>{percent(performance["average_realized_return"])}</strong></div>
      <div><span>Largest win</span><strong>{percent(performance["largest_win"])}</strong></div>
      <div><span>Largest loss</span><strong>{percent(performance["largest_loss"])}</strong></div>
      <div><span>Open unrealized P/L</span><strong>{money(performance["open_unrealized_profit_loss"])}</strong></div>
      <div><span>Average holding days</span><strong>{performance["average_holding_days"]:.2f}</strong></div>
    </div>
  </div>

  <div class="card">
    <h2>Recent Trades</h2>
    {render_table(
        ["ID", "Status", "Grade", "Entry", "Current / Exit", "Return", "P/L"],
        trade_rows
    )}
  </div>

  <div class="card">
    <h2>Recent AI Decisions</h2>
    {render_table(
        ["Time", "Prediction", "Confidence", "Price", "Action", "Reason"],
        decision_rows
    )}
  </div>
</div>
</body>
</html>
"""


def run() -> None:
    client = create_trading_client()
    account = account_snapshot(client)

    current_price = (
        account["position"]["current_price"]
        if account["position"] is not None
        else None
    )

    review = build_review(
        current_price=current_price,
    )

    payload = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "symbol": SYMBOL,
        "account": account,
        "memory_summary": database_summary(),
        "review": review,
        "recent_decisions": load_recent_decisions(),
        "recent_trades": recent_trades(limit=20),
        "order_submitted": False,
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    HTML_PATH.write_text(
        build_html(payload),
        encoding="utf-8",
    )

    print("GLD dashboard generated")
    print(f"HTML report: {HTML_PATH}")
    print(f"JSON report: {JSON_PATH}")
    print(
        json.dumps(
            {
                "account_equity": account["equity"],
                "open_position": (
                    account["position"] is not None
                ),
                "decision_count": payload[
                    "memory_summary"
                ]["decision_count"],
                "trade_count": payload[
                    "memory_summary"
                ]["trade_count"],
                "order_submitted": False,
            },
            indent=2,
        )
    )
    print("No order was submitted.")


if __name__ == "__main__":
    run()
