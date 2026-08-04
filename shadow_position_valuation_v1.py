"""
Shadow Position Valuation v1

Research-only daily mark-to-market reporting.

This module:
- reads the existing shadow trade ledger,
- reads the latest local daily CSV price for each open symbol,
- calculates unrealized P/L for every open position,
- calculates realized P/L from closed positions,
- calculates estimated account equity,
- writes JSON, CSV, and text reports,
- never changes the trade ledger,
- never creates a trading client,
- never submits an order.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPORTS_DIRECTORY = Path("reports")
PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"
VALUATION_DIRECTORY = REPORTS_DIRECTORY / "position_valuation"
DATA_DIRECTORY = Path("data")

TRADE_LEDGER_PATH = PERFORMANCE_DIRECTORY / "trade_ledger.json"
LATEST_JSON_PATH = VALUATION_DIRECTORY / "latest_summary.json"
LATEST_CSV_PATH = VALUATION_DIRECTORY / "open_positions.csv"
LATEST_TEXT_PATH = VALUATION_DIRECTORY / "latest_summary.txt"

STARTING_CAPITAL = 2000.0
VALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)


@dataclass
class PositionValuation:
    trade_id: str
    strategy_name: str
    symbol: str
    side: str
    entry_date: str
    entry_price: float
    current_price: float
    proposed_dollars: float
    shares: float
    stop_price: float | None
    target_price: float | None
    holding_days: int
    unrealized_pnl_dollars: float
    unrealized_pnl_percent: float
    distance_to_stop_percent: float | None
    distance_to_target_percent: float | None
    status: str


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"BUY", "LONG"}:
        return "LONG"
    if side in {"SELL", "SHORT"}:
        return "SHORT"
    return side or "LONG"


def load_latest_close(symbol: str) -> float | None:
    path = DATA_DIRECTORY / f"{symbol.upper()}_1D.csv"
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    close_column = next(
        (
            column
            for column in frame.columns
            if column.strip().lower() == "close"
        ),
        None,
    )

    if close_column is None:
        return None

    closes = pd.to_numeric(
        frame[close_column],
        errors="coerce",
    ).dropna()

    if closes.empty:
        return None

    return float(closes.iloc[-1])


def calculate_unrealized(
    side: str,
    entry_price: float,
    current_price: float,
    shares: float,
) -> tuple[float, float]:
    normalized_side = normalize_side(side)

    if normalized_side == "SHORT":
        pnl_dollars = (entry_price - current_price) * shares
        pnl_percent = (
            (entry_price - current_price) / entry_price * 100
            if entry_price > 0
            else 0.0
        )
    else:
        pnl_dollars = (current_price - entry_price) * shares
        pnl_percent = (
            (current_price - entry_price) / entry_price * 100
            if entry_price > 0
            else 0.0
        )

    return pnl_dollars, pnl_percent


def distance_percent(
    current_price: float,
    level: float | None,
) -> float | None:
    if level is None or current_price <= 0:
        return None
    return (level - current_price) / current_price * 100


def value_open_positions(
    ledger: list[dict[str, Any]],
) -> tuple[list[PositionValuation], list[str]]:
    valuations: list[PositionValuation] = []
    missing_prices: list[str] = []

    for trade in ledger:
        if str(trade.get("status", "")).upper() != "OPEN":
            continue

        symbol = str(trade.get("symbol", "")).upper()
        current_price = load_latest_close(symbol)

        if current_price is None:
            missing_prices.append(symbol)
            continue

        entry_price = safe_float(trade.get("entry_price"))
        shares = safe_float(trade.get("shares"))

        stop_price = (
            safe_float(trade.get("stop_price"))
            if trade.get("stop_price") is not None
            else None
        )
        target_price = (
            safe_float(trade.get("target_price"))
            if trade.get("target_price") is not None
            else None
        )

        pnl_dollars, pnl_percent = calculate_unrealized(
            str(trade.get("side", "LONG")),
            entry_price,
            current_price,
            shares,
        )

        valuations.append(
            PositionValuation(
                trade_id=str(trade.get("trade_id", "")),
                strategy_name=str(trade.get("strategy_name", "")),
                symbol=symbol,
                side=normalize_side(trade.get("side")),
                entry_date=str(trade.get("entry_date", "")),
                entry_price=round(entry_price, 4),
                current_price=round(current_price, 4),
                proposed_dollars=round(
                    safe_float(trade.get("proposed_dollars")),
                    2,
                ),
                shares=round(shares, 8),
                stop_price=(
                    round(stop_price, 4)
                    if stop_price is not None
                    else None
                ),
                target_price=(
                    round(target_price, 4)
                    if target_price is not None
                    else None
                ),
                holding_days=int(trade.get("holding_days", 0) or 0),
                unrealized_pnl_dollars=round(pnl_dollars, 2),
                unrealized_pnl_percent=round(pnl_percent, 4),
                distance_to_stop_percent=(
                    round(distance_percent(current_price, stop_price), 4)
                    if stop_price is not None
                    else None
                ),
                distance_to_target_percent=(
                    round(distance_percent(current_price, target_price), 4)
                    if target_price is not None
                    else None
                ),
                status="OPEN",
            )
        )

    return valuations, sorted(set(missing_prices))


def realized_pnl(ledger: list[dict[str, Any]]) -> float:
    return sum(
        safe_float(trade.get("pnl_dollars"))
        for trade in ledger
        if str(trade.get("status", "")).upper() == "CLOSED"
    )


def write_reports(
    valuations: list[PositionValuation],
    ledger: list[dict[str, Any]],
    missing_prices: list[str],
) -> dict[str, Any]:
    realized = realized_pnl(ledger)
    unrealized = sum(
        position.unrealized_pnl_dollars
        for position in valuations
    )
    estimated_equity = STARTING_CAPITAL + realized + unrealized

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "starting_capital": STARTING_CAPITAL,
        "open_position_count": len(valuations),
        "closed_position_count": sum(
            1
            for trade in ledger
            if str(trade.get("status", "")).upper() == "CLOSED"
        ),
        "realized_pnl_dollars": round(realized, 2),
        "unrealized_pnl_dollars": round(unrealized, 2),
        "estimated_equity": round(estimated_equity, 2),
        "estimated_total_return_percent": round(
            (estimated_equity - STARTING_CAPITAL)
            / STARTING_CAPITAL
            * 100,
            4,
        ),
        "missing_price_symbols": missing_prices,
        "positions": [
            asdict(position)
            for position in valuations
        ],
        "shadow_mode": True,
        "ledger_modified": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    LATEST_JSON_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        [asdict(position) for position in valuations]
    ).to_csv(
        LATEST_CSV_PATH,
        index=False,
    )

    lines = [
        "SHADOW POSITION VALUATION V1",
        "=" * 32,
        f"Generated at: {payload['generated_at']}",
        f"Open positions: {payload['open_position_count']}",
        f"Closed positions: {payload['closed_position_count']}",
        f"Realized P/L: ${payload['realized_pnl_dollars']:.2f}",
        f"Unrealized P/L: ${payload['unrealized_pnl_dollars']:.2f}",
        f"Estimated equity: ${payload['estimated_equity']:.2f}",
        (
            "Estimated total return: "
            f"{payload['estimated_total_return_percent']:.2f}%"
        ),
        "",
    ]

    for position in valuations:
        lines.extend(
            [
                (
                    f"{position.symbol} {position.side} | "
                    f"Entry ${position.entry_price:.4f} | "
                    f"Current ${position.current_price:.4f}"
                ),
                (
                    f"Unrealized P/L: "
                    f"${position.unrealized_pnl_dollars:.2f} "
                    f"({position.unrealized_pnl_percent:.2f}%)"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Research only.",
            "The trade ledger was not modified.",
            "No market request was made.",
            "No order was submitted.",
        ]
    )

    LATEST_TEXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return payload


def main() -> None:
    ledger = load_json(TRADE_LEDGER_PATH, [])

    if not isinstance(ledger, list):
        raise SystemExit("Trade ledger must contain a JSON list.")

    valuations, missing_prices = value_open_positions(ledger)

    payload = write_reports(
        valuations,
        ledger,
        missing_prices,
    )

    print("Shadow Position Valuation v1")
    print(json.dumps(payload, indent=2))
    print("The trade ledger was not modified.")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
