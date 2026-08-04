"""
Position Manager v2

Research-only shadow position lifecycle manager.

Default behavior is SAFE PREVIEW MODE:
- reads the existing trade ledger,
- reads the latest shadow proposal,
- reads local daily CSV market data,
- detects duplicate proposals,
- evaluates stop loss, profit target, and maximum holding period,
- calculates unrealized and realized P/L,
- writes a preview report,
- does NOT change the ledger unless --apply is supplied,
- never creates a broker client,
- never submits an order.

Examples
--------
Preview only:
    python position_manager_v2.py

Apply approved shadow-ledger changes:
    python position_manager_v2.py --apply
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

REPORTS_DIRECTORY = Path("reports")
PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"
POSITION_MANAGER_DIRECTORY = REPORTS_DIRECTORY / "position_manager"
POSITION_MANAGER_HISTORY_DIRECTORY = POSITION_MANAGER_DIRECTORY / "history"
SHADOW_OBSERVATION_DIRECTORY = REPORTS_DIRECTORY / "shadow_observation"
DATA_DIRECTORY = Path("data")

TRADE_LEDGER_PATH = PERFORMANCE_DIRECTORY / "trade_ledger.json"
LATEST_SHADOW_SUMMARY_PATH = (
    SHADOW_OBSERVATION_DIRECTORY / "latest_summary.json"
)

LATEST_REPORT_PATH = POSITION_MANAGER_DIRECTORY / "latest_summary.json"
LATEST_TEXT_PATH = POSITION_MANAGER_DIRECTORY / "latest_summary.txt"
OPEN_POSITIONS_CSV_PATH = POSITION_MANAGER_DIRECTORY / "open_positions.csv"
CLOSED_TODAY_CSV_PATH = POSITION_MANAGER_DIRECTORY / "closed_today.csv"

STARTING_CAPITAL = 2000.0
DEFAULT_MAXIMUM_HOLDING_DAYS = 20

POSITION_MANAGER_DIRECTORY.mkdir(parents=True, exist_ok=True)
POSITION_MANAGER_HISTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass
class ManagedTrade:
    trade_id: str
    role: str
    strategy_name: str
    symbol: str
    side: str
    entry_date: str
    entry_price: float
    proposed_dollars: float
    shares: float
    stop_price: Optional[float]
    target_price: Optional[float]
    maximum_holding_days: int
    status: str = "OPEN"
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_dollars: float = 0.0
    pnl_percent: float = 0.0
    holding_days: int = 0
    current_price: Optional[float] = None
    unrealized_pnl_dollars: float = 0.0
    unrealized_pnl_percent: float = 0.0
    last_valued_date: Optional[str] = None


@dataclass
class MarketBar:
    symbol: str
    bar_date: str
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read JSON file {path}: {error}") from error


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()

    if side in {"BUY", "LONG"}:
        return "LONG"

    if side in {"SELL", "SHORT"}:
        return "SHORT"

    return side or "LONG"


def parse_iso_date(value: str) -> date:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).date()


def holding_days(entry_date: str, report_date: str) -> int:
    return max(
        (parse_iso_date(report_date) - parse_iso_date(entry_date)).days,
        0,
    )


# ---------------------------------------------------------------------
# Ledger loading and migration
# ---------------------------------------------------------------------

def managed_trade_from_row(row: dict[str, Any]) -> ManagedTrade:
    """
    Load both the older v1 ledger schema and the extended v2 schema.
    """
    return ManagedTrade(
        trade_id=str(row.get("trade_id", "")),
        role=str(row.get("role", "UNKNOWN")).upper(),
        strategy_name=str(row.get("strategy_name", "UNKNOWN")),
        symbol=str(row.get("symbol", "")).upper(),
        side=normalize_side(row.get("side", "LONG")),
        entry_date=str(row.get("entry_date", "")),
        entry_price=safe_float(row.get("entry_price")),
        proposed_dollars=safe_float(row.get("proposed_dollars")),
        shares=safe_float(row.get("shares")),
        stop_price=(
            safe_float(row.get("stop_price"))
            if row.get("stop_price") is not None
            else None
        ),
        target_price=(
            safe_float(row.get("target_price"))
            if row.get("target_price") is not None
            else None
        ),
        maximum_holding_days=safe_int(
            row.get("maximum_holding_days"),
            DEFAULT_MAXIMUM_HOLDING_DAYS,
        ),
        status=str(row.get("status", "OPEN")).upper(),
        exit_date=row.get("exit_date"),
        exit_price=(
            safe_float(row.get("exit_price"))
            if row.get("exit_price") is not None
            else None
        ),
        exit_reason=row.get("exit_reason"),
        pnl_dollars=safe_float(row.get("pnl_dollars")),
        pnl_percent=safe_float(row.get("pnl_percent")),
        holding_days=safe_int(row.get("holding_days")),
        current_price=(
            safe_float(row.get("current_price"))
            if row.get("current_price") is not None
            else None
        ),
        unrealized_pnl_dollars=safe_float(
            row.get("unrealized_pnl_dollars")
        ),
        unrealized_pnl_percent=safe_float(
            row.get("unrealized_pnl_percent")
        ),
        last_valued_date=row.get("last_valued_date"),
    )


def load_ledger() -> list[ManagedTrade]:
    raw = load_json(TRADE_LEDGER_PATH, [])

    if not isinstance(raw, list):
        raise RuntimeError("Trade ledger must contain a JSON list.")

    return [
        managed_trade_from_row(row)
        for row in raw
        if isinstance(row, dict)
    ]


def ledger_payload(trades: list[ManagedTrade]) -> list[dict[str, Any]]:
    return [asdict(trade) for trade in trades]


# ---------------------------------------------------------------------
# Market-data loading
# ---------------------------------------------------------------------

def load_latest_bar(symbol: str) -> Optional[MarketBar]:
    path = DATA_DIRECTORY / f"{symbol.upper()}_1D.csv"

    if not path.exists():
        return None

    frame = pd.read_csv(path)

    rename_map: dict[str, str] = {}

    for column in frame.columns:
        normalized = column.strip().lower()

        if normalized in {"date", "datetime", "timestamp", "time"}:
            rename_map[column] = "date"
        elif normalized in {"open", "high", "low", "close"}:
            rename_map[column] = normalized

    frame = frame.rename(columns=rename_map)

    required = {"open", "high", "low", "close"}

    if not required.issubset(frame.columns):
        return None

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(subset=list(required))

    if frame.empty:
        return None

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values("date")
        row = frame.iloc[-1]
        bar_date = (
            row["date"].date().isoformat()
            if pd.notna(row["date"])
            else datetime.now().date().isoformat()
        )
    else:
        row = frame.iloc[-1]
        bar_date = datetime.now().date().isoformat()

    return MarketBar(
        symbol=symbol.upper(),
        bar_date=bar_date,
        open=round(float(row["open"]), 4),
        high=round(float(row["high"]), 4),
        low=round(float(row["low"]), 4),
        close=round(float(row["close"]), 4),
    )


# ---------------------------------------------------------------------
# Position calculations
# ---------------------------------------------------------------------

def calculate_pnl(
    side: str,
    entry_price: float,
    current_or_exit_price: float,
    shares: float,
) -> tuple[float, float]:
    if entry_price <= 0:
        return 0.0, 0.0

    if normalize_side(side) == "SHORT":
        pnl_dollars = (
            entry_price - current_or_exit_price
        ) * shares
        pnl_percent = (
            entry_price - current_or_exit_price
        ) / entry_price * 100
    else:
        pnl_dollars = (
            current_or_exit_price - entry_price
        ) * shares
        pnl_percent = (
            current_or_exit_price - entry_price
        ) / entry_price * 100

    return pnl_dollars, pnl_percent


def close_trade(
    trade: ManagedTrade,
    report_date: str,
    exit_price: float,
    reason: str,
) -> None:
    pnl_dollars, pnl_percent = calculate_pnl(
        trade.side,
        trade.entry_price,
        exit_price,
        trade.shares,
    )

    trade.status = "CLOSED"
    trade.exit_date = report_date
    trade.exit_price = round(exit_price, 4)
    trade.exit_reason = reason
    trade.pnl_dollars = round(pnl_dollars, 2)
    trade.pnl_percent = round(pnl_percent, 4)
    trade.current_price = round(exit_price, 4)
    trade.unrealized_pnl_dollars = 0.0
    trade.unrealized_pnl_percent = 0.0
    trade.last_valued_date = report_date


def evaluate_open_trade(
    trade: ManagedTrade,
    bar: MarketBar,
    report_date: str,
) -> Optional[str]:
    """
    Evaluate one open trade.

    Conservative rule:
    If both stop and target are touched in the same daily bar, the stop is
    processed first because intraday sequence is unknown.
    """
    trade.holding_days = holding_days(
        trade.entry_date,
        report_date,
    )

    if trade.side == "SHORT":
        if (
            trade.stop_price is not None
            and bar.high >= trade.stop_price
        ):
            close_trade(
                trade,
                report_date,
                trade.stop_price,
                "STOP_LOSS",
            )
            return "STOP_LOSS"

        if (
            trade.target_price is not None
            and bar.low <= trade.target_price
        ):
            close_trade(
                trade,
                report_date,
                trade.target_price,
                "TAKE_PROFIT",
            )
            return "TAKE_PROFIT"

    else:
        if (
            trade.stop_price is not None
            and bar.low <= trade.stop_price
        ):
            close_trade(
                trade,
                report_date,
                trade.stop_price,
                "STOP_LOSS",
            )
            return "STOP_LOSS"

        if (
            trade.target_price is not None
            and bar.high >= trade.target_price
        ):
            close_trade(
                trade,
                report_date,
                trade.target_price,
                "TAKE_PROFIT",
            )
            return "TAKE_PROFIT"

    if trade.holding_days >= trade.maximum_holding_days:
        close_trade(
            trade,
            report_date,
            bar.close,
            "MAX_HOLDING_PERIOD",
        )
        return "MAX_HOLDING_PERIOD"

    unrealized_dollars, unrealized_percent = calculate_pnl(
        trade.side,
        trade.entry_price,
        bar.close,
        trade.shares,
    )

    trade.current_price = bar.close
    trade.unrealized_pnl_dollars = round(
        unrealized_dollars,
        2,
    )
    trade.unrealized_pnl_percent = round(
        unrealized_percent,
        4,
    )
    trade.last_valued_date = report_date

    return None


# ---------------------------------------------------------------------
# Proposal handling
# ---------------------------------------------------------------------

def open_position_exists(
    trades: list[ManagedTrade],
    symbol: str,
    side: str,
) -> bool:
    normalized_symbol = symbol.strip().upper()
    normalized_side = normalize_side(side)

    return any(
        trade.status == "OPEN"
        and trade.symbol == normalized_symbol
        and normalize_side(trade.side) == normalized_side
        for trade in trades
    )


def build_trade_id(
    report_date: str,
    role: str,
    symbol: str,
    strategy_name: str,
) -> str:
    safe_strategy = (
        strategy_name.strip().upper().replace(" ", "-")
    )

    return (
        f"{report_date}|"
        f"{role.upper()}|"
        f"{symbol.upper()}|"
        f"{safe_strategy}"
    )


def create_trade_from_proposal(
    proposal: dict[str, Any],
    report_date: str,
    scanner_top_long: Optional[dict[str, Any]],
) -> Optional[ManagedTrade]:
    role = str(proposal.get("role", "UNKNOWN")).upper()
    symbol = str(proposal.get("symbol", "")).upper()
    strategy_name = str(
        proposal.get("strategy_name", f"{role}-STRATEGY")
    )
    side = normalize_side(proposal.get("side", "BUY"))

    if not symbol:
        return None

    entry_price = safe_float(proposal.get("entry_price"))

    if entry_price <= 0 and scanner_top_long:
        if (
            str(scanner_top_long.get("symbol", "")).upper()
            == symbol
        ):
            entry_price = safe_float(
                scanner_top_long.get("close")
            )

    if entry_price <= 0:
        bar = load_latest_bar(symbol)
        entry_price = bar.close if bar else 0.0

    proposed_dollars = safe_float(
        proposal.get("proposed_dollars")
    )

    if entry_price <= 0 or proposed_dollars <= 0:
        return None

    stop_price = safe_float(
        proposal.get("suggested_stop"),
        default=0.0,
    )
    target_price = safe_float(
        proposal.get("suggested_target"),
        default=0.0,
    )

    return ManagedTrade(
        trade_id=build_trade_id(
            report_date,
            role,
            symbol,
            strategy_name,
        ),
        role=role,
        strategy_name=strategy_name,
        symbol=symbol,
        side=side,
        entry_date=report_date,
        entry_price=round(entry_price, 4),
        proposed_dollars=round(proposed_dollars, 2),
        shares=round(proposed_dollars / entry_price, 8),
        stop_price=(
            round(stop_price, 4)
            if stop_price > 0
            else None
        ),
        target_price=(
            round(target_price, 4)
            if target_price > 0
            else None
        ),
        maximum_holding_days=safe_int(
            proposal.get("maximum_holding_days"),
            DEFAULT_MAXIMUM_HOLDING_DAYS,
        ),
        status="OPEN",
        current_price=round(entry_price, 4),
        last_valued_date=report_date,
    )


# ---------------------------------------------------------------------
# Portfolio processing
# ---------------------------------------------------------------------

def process_positions(
    trades: list[ManagedTrade],
    shadow_summary: dict[str, Any],
    report_date: str,
) -> dict[str, Any]:
    closed_today: list[ManagedTrade] = []
    missing_market_data: list[str] = []

    for trade in trades:
        if trade.status != "OPEN":
            continue

        bar = load_latest_bar(trade.symbol)

        if bar is None:
            missing_market_data.append(trade.symbol)
            continue

        exit_reason = evaluate_open_trade(
            trade,
            bar,
            report_date,
        )

        if exit_reason:
            closed_today.append(copy.deepcopy(trade))

    proposals = shadow_summary.get(
        "shadow_proposals",
        [],
    )

    if not isinstance(proposals, list):
        proposals = []

    scanner_top_long = shadow_summary.get(
        "scanner_top_long"
    )

    created_today: list[ManagedTrade] = []
    skipped_duplicates: list[dict[str, Any]] = []
    skipped_invalid: list[dict[str, Any]] = []

    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue

        symbol = str(
            proposal.get("symbol", "")
        ).upper()
        side = normalize_side(
            proposal.get("side", "BUY")
        )

        if open_position_exists(
            trades,
            symbol,
            side,
        ):
            skipped_duplicates.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "strategy_name": proposal.get(
                        "strategy_name"
                    ),
                    "reason": (
                        "An open position already exists for "
                        "this symbol and side."
                    ),
                }
            )
            continue

        trade = create_trade_from_proposal(
            proposal,
            report_date,
            scanner_top_long,
        )

        if trade is None:
            skipped_invalid.append(
                {
                    "proposal": proposal,
                    "reason": (
                        "Entry price or proposed dollars "
                        "were unavailable."
                    ),
                }
            )
            continue

        trades.append(trade)
        created_today.append(
            copy.deepcopy(trade)
        )

    open_trades = [
        trade
        for trade in trades
        if trade.status == "OPEN"
    ]
    closed_trades = [
        trade
        for trade in trades
        if trade.status == "CLOSED"
    ]

    realized_pnl = sum(
        trade.pnl_dollars
        for trade in closed_trades
    )
    unrealized_pnl = sum(
        trade.unrealized_pnl_dollars
        for trade in open_trades
    )
    estimated_equity = (
        STARTING_CAPITAL
        + realized_pnl
        + unrealized_pnl
    )

    return {
        "trades": trades,
        "created_today": created_today,
        "closed_today": closed_today,
        "skipped_duplicates": skipped_duplicates,
        "skipped_invalid": skipped_invalid,
        "missing_market_data": sorted(
            set(missing_market_data)
        ),
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "estimated_equity": estimated_equity,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def report_payload(
    result: dict[str, Any],
    report_date: str,
    apply_changes: bool,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "report_date": report_date,
        "mode": (
            "APPLY_SHADOW_LEDGER_CHANGES"
            if apply_changes
            else "PREVIEW_ONLY"
        ),
        "starting_capital": STARTING_CAPITAL,
        "open_position_count": len(
            result["open_trades"]
        ),
        "closed_position_count": len(
            result["closed_trades"]
        ),
        "created_today_count": len(
            result["created_today"]
        ),
        "closed_today_count": len(
            result["closed_today"]
        ),
        "duplicate_proposals_skipped": len(
            result["skipped_duplicates"]
        ),
        "invalid_proposals_skipped": len(
            result["skipped_invalid"]
        ),
        "realized_pnl_dollars": round(
            result["realized_pnl"],
            2,
        ),
        "unrealized_pnl_dollars": round(
            result["unrealized_pnl"],
            2,
        ),
        "estimated_equity": round(
            result["estimated_equity"],
            2,
        ),
        "estimated_total_return_percent": round(
            (
                result["estimated_equity"]
                - STARTING_CAPITAL
            )
            / STARTING_CAPITAL
            * 100,
            4,
        ),
        "missing_market_data": result[
            "missing_market_data"
        ],
        "created_today": [
            asdict(trade)
            for trade in result["created_today"]
        ],
        "closed_today": [
            asdict(trade)
            for trade in result["closed_today"]
        ],
        "skipped_duplicates": result[
            "skipped_duplicates"
        ],
        "skipped_invalid": result[
            "skipped_invalid"
        ],
        "open_positions": [
            asdict(trade)
            for trade in result["open_trades"]
        ],
        "shadow_mode": True,
        "ledger_modified": apply_changes,
        "trading_client_created": False,
        "market_request_made": False,
        "order_submitted": False,
    }


def write_reports(payload: dict[str, Any]) -> None:
    save_json(
        LATEST_REPORT_PATH,
        payload,
    )

    save_json(
        POSITION_MANAGER_HISTORY_DIRECTORY
        / f"{payload['report_date']}.json",
        payload,
    )

    pd.DataFrame(
        payload["open_positions"]
    ).to_csv(
        OPEN_POSITIONS_CSV_PATH,
        index=False,
    )

    pd.DataFrame(
        payload["closed_today"]
    ).to_csv(
        CLOSED_TODAY_CSV_PATH,
        index=False,
    )

    lines = [
        "POSITION MANAGER V2",
        "=" * 24,
        f"Report date: {payload['report_date']}",
        f"Mode: {payload['mode']}",
        (
            "Open positions: "
            f"{payload['open_position_count']}"
        ),
        (
            "Closed positions: "
            f"{payload['closed_position_count']}"
        ),
        (
            "Created today: "
            f"{payload['created_today_count']}"
        ),
        (
            "Closed today: "
            f"{payload['closed_today_count']}"
        ),
        (
            "Duplicate proposals skipped: "
            f"{payload['duplicate_proposals_skipped']}"
        ),
        (
            "Realized P/L: "
            f"${payload['realized_pnl_dollars']:.2f}"
        ),
        (
            "Unrealized P/L: "
            f"${payload['unrealized_pnl_dollars']:.2f}"
        ),
        (
            "Estimated equity: "
            f"${payload['estimated_equity']:.2f}"
        ),
        "",
        "Research shadow mode only.",
        (
            "Ledger modified: "
            f"{payload['ledger_modified']}"
        ),
        "No trading client was created.",
        "No market request was made.",
        "No order was submitted.",
    ]

    LATEST_TEXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manage research-only shadow positions. "
            "Defaults to preview mode."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply approved changes to the shadow trade ledger. "
            "No broker orders are ever submitted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    ledger = load_ledger()
    working_ledger = copy.deepcopy(ledger)

    shadow_summary = load_json(
        LATEST_SHADOW_SUMMARY_PATH,
        {},
    )

    if not isinstance(shadow_summary, dict):
        raise SystemExit(
            "Shadow observation summary must contain a JSON object."
        )

    report_date = str(
        shadow_summary.get(
            "run_date",
            datetime.now().date().isoformat(),
        )
    )

    result = process_positions(
        working_ledger,
        shadow_summary,
        report_date,
    )

    payload = report_payload(
        result,
        report_date,
        arguments.apply,
    )

    write_reports(payload)

    if arguments.apply:
        save_json(
            TRADE_LEDGER_PATH,
            ledger_payload(result["trades"]),
        )

    print("Position Manager v2")
    print(json.dumps(payload, indent=2))
    print(
        "Ledger changes were applied."
        if arguments.apply
        else "Preview only; the ledger was not modified."
    )
    print("No trading client was created.")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()