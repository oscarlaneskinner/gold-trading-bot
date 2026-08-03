"""
Shadow Performance Tracker v1

Research-only shadow performance tracking.

This module:
- reads the latest shadow observation summary,
- creates simulated shadow trades from proposals,
- prevents duplicate open positions for the same symbol and side, even if the selected strategy changes,
- updates open trades from local daily CSV market data,
- closes trades at stop, target, or maximum holding period,
- calculates performance statistics,
- writes a persistent ledger and daily reports,
- never creates a trading client,
- never submits an order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

REPORTS_DIRECTORY = Path("reports")
SHADOW_OBSERVATION_DIRECTORY = REPORTS_DIRECTORY / "shadow_observation"
PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"
PERFORMANCE_HISTORY_DIRECTORY = PERFORMANCE_DIRECTORY / "history"
DATA_DIRECTORY = Path("data")

LATEST_SHADOW_SUMMARY_PATH = SHADOW_OBSERVATION_DIRECTORY / "latest_summary.json"
TRADE_LEDGER_PATH = PERFORMANCE_DIRECTORY / "trade_ledger.json"
LATEST_PERFORMANCE_REPORT_PATH = PERFORMANCE_DIRECTORY / "latest_summary.json"
LATEST_PERFORMANCE_TEXT_PATH = PERFORMANCE_DIRECTORY / "latest_summary.txt"
EQUITY_CURVE_PATH = PERFORMANCE_DIRECTORY / "equity_curve.csv"

PERFORMANCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
PERFORMANCE_HISTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
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


@dataclass
class PerformanceSummary:
    generated_at: str
    report_date: str
    open_positions: int
    closed_positions: int
    wins: int
    losses: int
    breakeven_trades: int
    win_rate_percent: float
    total_pnl_dollars: float
    total_return_percent: float
    average_winner_percent: float
    average_loser_percent: float
    profit_factor: float
    expectancy_percent: float
    largest_winner_percent: float
    largest_loser_percent: float
    average_holding_days: float
    maximum_drawdown_percent: float
    starting_capital: float
    ending_equity: float
    new_trades_created: int
    trades_closed_today: int
    duplicate_open_positions_skipped: int
    notes: list[str]
    shadow_mode: bool
    market_request_made: bool
    order_submitted: bool


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        LOGGER.warning("Missing file: %s", path)
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        LOGGER.error("Could not load %s: %s", path, error)
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"BUY", "LONG"}:
        return "LONG"
    if side in {"SELL", "SHORT"}:
        return "SHORT"
    return side or "LONG"


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def date_difference_days(start_date: str, end_date: str) -> int:
    return max((parse_date(end_date).date() - parse_date(start_date).date()).days, 0)


def load_trade_ledger() -> list[ShadowTrade]:
    trades: list[ShadowTrade] = []
    for row in load_json(TRADE_LEDGER_PATH, []):
        try:
            trades.append(ShadowTrade(**row))
        except TypeError as error:
            LOGGER.warning("Skipping incompatible ledger row: %s", error)
    return trades


def save_trade_ledger(trades: list[ShadowTrade]) -> None:
    save_json(TRADE_LEDGER_PATH, [asdict(trade) for trade in trades])


def load_symbol_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIRECTORY / f"{symbol.upper()}_1D.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = column.strip().lower()
        if normalized in {"date", "datetime", "timestamp", "time"}:
            rename_map[column] = "date"
        elif normalized in {"open", "high", "low", "close", "volume"}:
            rename_map[column] = normalized
    frame = frame.rename(columns=rename_map)
    required_columns = {"open", "high", "low", "close"}
    if not required_columns.issubset(frame.columns):
        LOGGER.warning("%s is missing required OHLC columns.", path)
        return pd.DataFrame()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    else:
        frame["date"] = pd.RangeIndex(start=0, stop=len(frame), step=1)
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame.sort_values("date").reset_index(drop=True)


def latest_market_row(symbol: str) -> Optional[pd.Series]:
    frame = load_symbol_data(symbol)
    return None if frame.empty else frame.iloc[-1]


def trade_already_exists(
    trades: list[ShadowTrade],
    trade_id: str,
    strategy_name: str,
    symbol: str,
    side: str,
) -> bool:
    """
    Return True when this proposal would duplicate an existing position.

    Exact trade IDs are always rejected. In addition, only one open position
    is allowed for a given symbol and direction, even if Portfolio Commander
    selects a different strategy on a later day. This prevents accidental
    stacking of multiple AMZN-long positions against the same $2,000 research
    account.
    """
    del strategy_name  # Retained in the signature for compatibility.

    normalized_symbol = symbol.strip().upper()
    normalized_side = normalize_side(side)

    for trade in trades:
        if trade.trade_id == trade_id:
            return True

        if (
            trade.status == "OPEN"
            and trade.symbol.strip().upper() == normalized_symbol
            and normalize_side(trade.side) == normalized_side
        ):
            return True

    return False


def build_trade_id(report_date: str, role: str, symbol: str, strategy_name: str) -> str:
    safe_strategy = strategy_name.strip().upper().replace(" ", "-")
    return f"{report_date}|{role.upper()}|{symbol.upper()}|{safe_strategy}"


def infer_entry_price(proposal: dict[str, Any], scanner_top_long: Optional[dict[str, Any]]) -> float:
    direct_price = safe_float(proposal.get("entry_price"))
    if direct_price > 0:
        return direct_price
    symbol = str(proposal.get("symbol", "")).upper()
    if scanner_top_long and str(scanner_top_long.get("symbol", "")).upper() == symbol:
        scanner_close = safe_float(scanner_top_long.get("close"))
        if scanner_close > 0:
            return scanner_close
    market_row = latest_market_row(symbol)
    return 0.0 if market_row is None else safe_float(market_row.get("close"))


def infer_stop_price(proposal: dict[str, Any], scanner_top_long: Optional[dict[str, Any]]) -> Optional[float]:
    direct_stop = safe_float(proposal.get("suggested_stop"))
    if direct_stop > 0:
        return direct_stop
    symbol = str(proposal.get("symbol", "")).upper()
    if scanner_top_long and str(scanner_top_long.get("symbol", "")).upper() == symbol:
        scanner_stop = safe_float(scanner_top_long.get("suggested_stop"))
        if scanner_stop > 0:
            return scanner_stop
    return None


def infer_target_price(proposal: dict[str, Any], scanner_top_long: Optional[dict[str, Any]]) -> Optional[float]:
    direct_target = safe_float(proposal.get("suggested_target"))
    if direct_target > 0:
        return direct_target
    symbol = str(proposal.get("symbol", "")).upper()
    if scanner_top_long and str(scanner_top_long.get("symbol", "")).upper() == symbol:
        scanner_target = safe_float(scanner_top_long.get("suggested_target"))
        if scanner_target > 0:
            return scanner_target
    return None


def create_new_shadow_trades(
    shadow_summary: dict[str, Any],
    trades: list[ShadowTrade],
) -> tuple[int, int]:
    report_date = str(shadow_summary.get("run_date", datetime.now().date().isoformat()))
    proposals = shadow_summary.get("shadow_proposals", [])
    scanner_top_long = shadow_summary.get("scanner_top_long")
    created_count = 0
    skipped_duplicate_count = 0
    for proposal in proposals:
        role = str(proposal.get("role", "UNKNOWN")).upper()
        symbol = str(proposal.get("symbol", "")).upper()
        strategy_name = str(proposal.get("strategy_name", f"{role}-STRATEGY"))
        side = normalize_side(proposal.get("side", "BUY"))
        if not symbol:
            continue
        trade_id = build_trade_id(report_date, role, symbol, strategy_name)
        if trade_already_exists(trades, trade_id, strategy_name, symbol, side):
            LOGGER.info("Skipping duplicate open trade for %s %s %s.", strategy_name, symbol, side)
            skipped_duplicate_count += 1
            continue
        entry_price = infer_entry_price(proposal, scanner_top_long)
        proposed_dollars = safe_float(proposal.get("proposed_dollars"))
        if entry_price <= 0 or proposed_dollars <= 0:
            continue
        trades.append(
            ShadowTrade(
                trade_id=trade_id,
                role=role,
                strategy_name=strategy_name,
                symbol=symbol,
                side=side,
                entry_date=report_date,
                entry_price=round(entry_price, 4),
                proposed_dollars=round(proposed_dollars, 2),
                shares=round(proposed_dollars / entry_price, 8),
                stop_price=infer_stop_price(proposal, scanner_top_long),
                target_price=infer_target_price(proposal, scanner_top_long),
                maximum_holding_days=safe_int(proposal.get("maximum_holding_days", 20), 20),
            )
        )
        created_count += 1
    return created_count, skipped_duplicate_count


def calculate_trade_result(trade: ShadowTrade, exit_price: float) -> tuple[float, float]:
    if trade.side == "SHORT":
        pnl_dollars = (trade.entry_price - exit_price) * trade.shares
        pnl_percent = (trade.entry_price - exit_price) / trade.entry_price * 100
    else:
        pnl_dollars = (exit_price - trade.entry_price) * trade.shares
        pnl_percent = (exit_price - trade.entry_price) / trade.entry_price * 100
    return pnl_dollars, pnl_percent


def close_trade(trade: ShadowTrade, report_date: str, exit_price: float, exit_reason: str) -> None:
    pnl_dollars, pnl_percent = calculate_trade_result(trade, exit_price)
    trade.status = "CLOSED"
    trade.exit_date = report_date
    trade.exit_price = round(exit_price, 4)
    trade.exit_reason = exit_reason
    trade.pnl_dollars = round(pnl_dollars, 2)
    trade.pnl_percent = round(pnl_percent, 4)


def update_open_trade(trade: ShadowTrade, report_date: str) -> bool:
    market_row = latest_market_row(trade.symbol)
    if market_row is None:
        return False
    high = safe_float(market_row.get("high"))
    low = safe_float(market_row.get("low"))
    close = safe_float(market_row.get("close"))
    trade.holding_days = date_difference_days(trade.entry_date, report_date)
    if trade.side == "SHORT":
        if trade.stop_price is not None and high >= trade.stop_price:
            close_trade(trade, report_date, trade.stop_price, "STOP_LOSS")
            return True
        if trade.target_price is not None and low <= trade.target_price:
            close_trade(trade, report_date, trade.target_price, "TAKE_PROFIT")
            return True
    else:
        if trade.stop_price is not None and low <= trade.stop_price:
            close_trade(trade, report_date, trade.stop_price, "STOP_LOSS")
            return True
        if trade.target_price is not None and high >= trade.target_price:
            close_trade(trade, report_date, trade.target_price, "TAKE_PROFIT")
            return True
    if trade.holding_days >= trade.maximum_holding_days:
        close_trade(trade, report_date, close, "MAX_HOLDING_PERIOD")
        return True
    return False


def update_open_trades(trades: list[ShadowTrade], report_date: str) -> int:
    closed_today = 0
    for trade in trades:
        if trade.status == "OPEN" and update_open_trade(trade, report_date):
            closed_today += 1
    return closed_today


def calculate_maximum_drawdown(equity_values: list[float]) -> float:
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    maximum_drawdown = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak > 0:
            maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak * 100)
    return maximum_drawdown


def build_equity_curve(trades: list[ShadowTrade], starting_capital: float) -> pd.DataFrame:
    closed_trades = sorted(
        [trade for trade in trades if trade.status == "CLOSED" and trade.exit_date is not None],
        key=lambda trade: (trade.exit_date or "", trade.trade_id),
    )
    equity = starting_capital
    rows: list[dict[str, Any]] = [
        {"date": "START", "trade_id": "", "symbol": "", "pnl_dollars": 0.0, "equity": round(equity, 2)}
    ]
    for trade in closed_trades:
        equity += trade.pnl_dollars
        rows.append(
            {
                "date": trade.exit_date,
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "pnl_dollars": trade.pnl_dollars,
                "equity": round(equity, 2),
            }
        )
    return pd.DataFrame(rows)


def calculate_performance_summary(
    trades: list[ShadowTrade],
    report_date: str,
    starting_capital: float,
    new_trades_created: int,
    trades_closed_today: int,
    duplicate_open_positions_skipped: int,
) -> PerformanceSummary:
    closed = [trade for trade in trades if trade.status == "CLOSED"]
    open_trades = [trade for trade in trades if trade.status == "OPEN"]
    wins = [trade for trade in closed if trade.pnl_dollars > 0]
    losses = [trade for trade in closed if trade.pnl_dollars < 0]
    breakeven = [trade for trade in closed if trade.pnl_dollars == 0]
    closed_count = len(closed)
    win_rate = len(wins) / closed_count * 100 if closed_count else 0.0
    total_pnl = sum(trade.pnl_dollars for trade in closed)
    total_return = total_pnl / starting_capital * 100 if starting_capital > 0 else 0.0
    average_winner = sum(trade.pnl_percent for trade in wins) / len(wins) if wins else 0.0
    average_loser = sum(trade.pnl_percent for trade in losses) / len(losses) if losses else 0.0
    gross_profit = sum(trade.pnl_dollars for trade in wins)
    gross_loss = abs(sum(trade.pnl_dollars for trade in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    expectancy = sum(trade.pnl_percent for trade in closed) / closed_count if closed_count else 0.0
    largest_winner = max((trade.pnl_percent for trade in closed), default=0.0)
    largest_loser = min((trade.pnl_percent for trade in closed), default=0.0)
    average_holding_days = sum(trade.holding_days for trade in closed) / closed_count if closed_count else 0.0
    equity_curve = build_equity_curve(trades, starting_capital)
    equity_curve.to_csv(EQUITY_CURVE_PATH, index=False)
    equity_values = equity_curve["equity"].astype(float).tolist()
    maximum_drawdown = calculate_maximum_drawdown(equity_values)
    ending_equity = equity_values[-1] if equity_values else starting_capital
    notes = [
        "Research-only shadow performance tracking.",
        "No paper or live orders were submitted.",
    ]
    if closed_count == 0:
        notes.append("No trades have closed yet, so performance statistics are preliminary.")
    if duplicate_open_positions_skipped > 0:
        notes.append(f"Skipped {duplicate_open_positions_skipped} duplicate open position proposal(s).")
    return PerformanceSummary(
        generated_at=datetime.now().astimezone().isoformat(),
        report_date=report_date,
        open_positions=len(open_trades),
        closed_positions=closed_count,
        wins=len(wins),
        losses=len(losses),
        breakeven_trades=len(breakeven),
        win_rate_percent=round(win_rate, 4),
        total_pnl_dollars=round(total_pnl, 2),
        total_return_percent=round(total_return, 4),
        average_winner_percent=round(average_winner, 4),
        average_loser_percent=round(average_loser, 4),
        profit_factor=round(profit_factor, 4),
        expectancy_percent=round(expectancy, 4),
        largest_winner_percent=round(largest_winner, 4),
        largest_loser_percent=round(largest_loser, 4),
        average_holding_days=round(average_holding_days, 2),
        maximum_drawdown_percent=round(maximum_drawdown, 4),
        starting_capital=round(starting_capital, 2),
        ending_equity=round(ending_equity, 2),
        new_trades_created=new_trades_created,
        trades_closed_today=trades_closed_today,
        duplicate_open_positions_skipped=duplicate_open_positions_skipped,
        notes=notes,
        shadow_mode=True,
        market_request_made=False,
        order_submitted=False,
    )


def build_text_report(summary: PerformanceSummary) -> str:
    return "\n".join(
        [
            "SHADOW PERFORMANCE TRACKER V1",
            "=" * 30,
            f"Report date: {summary.report_date}",
            f"Open positions: {summary.open_positions}",
            f"Closed positions: {summary.closed_positions}",
            f"Wins: {summary.wins}",
            f"Losses: {summary.losses}",
            f"Win rate: {summary.win_rate_percent:.2f}%",
            f"Total P/L: ${summary.total_pnl_dollars:.2f}",
            f"Total return: {summary.total_return_percent:.2f}%",
            f"Profit factor: {summary.profit_factor:.2f}",
            f"Expectancy: {summary.expectancy_percent:.2f}%",
            f"Maximum drawdown: {summary.maximum_drawdown_percent:.2f}%",
            f"Ending equity: ${summary.ending_equity:.2f}",
            f"New trades created: {summary.new_trades_created}",
            f"Trades closed today: {summary.trades_closed_today}",
            f"Duplicate open proposals skipped: {summary.duplicate_open_positions_skipped}",
            "",
            "Shadow mode only.",
            "No order was submitted.",
        ]
    )


def main() -> None:
    LOGGER.info("Starting Shadow Performance Tracker v1.")
    shadow_summary = load_json(LATEST_SHADOW_SUMMARY_PATH, {})
    if not shadow_summary:
        raise SystemExit("No shadow observation summary was found.")
    report_date = str(shadow_summary.get("run_date", datetime.now().date().isoformat()))
    starting_capital = safe_float(shadow_summary.get("starting_capital", 2000.0), 2000.0)
    trades = load_trade_ledger()
    new_trades_created, duplicate_open_positions_skipped = create_new_shadow_trades(shadow_summary, trades)
    trades_closed_today = update_open_trades(trades, report_date)
    summary = calculate_performance_summary(
        trades=trades,
        report_date=report_date,
        starting_capital=starting_capital,
        new_trades_created=new_trades_created,
        trades_closed_today=trades_closed_today,
        duplicate_open_positions_skipped=duplicate_open_positions_skipped,
    )
    save_trade_ledger(trades)
    summary_payload = asdict(summary)
    save_json(LATEST_PERFORMANCE_REPORT_PATH, summary_payload)
    save_json(PERFORMANCE_HISTORY_DIRECTORY / f"{report_date}.json", summary_payload)
    LATEST_PERFORMANCE_TEXT_PATH.write_text(build_text_report(summary), encoding="utf-8")
    print("Shadow Performance Tracker v1")
    print(json.dumps(summary_payload, indent=2))
    print("Shadow mode only.")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()