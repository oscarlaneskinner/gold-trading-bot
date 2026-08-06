"""
Performance Analytics v1

Research-only analytics for the gold-trading-bot project.

This module:
- reads the shadow trade ledger,
- calculates portfolio, strategy, symbol, side, and time-based analytics,
- creates equity, drawdown, monthly, weekly, rolling-win-rate,
  rolling-expectancy, and rolling-Sharpe datasets,
- writes JSON, CSV, and text reports,
- never changes the trade ledger,
- never creates a broker client,
- never submits an order.

Run:
    python performance_analytics_v1.py

Optional custom starting capital:
    python performance_analytics_v1.py --starting-capital 2000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"
ANALYTICS_DIRECTORY = REPORTS_DIRECTORY / "performance_analytics"
HISTORY_DIRECTORY = ANALYTICS_DIRECTORY / "history"

TRADE_LEDGER_PATH = PERFORMANCE_DIRECTORY / "trade_ledger.json"

LATEST_SUMMARY_PATH = ANALYTICS_DIRECTORY / "latest_summary.json"
LATEST_TEXT_PATH = ANALYTICS_DIRECTORY / "latest_summary.txt"
TRADE_ANALYTICS_PATH = ANALYTICS_DIRECTORY / "trade_analytics.csv"
STRATEGY_ANALYTICS_PATH = ANALYTICS_DIRECTORY / "strategy_analytics.csv"
SYMBOL_ANALYTICS_PATH = ANALYTICS_DIRECTORY / "symbol_analytics.csv"
SIDE_ANALYTICS_PATH = ANALYTICS_DIRECTORY / "side_analytics.csv"
MONTHLY_RETURNS_PATH = ANALYTICS_DIRECTORY / "monthly_returns.csv"
WEEKLY_RETURNS_PATH = ANALYTICS_DIRECTORY / "weekly_returns.csv"
EQUITY_CURVE_PATH = ANALYTICS_DIRECTORY / "equity_curve.csv"
DRAWDOWN_PATH = ANALYTICS_DIRECTORY / "drawdown_curve.csv"
ROLLING_METRICS_PATH = ANALYTICS_DIRECTORY / "rolling_metrics.csv"
OPEN_POSITIONS_PATH = ANALYTICS_DIRECTORY / "open_positions.csv"

ANALYTICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
HISTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class AnalyticsSummary:
    generated_at: str
    starting_capital: float
    ending_equity: float
    open_position_count: int
    closed_position_count: int
    wins: int
    losses: int
    breakeven_trades: int
    win_rate_percent: float
    total_realized_pnl_dollars: float
    total_realized_return_percent: float
    total_unrealized_pnl_dollars: float
    estimated_equity_with_unrealized: float
    estimated_total_return_percent: float
    gross_profit_dollars: float
    gross_loss_dollars: float
    profit_factor: float
    average_trade_pnl_dollars: float
    average_trade_return_percent: float
    average_winner_dollars: float
    average_winner_percent: float
    average_loser_dollars: float
    average_loser_percent: float
    expectancy_dollars: float
    expectancy_percent: float
    largest_winner_dollars: float
    largest_winner_percent: float
    largest_loser_dollars: float
    largest_loser_percent: float
    average_holding_days: float
    median_holding_days: float
    maximum_holding_days: int
    maximum_drawdown_dollars: float
    maximum_drawdown_percent: float
    recovery_factor: float
    payoff_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    best_strategy: str | None
    worst_strategy: str | None
    best_symbol: str | None
    worst_symbol: str | None
    best_trade_id: str | None
    worst_trade_id: str | None
    first_exit_date: str | None
    last_exit_date: str | None
    analytics_status: str
    notes: list[str]
    shadow_mode: bool
    ledger_modified: bool
    trading_client_created: bool
    market_request_made: bool
    order_submitted: bool


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


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

    return side or "UNKNOWN"


def annualized_sharpe(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()

    if len(clean) < 2:
        return 0.0

    standard_deviation = float(clean.std(ddof=1))

    if standard_deviation == 0 or math.isnan(standard_deviation):
        return 0.0

    return float(
        clean.mean()
        / standard_deviation
        * math.sqrt(252)
    )


def annualized_sortino(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()

    if len(clean) < 2:
        return 0.0

    downside = clean[clean < 0]

    if downside.empty:
        return 0.0

    downside_deviation = float(
        np.sqrt(
            np.mean(
                np.square(downside)
            )
        )
    )

    if downside_deviation == 0 or math.isnan(downside_deviation):
        return 0.0

    return float(
        clean.mean()
        / downside_deviation
        * math.sqrt(252)
    )


# ============================================================
# LEDGER NORMALIZATION
# ============================================================

def ledger_to_frame(raw_ledger: Any) -> pd.DataFrame:
    if not isinstance(raw_ledger, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for item in raw_ledger:
        if not isinstance(item, dict):
            continue

        row = dict(item)

        row["trade_id"] = str(
            row.get("trade_id", "")
        )
        row["strategy_name"] = str(
            row.get("strategy_name", "UNKNOWN")
        )
        row["symbol"] = str(
            row.get("symbol", "UNKNOWN")
        ).upper()
        row["side"] = normalize_side(
            row.get("side")
        )
        row["status"] = str(
            row.get("status", "UNKNOWN")
        ).upper()

        for column in [
            "entry_price",
            "exit_price",
            "proposed_dollars",
            "shares",
            "stop_price",
            "target_price",
            "pnl_dollars",
            "pnl_percent",
            "current_price",
            "unrealized_pnl_dollars",
            "unrealized_pnl_percent",
        ]:
            row[column] = safe_float(
                row.get(column)
            )

        row["holding_days"] = safe_int(
            row.get("holding_days")
        )

        row["entry_date"] = pd.to_datetime(
            row.get("entry_date"),
            errors="coerce",
        )
        row["exit_date"] = pd.to_datetime(
            row.get("exit_date"),
            errors="coerce",
        )
        row["last_valued_date"] = pd.to_datetime(
            row.get("last_valued_date"),
            errors="coerce",
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# TRADE-LEVEL ANALYTICS
# ============================================================

def enrich_closed_trades(
    closed: pd.DataFrame,
) -> pd.DataFrame:
    if closed.empty:
        return closed.copy()

    result = closed.copy()

    result["is_win"] = (
        result["pnl_dollars"] > 0
    )
    result["is_loss"] = (
        result["pnl_dollars"] < 0
    )
    result["is_breakeven"] = (
        result["pnl_dollars"] == 0
    )

    result["exit_year"] = (
        result["exit_date"].dt.year
    )
    result["exit_month"] = (
        result["exit_date"]
        .dt.to_period("M")
        .astype(str)
    )
    result["exit_week"] = (
        result["exit_date"]
        .dt.to_period("W")
        .astype(str)
    )
    result["exit_weekday"] = (
        result["exit_date"].dt.day_name()
    )

    result = result.sort_values(
        ["exit_date", "trade_id"],
        na_position="last",
    ).reset_index(drop=True)

    return result


def group_analytics(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    columns = [
        group_column,
        "trade_count",
        "wins",
        "losses",
        "breakeven_trades",
        "win_rate_percent",
        "total_pnl_dollars",
        "average_pnl_dollars",
        "average_return_percent",
        "gross_profit_dollars",
        "gross_loss_dollars",
        "profit_factor",
        "expectancy_dollars",
        "expectancy_percent",
        "average_holding_days",
        "largest_winner_dollars",
        "largest_loser_dollars",
    ]

    if frame.empty or group_column not in frame.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    for group_name, group in frame.groupby(
        group_column,
        dropna=False,
    ):
        wins = group[group["pnl_dollars"] > 0]
        losses = group[group["pnl_dollars"] < 0]
        breakeven = group[group["pnl_dollars"] == 0]

        gross_profit = float(
            wins["pnl_dollars"].sum()
        )
        gross_loss = abs(
            float(losses["pnl_dollars"].sum())
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        elif gross_profit > 0:
            profit_factor = 99.0
        else:
            profit_factor = 0.0

        trade_count = len(group)

        rows.append(
            {
                group_column: (
                    "UNKNOWN"
                    if pd.isna(group_name)
                    else str(group_name)
                ),
                "trade_count": trade_count,
                "wins": len(wins),
                "losses": len(losses),
                "breakeven_trades": len(breakeven),
                "win_rate_percent": round(
                    len(wins) / trade_count * 100
                    if trade_count
                    else 0.0,
                    4,
                ),
                "total_pnl_dollars": round(
                    float(
                        group["pnl_dollars"].sum()
                    ),
                    2,
                ),
                "average_pnl_dollars": round(
                    float(
                        group["pnl_dollars"].mean()
                    ),
                    2,
                ),
                "average_return_percent": round(
                    float(
                        group["pnl_percent"].mean()
                    ),
                    4,
                ),
                "gross_profit_dollars": round(
                    gross_profit,
                    2,
                ),
                "gross_loss_dollars": round(
                    gross_loss,
                    2,
                ),
                "profit_factor": round(
                    profit_factor,
                    4,
                ),
                "expectancy_dollars": round(
                    float(
                        group["pnl_dollars"].mean()
                    ),
                    2,
                ),
                "expectancy_percent": round(
                    float(
                        group["pnl_percent"].mean()
                    ),
                    4,
                ),
                "average_holding_days": round(
                    float(
                        group["holding_days"].mean()
                    ),
                    2,
                ),
                "largest_winner_dollars": round(
                    float(
                        group["pnl_dollars"].max()
                    ),
                    2,
                ),
                "largest_loser_dollars": round(
                    float(
                        group["pnl_dollars"].min()
                    ),
                    2,
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["total_pnl_dollars", "profit_factor"],
        ascending=[False, False],
    ).reset_index(drop=True)


# ============================================================
# EQUITY, DRAWDOWN, AND TIME-BASED ANALYTICS
# ============================================================

def build_equity_curve(
    closed: pd.DataFrame,
    starting_capital: float,
) -> pd.DataFrame:
    rows = [
        {
            "sequence": 0,
            "date": None,
            "trade_id": "START",
            "symbol": "",
            "strategy_name": "",
            "pnl_dollars": 0.0,
            "equity": round(
                starting_capital,
                2,
            ),
            "return_from_start_percent": 0.0,
        }
    ]

    equity = starting_capital

    if not closed.empty:
        for index, trade in closed.reset_index(
            drop=True
        ).iterrows():
            pnl = safe_float(
                trade.get("pnl_dollars")
            )
            equity += pnl

            rows.append(
                {
                    "sequence": index + 1,
                    "date": (
                        trade["exit_date"].date().isoformat()
                        if pd.notna(
                            trade.get("exit_date")
                        )
                        else None
                    ),
                    "trade_id": trade.get(
                        "trade_id",
                        "",
                    ),
                    "symbol": trade.get(
                        "symbol",
                        "",
                    ),
                    "strategy_name": trade.get(
                        "strategy_name",
                        "",
                    ),
                    "pnl_dollars": round(
                        pnl,
                        2,
                    ),
                    "equity": round(
                        equity,
                        2,
                    ),
                    "return_from_start_percent": round(
                        (
                            equity
                            - starting_capital
                        )
                        / starting_capital
                        * 100
                        if starting_capital > 0
                        else 0.0,
                        4,
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_drawdown_curve(
    equity_curve: pd.DataFrame,
) -> pd.DataFrame:
    result = equity_curve.copy()

    result["running_peak"] = (
        result["equity"].cummax()
    )

    result["drawdown_dollars"] = (
        result["equity"]
        - result["running_peak"]
    )

    result["drawdown_percent"] = np.where(
        result["running_peak"] > 0,
        (
            result["equity"]
            - result["running_peak"]
        )
        / result["running_peak"]
        * 100,
        0.0,
    )

    return result


def periodic_returns(
    closed: pd.DataFrame,
    frequency: str,
    label: str,
    starting_capital: float,
) -> pd.DataFrame:
    columns = [
        label,
        "trade_count",
        "wins",
        "losses",
        "net_pnl_dollars",
        "return_on_starting_capital_percent",
    ]

    if closed.empty:
        return pd.DataFrame(columns=columns)

    dated = closed.dropna(
        subset=["exit_date"]
    ).copy()

    if dated.empty:
        return pd.DataFrame(columns=columns)

    dated[label] = (
        dated["exit_date"]
        .dt.to_period(frequency)
        .astype(str)
    )

    rows: list[dict[str, Any]] = []

    for period_name, group in dated.groupby(label):
        net_pnl = float(
            group["pnl_dollars"].sum()
        )

        rows.append(
            {
                label: str(period_name),
                "trade_count": len(group),
                "wins": int(
                    (group["pnl_dollars"] > 0).sum()
                ),
                "losses": int(
                    (group["pnl_dollars"] < 0).sum()
                ),
                "net_pnl_dollars": round(
                    net_pnl,
                    2,
                ),
                "return_on_starting_capital_percent": round(
                    net_pnl
                    / starting_capital
                    * 100
                    if starting_capital > 0
                    else 0.0,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        label
    ).reset_index(drop=True)


def build_rolling_metrics(
    closed: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    columns = [
        "sequence",
        "exit_date",
        "trade_id",
        "rolling_trade_count",
        "rolling_win_rate_percent",
        "rolling_expectancy_dollars",
        "rolling_expectancy_percent",
        "rolling_sharpe_ratio",
    ]

    if closed.empty:
        return pd.DataFrame(columns=columns)

    result = closed[
        [
            "exit_date",
            "trade_id",
            "pnl_dollars",
            "pnl_percent",
        ]
    ].copy()

    result = result.reset_index(drop=True)
    result["sequence"] = (
        np.arange(len(result)) + 1
    )

    wins = (
        result["pnl_dollars"] > 0
    ).astype(float)

    result["rolling_trade_count"] = (
        result["pnl_dollars"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .count()
        .astype(int)
    )

    result["rolling_win_rate_percent"] = (
        wins.rolling(
            window=window,
            min_periods=1,
        )
        .mean()
        * 100
    )

    result["rolling_expectancy_dollars"] = (
        result["pnl_dollars"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    result["rolling_expectancy_percent"] = (
        result["pnl_percent"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    sharpe_values: list[float] = []

    for index in range(len(result)):
        start_index = max(
            0,
            index - window + 1,
        )

        sample = result.loc[
            start_index:index,
            "pnl_percent",
        ] / 100

        sharpe_values.append(
            annualized_sharpe(sample)
        )

    result["rolling_sharpe_ratio"] = (
        sharpe_values
    )

    for column in [
        "rolling_win_rate_percent",
        "rolling_expectancy_dollars",
        "rolling_expectancy_percent",
        "rolling_sharpe_ratio",
    ]:
        result[column] = (
            result[column].round(4)
        )

    result["exit_date"] = (
        result["exit_date"]
        .dt.strftime("%Y-%m-%d")
    )

    return result[columns]


# ============================================================
# SUMMARY CALCULATION
# ============================================================

def calculate_summary(
    ledger: pd.DataFrame,
    closed: pd.DataFrame,
    open_positions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    drawdown_curve: pd.DataFrame,
    strategy_analytics: pd.DataFrame,
    symbol_analytics: pd.DataFrame,
    starting_capital: float,
) -> AnalyticsSummary:
    closed_count = len(closed)
    open_count = len(open_positions)

    wins = (
        closed[closed["pnl_dollars"] > 0]
        if not closed.empty
        else pd.DataFrame()
    )
    losses = (
        closed[closed["pnl_dollars"] < 0]
        if not closed.empty
        else pd.DataFrame()
    )
    breakeven = (
        closed[closed["pnl_dollars"] == 0]
        if not closed.empty
        else pd.DataFrame()
    )

    realized_pnl = (
        float(
            closed["pnl_dollars"].sum()
        )
        if not closed.empty
        else 0.0
    )

    unrealized_pnl = (
        float(
            open_positions[
                "unrealized_pnl_dollars"
            ].sum()
        )
        if (
            not open_positions.empty
            and "unrealized_pnl_dollars"
            in open_positions.columns
        )
        else 0.0
    )

    ending_equity = (
        float(
            equity_curve.iloc[-1]["equity"]
        )
        if not equity_curve.empty
        else starting_capital
    )

    estimated_equity = (
        ending_equity
        + unrealized_pnl
    )

    gross_profit = (
        float(
            wins["pnl_dollars"].sum()
        )
        if not wins.empty
        else 0.0
    )

    gross_loss = abs(
        float(
            losses["pnl_dollars"].sum()
        )
        if not losses.empty
        else 0.0
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = 99.0
    else:
        profit_factor = 0.0

    average_winner_dollars = (
        float(
            wins["pnl_dollars"].mean()
        )
        if not wins.empty
        else 0.0
    )

    average_loser_dollars = (
        float(
            losses["pnl_dollars"].mean()
        )
        if not losses.empty
        else 0.0
    )

    average_winner_percent = (
        float(
            wins["pnl_percent"].mean()
        )
        if not wins.empty
        else 0.0
    )

    average_loser_percent = (
        float(
            losses["pnl_percent"].mean()
        )
        if not losses.empty
        else 0.0
    )

    payoff_ratio = (
        abs(
            average_winner_dollars
            / average_loser_dollars
        )
        if average_loser_dollars != 0
        else 0.0
    )

    max_drawdown_dollars = (
        abs(
            float(
                drawdown_curve[
                    "drawdown_dollars"
                ].min()
            )
        )
        if not drawdown_curve.empty
        else 0.0
    )

    max_drawdown_percent = (
        abs(
            float(
                drawdown_curve[
                    "drawdown_percent"
                ].min()
            )
        )
        if not drawdown_curve.empty
        else 0.0
    )

    recovery_factor = (
        realized_pnl
        / max_drawdown_dollars
        if max_drawdown_dollars > 0
        else 0.0
    )

    trade_returns = (
        closed["pnl_percent"] / 100
        if not closed.empty
        else pd.Series(
            dtype=float
        )
    )

    best_strategy = (
        str(
            strategy_analytics.iloc[0][
                "strategy_name"
            ]
        )
        if not strategy_analytics.empty
        else None
    )

    worst_strategy = (
        str(
            strategy_analytics.iloc[-1][
                "strategy_name"
            ]
        )
        if not strategy_analytics.empty
        else None
    )

    best_symbol = (
        str(
            symbol_analytics.iloc[0][
                "symbol"
            ]
        )
        if not symbol_analytics.empty
        else None
    )

    worst_symbol = (
        str(
            symbol_analytics.iloc[-1][
                "symbol"
            ]
        )
        if not symbol_analytics.empty
        else None
    )

    best_trade_id = (
        str(
            closed.loc[
                closed["pnl_dollars"].idxmax(),
                "trade_id",
            ]
        )
        if not closed.empty
        else None
    )

    worst_trade_id = (
        str(
            closed.loc[
                closed["pnl_dollars"].idxmin(),
                "trade_id",
            ]
        )
        if not closed.empty
        else None
    )

    notes = [
        "Research-only performance analytics.",
        "The trade ledger was not modified.",
        "No trading client was created.",
        "No order was submitted.",
    ]

    if closed_count == 0:
        analytics_status = (
            "WAITING_FOR_CLOSED_TRADES"
        )
        notes.append(
            "No closed trades are available yet; "
            "closed-trade statistics are preliminary."
        )
    else:
        analytics_status = "READY"

    return AnalyticsSummary(
        generated_at=datetime.now().astimezone().isoformat(),
        starting_capital=round(
            starting_capital,
            2,
        ),
        ending_equity=round(
            ending_equity,
            2,
        ),
        open_position_count=open_count,
        closed_position_count=closed_count,
        wins=len(wins),
        losses=len(losses),
        breakeven_trades=len(breakeven),
        win_rate_percent=round(
            len(wins)
            / closed_count
            * 100
            if closed_count
            else 0.0,
            4,
        ),
        total_realized_pnl_dollars=round(
            realized_pnl,
            2,
        ),
        total_realized_return_percent=round(
            realized_pnl
            / starting_capital
            * 100
            if starting_capital > 0
            else 0.0,
            4,
        ),
        total_unrealized_pnl_dollars=round(
            unrealized_pnl,
            2,
        ),
        estimated_equity_with_unrealized=round(
            estimated_equity,
            2,
        ),
        estimated_total_return_percent=round(
            (
                estimated_equity
                - starting_capital
            )
            / starting_capital
            * 100
            if starting_capital > 0
            else 0.0,
            4,
        ),
        gross_profit_dollars=round(
            gross_profit,
            2,
        ),
        gross_loss_dollars=round(
            gross_loss,
            2,
        ),
        profit_factor=round(
            profit_factor,
            4,
        ),
        average_trade_pnl_dollars=round(
            float(
                closed["pnl_dollars"].mean()
            )
            if closed_count
            else 0.0,
            2,
        ),
        average_trade_return_percent=round(
            float(
                closed["pnl_percent"].mean()
            )
            if closed_count
            else 0.0,
            4,
        ),
        average_winner_dollars=round(
            average_winner_dollars,
            2,
        ),
        average_winner_percent=round(
            average_winner_percent,
            4,
        ),
        average_loser_dollars=round(
            average_loser_dollars,
            2,
        ),
        average_loser_percent=round(
            average_loser_percent,
            4,
        ),
        expectancy_dollars=round(
            float(
                closed["pnl_dollars"].mean()
            )
            if closed_count
            else 0.0,
            2,
        ),
        expectancy_percent=round(
            float(
                closed["pnl_percent"].mean()
            )
            if closed_count
            else 0.0,
            4,
        ),
        largest_winner_dollars=round(
            float(
                closed["pnl_dollars"].max()
            )
            if closed_count
            else 0.0,
            2,
        ),
        largest_winner_percent=round(
            float(
                closed["pnl_percent"].max()
            )
            if closed_count
            else 0.0,
            4,
        ),
        largest_loser_dollars=round(
            float(
                closed["pnl_dollars"].min()
            )
            if closed_count
            else 0.0,
            2,
        ),
        largest_loser_percent=round(
            float(
                closed["pnl_percent"].min()
            )
            if closed_count
            else 0.0,
            4,
        ),
        average_holding_days=round(
            float(
                closed["holding_days"].mean()
            )
            if closed_count
            else 0.0,
            2,
        ),
        median_holding_days=round(
            float(
                closed["holding_days"].median()
            )
            if closed_count
            else 0.0,
            2,
        ),
        maximum_holding_days=int(
            closed["holding_days"].max()
        )
        if closed_count
        else 0,
        maximum_drawdown_dollars=round(
            max_drawdown_dollars,
            2,
        ),
        maximum_drawdown_percent=round(
            max_drawdown_percent,
            4,
        ),
        recovery_factor=round(
            recovery_factor,
            4,
        ),
        payoff_ratio=round(
            payoff_ratio,
            4,
        ),
        sharpe_ratio=round(
            annualized_sharpe(
                trade_returns
            ),
            4,
        ),
        sortino_ratio=round(
            annualized_sortino(
                trade_returns
            ),
            4,
        ),
        best_strategy=best_strategy,
        worst_strategy=worst_strategy,
        best_symbol=best_symbol,
        worst_symbol=worst_symbol,
        best_trade_id=best_trade_id,
        worst_trade_id=worst_trade_id,
        first_exit_date=(
            closed["exit_date"]
            .min()
            .date()
            .isoformat()
            if (
                closed_count
                and pd.notna(
                    closed["exit_date"].min()
                )
            )
            else None
        ),
        last_exit_date=(
            closed["exit_date"]
            .max()
            .date()
            .isoformat()
            if (
                closed_count
                and pd.notna(
                    closed["exit_date"].max()
                )
            )
            else None
        ),
        analytics_status=analytics_status,
        notes=notes,
        shadow_mode=True,
        ledger_modified=False,
        trading_client_created=False,
        market_request_made=False,
        order_submitted=False,
    )


# ============================================================
# REPORT WRITING
# ============================================================

def write_text_report(
    summary: AnalyticsSummary,
) -> None:
    lines = [
        "PERFORMANCE ANALYTICS V1",
        "=" * 32,
        f"Generated at: {summary.generated_at}",
        f"Analytics status: {summary.analytics_status}",
        "",
        "PORTFOLIO",
        "-" * 20,
        f"Starting capital: ${summary.starting_capital:.2f}",
        f"Ending equity: ${summary.ending_equity:.2f}",
        (
            "Estimated equity with unrealized P/L: "
            f"${summary.estimated_equity_with_unrealized:.2f}"
        ),
        (
            "Estimated total return: "
            f"{summary.estimated_total_return_percent:.2f}%"
        ),
        "",
        "TRADE RESULTS",
        "-" * 20,
        f"Open positions: {summary.open_position_count}",
        f"Closed positions: {summary.closed_position_count}",
        f"Wins: {summary.wins}",
        f"Losses: {summary.losses}",
        f"Breakeven: {summary.breakeven_trades}",
        f"Win rate: {summary.win_rate_percent:.2f}%",
        (
            "Total realized P/L: "
            f"${summary.total_realized_pnl_dollars:.2f}"
        ),
        (
            "Total unrealized P/L: "
            f"${summary.total_unrealized_pnl_dollars:.2f}"
        ),
        f"Profit factor: {summary.profit_factor:.2f}",
        (
            "Expectancy: "
            f"${summary.expectancy_dollars:.2f} "
            f"({summary.expectancy_percent:.2f}%)"
        ),
        "",
        "RISK",
        "-" * 20,
        (
            "Maximum drawdown: "
            f"${summary.maximum_drawdown_dollars:.2f} "
            f"({summary.maximum_drawdown_percent:.2f}%)"
        ),
        f"Sharpe ratio: {summary.sharpe_ratio:.2f}",
        f"Sortino ratio: {summary.sortino_ratio:.2f}",
        f"Recovery factor: {summary.recovery_factor:.2f}",
        f"Payoff ratio: {summary.payoff_ratio:.2f}",
        "",
        "LEADERS",
        "-" * 20,
        f"Best strategy: {summary.best_strategy}",
        f"Worst strategy: {summary.worst_strategy}",
        f"Best symbol: {summary.best_symbol}",
        f"Worst symbol: {summary.worst_symbol}",
        "",
        "SAFETY",
        "-" * 20,
        "Shadow mode: True",
        "Ledger modified: False",
        "Trading client created: False",
        "Market request made: False",
        "Order submitted: False",
    ]

    LATEST_TEXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate research-only performance analytics "
            "from the shadow trade ledger."
        )
    )

    parser.add_argument(
        "--starting-capital",
        type=float,
        default=2000.0,
        help="Starting capital used for return calculations.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Rolling trade window for rolling analytics.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.starting_capital <= 0:
        raise SystemExit(
            "Starting capital must be greater than zero."
        )

    if arguments.rolling_window <= 0:
        raise SystemExit(
            "Rolling window must be greater than zero."
        )

    raw_ledger = load_json(
        TRADE_LEDGER_PATH,
        [],
    )

    ledger = ledger_to_frame(
        raw_ledger
    )

    if ledger.empty:
        ledger = pd.DataFrame(
            columns=[
                "trade_id",
                "strategy_name",
                "symbol",
                "side",
                "status",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "proposed_dollars",
                "shares",
                "pnl_dollars",
                "pnl_percent",
                "holding_days",
                "unrealized_pnl_dollars",
                "unrealized_pnl_percent",
            ]
        )

    closed = enrich_closed_trades(
        ledger[
            ledger["status"] == "CLOSED"
        ].copy()
    )

    open_positions = ledger[
        ledger["status"] == "OPEN"
    ].copy()

    strategy_analytics = group_analytics(
        closed,
        "strategy_name",
    )

    symbol_analytics = group_analytics(
        closed,
        "symbol",
    )

    side_analytics = group_analytics(
        closed,
        "side",
    )

    equity_curve = build_equity_curve(
        closed,
        arguments.starting_capital,
    )

    drawdown_curve = build_drawdown_curve(
        equity_curve
    )

    monthly_returns = periodic_returns(
        closed,
        "M",
        "month",
        arguments.starting_capital,
    )

    weekly_returns = periodic_returns(
        closed,
        "W",
        "week",
        arguments.starting_capital,
    )

    rolling_metrics = build_rolling_metrics(
        closed,
        arguments.rolling_window,
    )

    summary = calculate_summary(
        ledger=ledger,
        closed=closed,
        open_positions=open_positions,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        strategy_analytics=strategy_analytics,
        symbol_analytics=symbol_analytics,
        starting_capital=arguments.starting_capital,
    )

    summary_payload = asdict(summary)

    save_json(
        LATEST_SUMMARY_PATH,
        summary_payload,
    )

    save_json(
        HISTORY_DIRECTORY
        / (
            datetime.now()
            .astimezone()
            .strftime("%Y-%m-%d_%H-%M-%S")
            + ".json"
        ),
        summary_payload,
    )

    ledger.to_csv(
        TRADE_ANALYTICS_PATH,
        index=False,
    )

    strategy_analytics.to_csv(
        STRATEGY_ANALYTICS_PATH,
        index=False,
    )

    symbol_analytics.to_csv(
        SYMBOL_ANALYTICS_PATH,
        index=False,
    )

    side_analytics.to_csv(
        SIDE_ANALYTICS_PATH,
        index=False,
    )

    monthly_returns.to_csv(
        MONTHLY_RETURNS_PATH,
        index=False,
    )

    weekly_returns.to_csv(
        WEEKLY_RETURNS_PATH,
        index=False,
    )

    equity_curve.to_csv(
        EQUITY_CURVE_PATH,
        index=False,
    )

    drawdown_curve.to_csv(
        DRAWDOWN_PATH,
        index=False,
    )

    rolling_metrics.to_csv(
        ROLLING_METRICS_PATH,
        index=False,
    )

    open_positions.to_csv(
        OPEN_POSITIONS_PATH,
        index=False,
    )

    write_text_report(summary)

    print("Performance Analytics v1")
    print(json.dumps(summary_payload, indent=2))
    print()
    print(f"JSON report: {LATEST_SUMMARY_PATH}")
    print(f"Text report: {LATEST_TEXT_PATH}")
    print(f"Strategy analytics: {STRATEGY_ANALYTICS_PATH}")
    print(f"Symbol analytics: {SYMBOL_ANALYTICS_PATH}")
    print(f"Equity curve: {EQUITY_CURVE_PATH}")
    print("The trade ledger was not modified.")
    print("No trading client was created.")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()