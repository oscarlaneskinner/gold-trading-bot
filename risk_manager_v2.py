"""
Risk Manager v2

Research-only portfolio risk engine for the gold-trading-bot project.

This version prefers the newest Position Manager v2 valuation data
when matching an open trade by trade_id.

It:
- reads the shadow trade ledger,
- reads Position Manager v2,
- reads Portfolio Commander,
- calculates exposure and portfolio heat,
- checks stops and targets,
- evaluates loss limits,
- assigns a 0-100 risk score,
- produces risk warnings,
- never modifies the trade ledger,
- never creates a broker client,
- never submits an order.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"
POSITION_MANAGER_DIRECTORY = REPORTS_DIRECTORY / "position_manager"
PORTFOLIO_DIRECTORY = REPORTS_DIRECTORY / "portfolio"
RISK_DIRECTORY = REPORTS_DIRECTORY / "risk_manager"
RISK_HISTORY_DIRECTORY = RISK_DIRECTORY / "history"

TRADE_LEDGER_PATH = (
    PERFORMANCE_DIRECTORY / "trade_ledger.json"
)

PERFORMANCE_SUMMARY_PATH = (
    PERFORMANCE_DIRECTORY / "latest_summary.json"
)

POSITION_MANAGER_SUMMARY_PATH = (
    POSITION_MANAGER_DIRECTORY / "latest_summary.json"
)

PORTFOLIO_COMMANDER_PATH = (
    PORTFOLIO_DIRECTORY / "portfolio_commander_v1.json"
)

LATEST_SUMMARY_PATH = (
    RISK_DIRECTORY / "latest_summary.json"
)

LATEST_TEXT_PATH = (
    RISK_DIRECTORY / "latest_summary.txt"
)

POSITION_RISK_PATH = (
    RISK_DIRECTORY / "position_risk.csv"
)

WARNINGS_PATH = (
    RISK_DIRECTORY / "risk_warnings.csv"
)

RISK_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

RISK_HISTORY_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class PositionRisk:
    trade_id: str
    symbol: str
    strategy_name: str
    side: str
    proposed_dollars: float
    allocation_percent: float
    entry_price: float
    current_price: float
    stop_price: float | None
    target_price: float | None
    shares: float
    unrealized_pnl_dollars: float
    unrealized_pnl_percent: float
    stop_risk_dollars: float
    stop_risk_percent_of_position: float
    stop_risk_percent_of_account: float
    has_stop: bool
    has_target: bool
    holding_days: int
    status: str


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(
    path: Path,
    default: Any,
) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return default


def save_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None:
            return default

        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def normalize_side(
    value: Any,
) -> str:
    side = str(
        value or ""
    ).strip().upper()

    if side in {
        "BUY",
        "LONG",
    }:
        return "LONG"

    if side in {
        "SELL",
        "SHORT",
    }:
        return "SHORT"

    return side or "UNKNOWN"


# ============================================================
# POSITION MANAGER MERGE
# ============================================================

def build_position_manager_lookup(
    position_manager_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Build a dictionary of the newest Position Manager
    open-position records indexed by trade_id.
    """

    lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    open_positions = (
        position_manager_summary.get(
            "open_positions",
            [],
        )
    )

    if not isinstance(
        open_positions,
        list,
    ):
        return lookup

    for position in open_positions:
        if not isinstance(
            position,
            dict,
        ):
            continue

        trade_id = str(
            position.get(
                "trade_id",
                "",
            )
        ).strip()

        if not trade_id:
            continue

        lookup[
            trade_id
        ] = position

    return lookup


def merge_latest_position_data(
    ledger_trade: dict[str, Any],
    position_manager_lookup: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    """
    Overlay newer Position Manager valuation fields
    onto the ledger trade without changing either source file.
    """

    merged = dict(
        ledger_trade
    )

    trade_id = str(
        ledger_trade.get(
            "trade_id",
            "",
        )
    ).strip()

    newer = (
        position_manager_lookup.get(
            trade_id
        )
    )

    if not newer:
        return merged

    fields_to_refresh = [
        "current_price",
        "unrealized_pnl_dollars",
        "unrealized_pnl_percent",
        "holding_days",
        "last_valued_date",
        "stop_price",
        "target_price",
        "shares",
        "proposed_dollars",
        "status",
    ]

    for field in fields_to_refresh:
        if field in newer:
            merged[
                field
            ] = newer[
                field
            ]

    return merged


# ============================================================
# POSITION RISK CALCULATIONS
# ============================================================

def calculate_stop_risk(
    side: str,
    entry_price: float,
    stop_price: float | None,
    shares: float,
) -> float:
    if (
        stop_price is None
        or entry_price <= 0
        or shares <= 0
    ):
        return 0.0

    if normalize_side(
        side
    ) == "SHORT":

        risk_per_share = max(
            stop_price
            - entry_price,
            0.0,
        )

    else:

        risk_per_share = max(
            entry_price
            - stop_price,
            0.0,
        )

    return (
        risk_per_share
        * shares
    )


def build_position_risk(
    trade: dict[str, Any],
    starting_capital: float,
) -> PositionRisk:

    proposed_dollars = safe_float(
        trade.get(
            "proposed_dollars"
        )
    )

    entry_price = safe_float(
        trade.get(
            "entry_price"
        )
    )

    current_price = safe_float(
        trade.get(
            "current_price",
            entry_price,
        ),
        default=entry_price,
    )

    shares = safe_float(
        trade.get(
            "shares"
        )
    )

    stop_price = (
        safe_float(
            trade.get(
                "stop_price"
            )
        )
        if trade.get(
            "stop_price"
        ) is not None
        else None
    )

    target_price = (
        safe_float(
            trade.get(
                "target_price"
            )
        )
        if trade.get(
            "target_price"
        ) is not None
        else None
    )

    unrealized_pnl_dollars = (
        safe_float(
            trade.get(
                "unrealized_pnl_dollars"
            )
        )
    )

    unrealized_pnl_percent = (
        safe_float(
            trade.get(
                "unrealized_pnl_percent"
            )
        )
    )

    side = normalize_side(
        trade.get(
            "side"
        )
    )

    stop_risk_dollars = (
        calculate_stop_risk(
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            shares=shares,
        )
    )

    allocation_percent = (
        proposed_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    stop_risk_percent_of_position = (
        stop_risk_dollars
        / proposed_dollars
        * 100
        if proposed_dollars > 0
        else 0.0
    )

    stop_risk_percent_of_account = (
        stop_risk_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    return PositionRisk(
        trade_id=str(
            trade.get(
                "trade_id",
                "",
            )
        ),

        symbol=str(
            trade.get(
                "symbol",
                "UNKNOWN",
            )
        ).upper(),

        strategy_name=str(
            trade.get(
                "strategy_name",
                "UNKNOWN",
            )
        ),

        side=side,

        proposed_dollars=round(
            proposed_dollars,
            2,
        ),

        allocation_percent=round(
            allocation_percent,
            4,
        ),

        entry_price=round(
            entry_price,
            4,
        ),

        current_price=round(
            current_price,
            4,
        ),

        stop_price=(
            round(
                stop_price,
                4,
            )
            if stop_price is not None
            else None
        ),

        target_price=(
            round(
                target_price,
                4,
            )
            if target_price is not None
            else None
        ),

        shares=round(
            shares,
            8,
        ),

        unrealized_pnl_dollars=round(
            unrealized_pnl_dollars,
            2,
        ),

        unrealized_pnl_percent=round(
            unrealized_pnl_percent,
            4,
        ),

        stop_risk_dollars=round(
            stop_risk_dollars,
            2,
        ),

        stop_risk_percent_of_position=round(
            stop_risk_percent_of_position,
            4,
        ),

        stop_risk_percent_of_account=round(
            stop_risk_percent_of_account,
            4,
        ),

        has_stop=(
            stop_price is not None
            and stop_price > 0
        ),

        has_target=(
            target_price is not None
            and target_price > 0
        ),

        holding_days=safe_int(
            trade.get(
                "holding_days"
            )
        ),

        status=str(
            trade.get(
                "status",
                "OPEN",
            )
        ).upper(),
    )


# ============================================================
# WARNINGS
# ============================================================

def add_warning(
    warnings: list[
        dict[str, Any]
    ],
    severity: str,
    category: str,
    message: str,
    symbol: str | None = None,
) -> None:

    warnings.append(
        {
            "severity": severity,
            "category": category,
            "symbol": (
                symbol or ""
            ),
            "message": message,
        }
    )


# ============================================================
# RISK EVALUATION
# ============================================================

def evaluate_risk(
    positions: list[
        PositionRisk
    ],
    starting_capital: float,
    realized_pnl: float,
    unrealized_pnl: float,
    max_position_percent: float,
    max_total_exposure_percent: float,
    max_single_symbol_percent: float,
    max_portfolio_heat_percent: float,
    daily_loss_limit_percent: float,
    max_drawdown_percent: float,
) -> tuple[
    list[
        dict[str, Any]
    ],
    dict[str, Any],
]:

    warnings: list[
        dict[str, Any]
    ] = []

    total_exposure_dollars = sum(
        position.proposed_dollars
        for position in positions
    )

    total_exposure_percent = (
        total_exposure_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    total_heat_dollars = sum(
        position.stop_risk_dollars
        for position in positions
    )

    total_heat_percent = (
        total_heat_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    symbol_exposure: dict[
        str,
        float,
    ] = {}

    for position in positions:

        symbol_exposure[
            position.symbol
        ] = (
            symbol_exposure.get(
                position.symbol,
                0.0,
            )
            + position.proposed_dollars
        )

        if (
            position.allocation_percent
            > max_position_percent
        ):

            add_warning(
                warnings=warnings,
                severity="HIGH",
                category="POSITION_SIZE",
                symbol=position.symbol,
                message=(
                    f"{position.symbol} allocation is "
                    f"{position.allocation_percent:.2f}% "
                    f"of capital, above the "
                    f"{max_position_percent:.2f}% limit."
                ),
            )

        if not position.has_stop:

            add_warning(
                warnings=warnings,
                severity="HIGH",
                category="STOP_LOSS",
                symbol=position.symbol,
                message=(
                    f"{position.symbol} has no "
                    "valid stop-loss price."
                ),
            )

        if not position.has_target:

            add_warning(
                warnings=warnings,
                severity="MEDIUM",
                category="PROFIT_TARGET",
                symbol=position.symbol,
                message=(
                    f"{position.symbol} has no "
                    "valid profit target."
                ),
            )

    for (
        symbol,
        dollars,
    ) in symbol_exposure.items():

        symbol_percent = (
            dollars
            / starting_capital
            * 100
            if starting_capital > 0
            else 0.0
        )

        if (
            symbol_percent
            > max_single_symbol_percent
        ):

            add_warning(
                warnings=warnings,
                severity="HIGH",
                category="SYMBOL_CONCENTRATION",
                symbol=symbol,
                message=(
                    f"{symbol} represents "
                    f"{symbol_percent:.2f}% "
                    "of capital, above the "
                    f"{max_single_symbol_percent:.2f}% "
                    "single-symbol limit."
                ),
            )

    if (
        total_exposure_percent
        > max_total_exposure_percent
    ):

        add_warning(
            warnings=warnings,
            severity="HIGH",
            category="TOTAL_EXPOSURE",
            message=(
                "Total portfolio exposure is "
                f"{total_exposure_percent:.2f}%, "
                "above the "
                f"{max_total_exposure_percent:.2f}% "
                "limit."
            ),
        )

    if (
        total_heat_percent
        > max_portfolio_heat_percent
    ):

        add_warning(
            warnings=warnings,
            severity="HIGH",
            category="PORTFOLIO_HEAT",
            message=(
                "Estimated portfolio stop-loss heat "
                f"is {total_heat_percent:.2f}% "
                "of capital, above the "
                f"{max_portfolio_heat_percent:.2f}% "
                "limit."
            ),
        )

    combined_pnl = (
        realized_pnl
        + unrealized_pnl
    )

    combined_return_percent = (
        combined_pnl
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    if (
        combined_return_percent
        <= -abs(
            daily_loss_limit_percent
        )
    ):

        add_warning(
            warnings=warnings,
            severity="CRITICAL",
            category="LOSS_LIMIT",
            message=(
                "Combined realized and unrealized "
                f"P/L is {combined_return_percent:.2f}% "
                "of starting capital, beyond the "
                f"{daily_loss_limit_percent:.2f}% "
                "loss-limit threshold."
            ),
        )

    if (
        combined_return_percent
        <= -abs(
            max_drawdown_percent
        )
    ):

        add_warning(
            warnings=warnings,
            severity="CRITICAL",
            category="DRAWDOWN",
            message=(
                "Estimated drawdown has reached "
                f"{abs(combined_return_percent):.2f}%, "
                "beyond the "
                f"{max_drawdown_percent:.2f}% "
                "maximum drawdown threshold."
            ),
        )

    summary = {
        "total_exposure_dollars": round(
            total_exposure_dollars,
            2,
        ),

        "total_exposure_percent": round(
            total_exposure_percent,
            4,
        ),

        "portfolio_heat_dollars": round(
            total_heat_dollars,
            2,
        ),

        "portfolio_heat_percent": round(
            total_heat_percent,
            4,
        ),

        "combined_pnl_dollars": round(
            combined_pnl,
            2,
        ),

        "combined_return_percent": round(
            combined_return_percent,
            4,
        ),

        "symbol_exposure": {
            symbol: {
                "dollars": round(
                    dollars,
                    2,
                ),
                "percent": round(
                    (
                        dollars
                        / starting_capital
                        * 100
                    )
                    if starting_capital > 0
                    else 0.0,
                    4,
                ),
            }
            for (
                symbol,
                dollars,
            ) in sorted(
                symbol_exposure.items()
            )
        },
    }

    return (
        warnings,
        summary,
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    warnings: list[
        dict[str, Any]
    ],
    total_exposure_percent: float,
    max_total_exposure_percent: float,
    portfolio_heat_percent: float,
    max_portfolio_heat_percent: float,
) -> int:

    score = 0.0

    for warning in warnings:

        severity = str(
            warning.get(
                "severity",
                "",
            )
        ).upper()

        if severity == "CRITICAL":
            score += 30

        elif severity == "HIGH":
            score += 18

        elif severity == "MEDIUM":
            score += 8

        elif severity == "LOW":
            score += 3

    if (
        max_total_exposure_percent
        > 0
    ):

        exposure_ratio = (
            total_exposure_percent
            / max_total_exposure_percent
        )

        score += min(
            max(
                exposure_ratio,
                0.0,
            )
            * 20,
            25,
        )

    if (
        max_portfolio_heat_percent
        > 0
    ):

        heat_ratio = (
            portfolio_heat_percent
            / max_portfolio_heat_percent
        )

        score += min(
            max(
                heat_ratio,
                0.0,
            )
            * 20,
            25,
        )

    return int(
        round(
            min(
                max(
                    score,
                    0.0,
                ),
                100.0,
            )
        )
    )


def risk_level(
    score: int,
) -> str:

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 35:
        return "MODERATE"

    if score >= 15:
        return "LOW"

    return "MINIMAL"


# ============================================================
# TRADE ACCEPTANCE
# ============================================================

def trade_acceptance_status(
    warnings: list[
        dict[str, Any]
    ],
    risk_score: int,
) -> tuple[
    str,
    list[str],
]:

    critical = [
        warning
        for warning in warnings
        if warning[
            "severity"
        ] == "CRITICAL"
    ]

    high = [
        warning
        for warning in warnings
        if warning[
            "severity"
        ] == "HIGH"
    ]

    if (
        critical
        or risk_score >= 80
    ):

        return (
            "BLOCK_NEW_SHADOW_TRADES",
            [
                "Critical portfolio-risk conditions "
                "are active or the risk score is at "
                "least 80."
            ],
        )

    if (
        high
        or risk_score >= 60
    ):

        return (
            "REVIEW_BEFORE_NEW_SHADOW_TRADE",
            [
                "High-severity portfolio-risk "
                "conditions are active or the "
                "risk score is elevated."
            ],
        )

    return (
        "SHADOW_RESEARCH_ALLOWED",
        [
            "No critical or high-severity "
            "portfolio-risk condition requires "
            "blocking new shadow trades."
        ],
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    payload: dict[str, Any],
) -> None:

    lines = [
        "RISK MANAGER V2",
        "=" * 28,

        f"Generated at: "
        f"{payload['generated_at']}",

        f"Risk score: "
        f"{payload['risk_score']}/100",

        f"Risk level: "
        f"{payload['risk_level']}",

        (
            "Trade acceptance status: "
            f"{payload['trade_acceptance_status']}"
        ),

        "",

        "PORTFOLIO",
        "-" * 20,

        (
            "Starting capital: "
            f"${payload['starting_capital']:.2f}"
        ),

        (
            "Open positions: "
            f"{payload['open_position_count']}"
        ),

        (
            "Total exposure: "
            f"${payload['portfolio_metrics']['total_exposure_dollars']:.2f} "
            f"({payload['portfolio_metrics']['total_exposure_percent']:.2f}%)"
        ),

        (
            "Portfolio heat: "
            f"${payload['portfolio_metrics']['portfolio_heat_dollars']:.2f} "
            f"({payload['portfolio_metrics']['portfolio_heat_percent']:.2f}%)"
        ),

        (
            "Combined P/L: "
            f"${payload['portfolio_metrics']['combined_pnl_dollars']:.2f} "
            f"({payload['portfolio_metrics']['combined_return_percent']:.2f}%)"
        ),

        "",

        "WARNINGS",
        "-" * 20,
    ]

    if payload[
        "warnings"
    ]:

        for warning in payload[
            "warnings"
        ]:

            lines.append(
                f"{warning['severity']} | "
                f"{warning['category']} | "
                f"{warning['symbol']} | "
                f"{warning['message']}"
            )

    else:

        lines.append(
            "No active risk warnings."
        )

    lines.extend(
        [
            "",
            "SAFETY",
            "-" * 20,

            "Research shadow mode only.",
            "Ledger modified: False",
            "Trading client created: False",
            "Market request made: False",
            "Order submitted: False",
        ]
    )

    LATEST_TEXT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate research-only portfolio "
            "risk analysis."
        )
    )

    parser.add_argument(
        "--starting-capital",
        type=float,
        default=2000.0,
    )

    parser.add_argument(
        "--max-position-percent",
        type=float,
        default=35.0,
    )

    parser.add_argument(
        "--max-total-exposure-percent",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--max-single-symbol-percent",
        type=float,
        default=35.0,
    )

    parser.add_argument(
        "--max-portfolio-heat-percent",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--daily-loss-limit-percent",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-drawdown-percent",
        type=float,
        default=12.0,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    arguments = parse_arguments()

    if (
        arguments.starting_capital
        <= 0
    ):

        raise SystemExit(
            "Starting capital must be "
            "greater than zero."
        )

    raw_ledger = load_json(
        TRADE_LEDGER_PATH,
        [],
    )

    performance_summary = load_json(
        PERFORMANCE_SUMMARY_PATH,
        {},
    )

    position_manager_summary = load_json(
        POSITION_MANAGER_SUMMARY_PATH,
        {},
    )

    portfolio_commander = load_json(
        PORTFOLIO_COMMANDER_PATH,
        {},
    )

    starting_capital = safe_float(
        position_manager_summary.get(
            "starting_capital",
            performance_summary.get(
                "starting_capital",
                arguments.starting_capital,
            ),
        ),
        default=(
            arguments.starting_capital
        ),
    )

    position_manager_lookup = (
        build_position_manager_lookup(
            position_manager_summary
        )
    )

    open_trade_rows: list[
        dict[str, Any]
    ] = []

    if isinstance(
        raw_ledger,
        list,
    ):

        for trade in raw_ledger:

            if not isinstance(
                trade,
                dict,
            ):
                continue

            if str(
                trade.get(
                    "status",
                    "",
                )
            ).upper() != "OPEN":

                continue

            merged_trade = (
                merge_latest_position_data(
                    ledger_trade=trade,
                    position_manager_lookup=(
                        position_manager_lookup
                    ),
                )
            )

            open_trade_rows.append(
                merged_trade
            )

    position_risks = [
        build_position_risk(
            trade,
            starting_capital,
        )
        for trade in open_trade_rows
    ]

    realized_pnl = safe_float(
        position_manager_summary.get(
            "realized_pnl_dollars",
            performance_summary.get(
                "total_pnl_dollars",
                0.0,
            ),
        )
    )

    if position_risks:

        unrealized_pnl = sum(
            position.unrealized_pnl_dollars
            for position in position_risks
        )

    else:

        unrealized_pnl = safe_float(
            position_manager_summary.get(
                "unrealized_pnl_dollars",
                0.0,
            )
        )

    (
        warnings,
        portfolio_metrics,
    ) = evaluate_risk(

        positions=position_risks,

        starting_capital=(
            starting_capital
        ),

        realized_pnl=(
            realized_pnl
        ),

        unrealized_pnl=(
            unrealized_pnl
        ),

        max_position_percent=(
            arguments.max_position_percent
        ),

        max_total_exposure_percent=(
            arguments.max_total_exposure_percent
        ),

        max_single_symbol_percent=(
            arguments.max_single_symbol_percent
        ),

        max_portfolio_heat_percent=(
            arguments.max_portfolio_heat_percent
        ),

        daily_loss_limit_percent=(
            arguments.daily_loss_limit_percent
        ),

        max_drawdown_percent=(
            arguments.max_drawdown_percent
        ),
    )

    score = calculate_risk_score(

        warnings=warnings,

        total_exposure_percent=(
            portfolio_metrics[
                "total_exposure_percent"
            ]
        ),

        max_total_exposure_percent=(
            arguments.max_total_exposure_percent
        ),

        portfolio_heat_percent=(
            portfolio_metrics[
                "portfolio_heat_percent"
            ]
        ),

        max_portfolio_heat_percent=(
            arguments.max_portfolio_heat_percent
        ),
    )

    level = risk_level(
        score
    )

    (
        acceptance_status,
        acceptance_reasons,
    ) = trade_acceptance_status(
        warnings=warnings,
        risk_score=score,
    )

    payload = {

        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "starting_capital": round(
            starting_capital,
            2,
        ),

        "open_position_count": len(
            position_risks
        ),

        "risk_score": score,

        "risk_level": level,

        "trade_acceptance_status": (
            acceptance_status
        ),

        "trade_acceptance_reasons": (
            acceptance_reasons
        ),

        "limits": {

            "max_position_percent": (
                arguments.max_position_percent
            ),

            "max_total_exposure_percent": (
                arguments.max_total_exposure_percent
            ),

            "max_single_symbol_percent": (
                arguments.max_single_symbol_percent
            ),

            "max_portfolio_heat_percent": (
                arguments.max_portfolio_heat_percent
            ),

            "daily_loss_limit_percent": (
                arguments.daily_loss_limit_percent
            ),

            "max_drawdown_percent": (
                arguments.max_drawdown_percent
            ),
        },

        "portfolio_metrics": (
            portfolio_metrics
        ),

        "positions": [
            asdict(
                position
            )
            for position
            in position_risks
        ],

        "warnings": warnings,

        "warning_count": len(
            warnings
        ),

        "critical_warning_count": sum(
            warning[
                "severity"
            ] == "CRITICAL"
            for warning in warnings
        ),

        "high_warning_count": sum(
            warning[
                "severity"
            ] == "HIGH"
            for warning in warnings
        ),

        "portfolio_commander_allocations": (
            portfolio_commander.get(
                "allocations",
                [],
            )
        ),

        "position_manager_data_used": (
            bool(
                position_manager_lookup
            )
        ),

        "shadow_mode": True,

        "ledger_modified": False,

        "trading_client_created": False,

        "market_request_made": False,

        "order_submitted": False,
    }

    save_json(
        LATEST_SUMMARY_PATH,
        payload,
    )

    save_json(
        RISK_HISTORY_DIRECTORY
        / (
            datetime.now()
            .astimezone()
            .strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        ),
        payload,
    )

    pd.DataFrame(
        [
            asdict(
                position
            )
            for position
            in position_risks
        ]
    ).to_csv(
        POSITION_RISK_PATH,
        index=False,
    )

    pd.DataFrame(
        warnings,
        columns=[
            "severity",
            "category",
            "symbol",
            "message",
        ],
    ).to_csv(
        WARNINGS_PATH,
        index=False,
    )

    write_text_report(
        payload
    )

    print(
        "Risk Manager v2"
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()

    print(
        f"JSON report: "
        f"{LATEST_SUMMARY_PATH}"
    )

    print(
        f"Text report: "
        f"{LATEST_TEXT_PATH}"
    )

    print(
        f"Position risk: "
        f"{POSITION_RISK_PATH}"
    )

    print(
        f"Warnings: "
        f"{WARNINGS_PATH}"
    )

    print(
        "The trade ledger was not modified."
    )

    print(
        "No trading client was created."
    )

    print(
        "No market request was made."
    )

    print(
        "No order was submitted."
    )


if __name__ == "__main__":
    main()