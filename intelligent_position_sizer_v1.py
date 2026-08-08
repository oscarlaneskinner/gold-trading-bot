"""
Intelligent Position Sizer v1

Research-only adaptive position sizing engine for the
gold-trading-bot platform.

This module recommends position sizes using:

- Starting capital
- Portfolio Commander allocation
- Risk Manager score
- Current portfolio exposure
- Portfolio heat
- Market regime
- Scanner confidence score
- ATR / volatility
- Existing open positions
- Stop-loss distance

It does NOT:
- modify the trade ledger,
- create a trading client,
- submit an order,
- place paper trades,
- place live trades.

Run:
    python intelligent_position_sizer_v1.py
"""

from __future__ import annotations

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

PORTFOLIO_DIRECTORY = (
    REPORTS_DIRECTORY / "portfolio"
)

RISK_DIRECTORY = (
    REPORTS_DIRECTORY / "risk_manager"
)

MARKET_REGIME_DIRECTORY = (
    REPORTS_DIRECTORY / "market_regime"
)

SCANNER_DIRECTORY = (
    REPORTS_DIRECTORY / "scanner"
)

SHADOW_DIRECTORY = (
    REPORTS_DIRECTORY / "shadow"
)

POSITION_MANAGER_DIRECTORY = (
    REPORTS_DIRECTORY / "position_manager"
)

SIZER_DIRECTORY = (
    REPORTS_DIRECTORY / "position_sizer"
)

SIZER_HISTORY_DIRECTORY = (
    SIZER_DIRECTORY / "history"
)


PORTFOLIO_PATH = (
    PORTFOLIO_DIRECTORY
    / "portfolio_commander_v1.json"
)

RISK_PATH = (
    RISK_DIRECTORY
    / "latest_summary.json"
)

MARKET_REGIME_PATH = (
    MARKET_REGIME_DIRECTORY
    / "market_regime_lab_v1.json"
)

SCANNER_PATH = (
    SCANNER_DIRECTORY
    / "championship_scanner_v1.json"
)

SHADOW_PATH = (
    SHADOW_DIRECTORY
    / "two_bot_shadow_controller_v1.json"
)

POSITION_MANAGER_PATH = (
    POSITION_MANAGER_DIRECTORY
    / "latest_summary.json"
)

LATEST_SUMMARY_PATH = (
    SIZER_DIRECTORY
    / "latest_summary.json"
)

LATEST_TEXT_PATH = (
    SIZER_DIRECTORY
    / "latest_summary.txt"
)

SIZING_DECISIONS_PATH = (
    SIZER_DIRECTORY
    / "position_sizing_decisions.csv"
)


SIZER_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

SIZER_HISTORY_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_STARTING_CAPITAL = 2000.0

MIN_POSITION_PERCENT = 2.5
MAX_POSITION_PERCENT = 35.0

MAX_ACCOUNT_RISK_PER_TRADE_PERCENT = 2.5

MAX_TOTAL_EXPOSURE_PERCENT = 80.0

MAX_PORTFOLIO_HEAT_PERCENT = 8.0

MIN_SCANNER_SCORE = 50.0
MAX_SCANNER_SCORE = 100.0


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class PositionSizingDecision:
    symbol: str
    side: str
    role: str
    strategy_name: str

    scanner_score: float
    atr_percent: float

    starting_capital: float

    base_allocation_dollars: float
    base_allocation_percent: float

    risk_score: int
    risk_level: str

    market_regime: str

    risk_multiplier: float
    regime_multiplier: float
    confidence_multiplier: float
    volatility_multiplier: float
    exposure_multiplier: float
    heat_multiplier: float

    raw_recommended_dollars: float

    stop_risk_limited_dollars: float

    final_recommended_dollars: float
    final_recommended_percent: float

    estimated_shares: float

    entry_reference_price: float
    suggested_stop: float | None
    suggested_target: float | None

    estimated_stop_risk_dollars: float
    estimated_stop_risk_percent_of_account: float

    decision: str
    reasons: list[str]

    shadow_only: bool


# ============================================================
# HELPERS
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

        return float(
            value
        )

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

        return int(
            value
        )

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


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# MULTIPLIERS
# ============================================================

def calculate_risk_multiplier(
    risk_score: int,
) -> float:

    if risk_score >= 80:
        return 0.0

    if risk_score >= 60:
        return 0.40

    if risk_score >= 40:
        return 0.65

    if risk_score >= 20:
        return 0.85

    return 1.00


def calculate_regime_multiplier(
    side: str,
    regime: str,
    permissions: dict[str, Any],
) -> float:

    side = normalize_side(
        side
    )

    regime = str(
        regime or ""
    ).upper()

    if side == "LONG":

        if not permissions.get(
            "allow_long_bot",
            False,
        ):
            return 0.0

        if regime == "BULL":
            return 1.00

        if regime == "NEUTRAL":
            return 0.75

        if regime == "BEAR":
            return 0.40

    if side == "SHORT":

        if not permissions.get(
            "allow_short_bot",
            False,
        ):
            return 0.0

        if regime == "BEAR":
            return 1.00

        if regime == "NEUTRAL":
            return 0.75

        if regime == "BULL":
            return 0.40

    return 0.50


def calculate_confidence_multiplier(
    scanner_score: float,
) -> float:

    if scanner_score <= 0:
        return 0.50

    normalized = (
        scanner_score
        - MIN_SCANNER_SCORE
    ) / (
        MAX_SCANNER_SCORE
        - MIN_SCANNER_SCORE
    )

    normalized = clamp(
        normalized,
        0.0,
        1.0,
    )

    return (
        0.60
        + (
            normalized
            * 0.40
        )
    )


def calculate_volatility_multiplier(
    atr_percent: float,
) -> float:

    if atr_percent <= 0:
        return 1.00

    if atr_percent <= 1.5:
        return 1.00

    if atr_percent <= 2.5:
        return 0.90

    if atr_percent <= 3.5:
        return 0.80

    if atr_percent <= 5.0:
        return 0.65

    return 0.50


def calculate_exposure_multiplier(
    current_exposure_percent: float,
) -> float:

    if current_exposure_percent >= 80:
        return 0.0

    if current_exposure_percent >= 65:
        return 0.40

    if current_exposure_percent >= 50:
        return 0.60

    if current_exposure_percent >= 35:
        return 0.80

    return 1.00


def calculate_heat_multiplier(
    current_heat_percent: float,
) -> float:

    if current_heat_percent >= 8:
        return 0.0

    if current_heat_percent >= 6:
        return 0.40

    if current_heat_percent >= 4:
        return 0.65

    if current_heat_percent >= 2:
        return 0.85

    return 1.00


# ============================================================
# PORTFOLIO ALLOCATION LOOKUP
# ============================================================

def build_allocation_lookup(
    portfolio: dict[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:

    lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    allocations = portfolio.get(
        "allocations",
        [],
    )

    if not isinstance(
        allocations,
        list,
    ):
        return lookup

    for allocation in allocations:

        if not isinstance(
            allocation,
            dict,
        ):
            continue

        role = str(
            allocation.get(
                "role",
                "",
            )
        ).upper()

        if role:

            lookup[
                role
            ] = allocation

    return lookup


# ============================================================
# SCANNER LOOKUP
# ============================================================

def build_scanner_lookup(
    scanner: dict[str, Any],
) -> dict[
    tuple[
        str,
        str,
    ],
    dict[str, Any],
]:

    lookup: dict[
        tuple[
            str,
            str,
        ],
        dict[str, Any],
    ] = {}

    for collection_name in [
        "top_longs",
        "top_shorts",
    ]:

        collection = scanner.get(
            collection_name,
            [],
        )

        if not isinstance(
            collection,
            list,
        ):
            continue

        for item in collection:

            if not isinstance(
                item,
                dict,
            ):
                continue

            symbol = str(
                item.get(
                    "symbol",
                    "",
                )
            ).upper()

            side = normalize_side(
                item.get(
                    "side",
                    "",
                )
            )

            if symbol:

                lookup[
                    (
                        symbol,
                        side,
                    )
                ] = item

    return lookup


# ============================================================
# STOP-BASED MAXIMUM SIZE
# ============================================================

def calculate_stop_risk_limited_size(
    starting_capital: float,
    entry_price: float,
    stop_price: float | None,
    side: str,
) -> float:

    if (
        starting_capital <= 0
        or entry_price <= 0
        or stop_price is None
        or stop_price <= 0
    ):

        return 0.0

    side = normalize_side(
        side
    )

    if side == "SHORT":

        stop_distance = (
            stop_price
            - entry_price
        )

    else:

        stop_distance = (
            entry_price
            - stop_price
        )

    if stop_distance <= 0:
        return 0.0

    maximum_risk_dollars = (
        starting_capital
        * MAX_ACCOUNT_RISK_PER_TRADE_PERCENT
        / 100
    )

    maximum_shares = (
        maximum_risk_dollars
        / stop_distance
    )

    return (
        maximum_shares
        * entry_price
    )


# ============================================================
# POSITION DECISION
# ============================================================

def size_proposal(
    proposal: dict[str, Any],
    starting_capital: float,
    allocation_lookup: dict[
        str,
        dict[str, Any],
    ],
    scanner_lookup: dict[
        tuple[
            str,
            str,
        ],
        dict[str, Any],
    ],
    risk_summary: dict[str, Any],
    market_regime: dict[str, Any],
) -> PositionSizingDecision:

    symbol = str(
        proposal.get(
            "symbol",
            "",
        )
    ).upper()

    side = normalize_side(
        proposal.get(
            "side",
            "",
        )
    )

    role = str(
        proposal.get(
            "role",
            side,
        )
    ).upper()

    strategy_name = str(
        proposal.get(
            "strategy_name",
            "UNKNOWN",
        )
    )

    scanner_item = scanner_lookup.get(
        (
            symbol,
            side,
        ),
        {},
    )

    scanner_score = safe_float(
        proposal.get(
            "scanner_score",
            scanner_item.get(
                "score",
                0.0,
            ),
        )
    )

    atr_percent = safe_float(
        scanner_item.get(
            "atr_percent",
            0.0,
        )
    )

    entry_reference_price = safe_float(
        scanner_item.get(
            "close",
            0.0,
        )
    )

    suggested_stop = (
        safe_float(
            proposal.get(
                "suggested_stop",
                scanner_item.get(
                    "suggested_stop"
                ),
            )
        )
        if (
            proposal.get(
                "suggested_stop"
            ) is not None
            or scanner_item.get(
                "suggested_stop"
            ) is not None
        )
        else None
    )

    suggested_target = (
        safe_float(
            proposal.get(
                "suggested_target",
                scanner_item.get(
                    "suggested_target"
                ),
            )
        )
        if (
            proposal.get(
                "suggested_target"
            ) is not None
            or scanner_item.get(
                "suggested_target"
            ) is not None
        )
        else None
    )

    allocation = allocation_lookup.get(
        role,
        {},
    )

    base_allocation_dollars = safe_float(
        allocation.get(
            "allocation_dollars",
            proposal.get(
                "proposed_dollars",
                0.0,
            ),
        )
    )

    base_allocation_percent = (
        base_allocation_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    risk_score = safe_int(
        risk_summary.get(
            "risk_score",
            100,
        ),
        100,
    )

    risk_level = str(
        risk_summary.get(
            "risk_level",
            "UNKNOWN",
        )
    ).upper()

    risk_acceptance_status = str(
        risk_summary.get(
            "trade_acceptance_status",
            "BLOCK_NEW_SHADOW_TRADES",
        )
    ).upper()

    risk_metrics = risk_summary.get(
        "portfolio_metrics",
        {},
    )

    current_exposure_percent = safe_float(
        risk_metrics.get(
            "total_exposure_percent",
            0.0,
        )
    )

    current_heat_percent = safe_float(
        risk_metrics.get(
            "portfolio_heat_percent",
            0.0,
        )
    )

    regime = str(
        market_regime.get(
            "regime",
            "UNKNOWN",
        )
    ).upper()

    permissions = market_regime.get(
        "permissions",
        {},
    )

    if not isinstance(
        permissions,
        dict,
    ):
        permissions = {}

    risk_multiplier = (
        calculate_risk_multiplier(
            risk_score
        )
    )

    regime_multiplier = (
        calculate_regime_multiplier(
            side=side,
            regime=regime,
            permissions=permissions,
        )
    )

    confidence_multiplier = (
        calculate_confidence_multiplier(
            scanner_score
        )
    )

    volatility_multiplier = (
        calculate_volatility_multiplier(
            atr_percent
        )
    )

    exposure_multiplier = (
        calculate_exposure_multiplier(
            current_exposure_percent
        )
    )

    heat_multiplier = (
        calculate_heat_multiplier(
            current_heat_percent
        )
    )

    raw_recommended_dollars = (
        base_allocation_dollars
        * risk_multiplier
        * regime_multiplier
        * confidence_multiplier
        * volatility_multiplier
        * exposure_multiplier
        * heat_multiplier
    )

    stop_risk_limited_dollars = (
        calculate_stop_risk_limited_size(
            starting_capital=starting_capital,
            entry_price=entry_reference_price,
            stop_price=suggested_stop,
            side=side,
        )
    )

    maximum_position_dollars = (
        starting_capital
        * MAX_POSITION_PERCENT
        / 100
    )

    minimum_position_dollars = (
        starting_capital
        * MIN_POSITION_PERCENT
        / 100
    )

    available_exposure_percent = max(
        MAX_TOTAL_EXPOSURE_PERCENT
        - current_exposure_percent,
        0.0,
    )

    available_exposure_dollars = (
        starting_capital
        * available_exposure_percent
        / 100
    )

    candidate_sizes = [
        raw_recommended_dollars,
        maximum_position_dollars,
        available_exposure_dollars,
    ]

    if (
        stop_risk_limited_dollars
        > 0
    ):

        candidate_sizes.append(
            stop_risk_limited_dollars
        )

    final_recommended_dollars = max(
        min(
            candidate_sizes
        ),
        0.0,
    )

    reasons: list[str] = []

    decision = (
        "SIZE_CALCULATED"
    )

    if (
        risk_acceptance_status
        == "BLOCK_NEW_SHADOW_TRADES"
    ):

        final_recommended_dollars = 0.0

        decision = (
            "NO_POSITION"
        )

        reasons.append(
            "Risk Manager is blocking "
            "new shadow trades."
        )

    elif (
        risk_acceptance_status
        == "REVIEW_BEFORE_NEW_SHADOW_TRADE"
    ):

        final_recommended_dollars = min(
            final_recommended_dollars,
            base_allocation_dollars
            * 0.50,
        )

        decision = (
            "REDUCED_REVIEW_SIZE"
        )

        reasons.append(
            "Risk Manager requires review, "
            "so the size was reduced."
        )

    if regime_multiplier <= 0:

        final_recommended_dollars = 0.0

        decision = (
            "NO_POSITION"
        )

        reasons.append(
            "Market regime permissions "
            "do not allow this trade direction."
        )

    if exposure_multiplier <= 0:

        final_recommended_dollars = 0.0

        decision = (
            "NO_POSITION"
        )

        reasons.append(
            "Portfolio exposure limit "
            "has been reached."
        )

    if heat_multiplier <= 0:

        final_recommended_dollars = 0.0

        decision = (
            "NO_POSITION"
        )

        reasons.append(
            "Portfolio heat limit "
            "has been reached."
        )

    if (
        final_recommended_dollars
        > 0
        and final_recommended_dollars
        < minimum_position_dollars
    ):

        final_recommended_dollars = 0.0

        decision = (
            "NO_POSITION"
        )

        reasons.append(
            "Calculated position is below "
            "the minimum useful position size."
        )

    if (
        scanner_score
        < MIN_SCANNER_SCORE
    ):

        final_recommended_dollars = 0.0

        decision = (
            "NO_POSITION"
        )

        reasons.append(
            "Scanner confidence is below "
            "the minimum threshold."
        )

    if not reasons:

        reasons.append(
            "Position size adjusted using risk, "
            "market regime, scanner confidence, "
            "volatility, exposure, and portfolio heat."
        )

    final_recommended_percent = (
        final_recommended_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    estimated_shares = (
        final_recommended_dollars
        / entry_reference_price
        if (
            final_recommended_dollars > 0
            and entry_reference_price > 0
        )
        else 0.0
    )

    estimated_stop_risk_dollars = 0.0

    if (
        estimated_shares > 0
        and suggested_stop is not None
        and entry_reference_price > 0
    ):

        if side == "SHORT":

            stop_distance = max(
                suggested_stop
                - entry_reference_price,
                0.0,
            )

        else:

            stop_distance = max(
                entry_reference_price
                - suggested_stop,
                0.0,
            )

        estimated_stop_risk_dollars = (
            stop_distance
            * estimated_shares
        )

    estimated_stop_risk_percent = (
        estimated_stop_risk_dollars
        / starting_capital
        * 100
        if starting_capital > 0
        else 0.0
    )

    return PositionSizingDecision(

        symbol=symbol,

        side=side,

        role=role,

        strategy_name=(
            strategy_name
        ),

        scanner_score=round(
            scanner_score,
            4,
        ),

        atr_percent=round(
            atr_percent,
            4,
        ),

        starting_capital=round(
            starting_capital,
            2,
        ),

        base_allocation_dollars=round(
            base_allocation_dollars,
            2,
        ),

        base_allocation_percent=round(
            base_allocation_percent,
            4,
        ),

        risk_score=risk_score,

        risk_level=(
            risk_level
        ),

        market_regime=(
            regime
        ),

        risk_multiplier=round(
            risk_multiplier,
            4,
        ),

        regime_multiplier=round(
            regime_multiplier,
            4,
        ),

        confidence_multiplier=round(
            confidence_multiplier,
            4,
        ),

        volatility_multiplier=round(
            volatility_multiplier,
            4,
        ),

        exposure_multiplier=round(
            exposure_multiplier,
            4,
        ),

        heat_multiplier=round(
            heat_multiplier,
            4,
        ),

        raw_recommended_dollars=round(
            raw_recommended_dollars,
            2,
        ),

        stop_risk_limited_dollars=round(
            stop_risk_limited_dollars,
            2,
        ),

        final_recommended_dollars=round(
            final_recommended_dollars,
            2,
        ),

        final_recommended_percent=round(
            final_recommended_percent,
            4,
        ),

        estimated_shares=round(
            estimated_shares,
            8,
        ),

        entry_reference_price=round(
            entry_reference_price,
            4,
        ),

        suggested_stop=(
            round(
                suggested_stop,
                4,
            )
            if suggested_stop
            is not None
            else None
        ),

        suggested_target=(
            round(
                suggested_target,
                4,
            )
            if suggested_target
            is not None
            else None
        ),

        estimated_stop_risk_dollars=round(
            estimated_stop_risk_dollars,
            2,
        ),

        estimated_stop_risk_percent_of_account=round(
            estimated_stop_risk_percent,
            4,
        ),

        decision=decision,

        reasons=reasons,

        shadow_only=True,
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    payload: dict[str, Any],
) -> None:

    lines = [
        "INTELLIGENT POSITION SIZER V1",
        "=" * 38,

        (
            "Generated at: "
            f"{payload['generated_at']}"
        ),

        (
            "Starting capital: "
            f"${payload['starting_capital']:.2f}"
        ),

        (
            "Proposal count: "
            f"{payload['proposal_count']}"
        ),

        (
            "Sized positions: "
            f"{payload['positive_size_count']}"
        ),

        (
            "Zero-size positions: "
            f"{payload['zero_size_count']}"
        ),

        "",

        "SIZING DECISIONS",
        "-" * 24,
    ]

    decisions = payload.get(
        "decisions",
        [],
    )

    if decisions:

        for decision in decisions:

            lines.append(
                (
                    f"{decision['symbol']} "
                    f"{decision['side']} | "
                    f"{decision['decision']} | "
                    f"${decision['final_recommended_dollars']:.2f} "
                    f"({decision['final_recommended_percent']:.2f}%)"
                )
            )

            lines.append(
                (
                    "  Multipliers: "
                    f"risk={decision['risk_multiplier']:.2f}, "
                    f"regime={decision['regime_multiplier']:.2f}, "
                    f"confidence={decision['confidence_multiplier']:.2f}, "
                    f"volatility={decision['volatility_multiplier']:.2f}, "
                    f"exposure={decision['exposure_multiplier']:.2f}, "
                    f"heat={decision['heat_multiplier']:.2f}"
                )
            )

            for reason in decision[
                "reasons"
            ]:

                lines.append(
                    f"  - {reason}"
                )

    else:

        lines.append(
            "No shadow proposals were available."
        )

    lines.extend(
        [
            "",
            "SAFETY",
            "-" * 24,

            "Research sizing only.",
            "Trade ledger modified: False",
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
# MAIN
# ============================================================

def main() -> None:

    portfolio = load_json(
        PORTFOLIO_PATH,
        {},
    )

    risk_summary = load_json(
        RISK_PATH,
        {},
    )

    market_regime = load_json(
        MARKET_REGIME_PATH,
        {},
    )

    scanner = load_json(
        SCANNER_PATH,
        {},
    )

    shadow = load_json(
        SHADOW_PATH,
        {},
    )

    position_manager = load_json(
        POSITION_MANAGER_PATH,
        {},
    )

    starting_capital = safe_float(
        risk_summary.get(
            "starting_capital",
            position_manager.get(
                "starting_capital",
                portfolio.get(
                    "starting_capital",
                    DEFAULT_STARTING_CAPITAL,
                ),
            ),
        ),
        DEFAULT_STARTING_CAPITAL,
    )

    allocations = (
        build_allocation_lookup(
            portfolio
        )
    )

    scanner_lookup = (
        build_scanner_lookup(
            scanner
        )
    )

    proposals = shadow.get(
        "proposals",
        [],
    )

    if not isinstance(
        proposals,
        list,
    ):

        proposals = []

    decisions: list[
        PositionSizingDecision
    ] = []

    for proposal in proposals:

        if not isinstance(
            proposal,
            dict,
        ):
            continue

        decision = (
            size_proposal(
                proposal=proposal,
                starting_capital=(
                    starting_capital
                ),
                allocation_lookup=(
                    allocations
                ),
                scanner_lookup=(
                    scanner_lookup
                ),
                risk_summary=(
                    risk_summary
                ),
                market_regime=(
                    market_regime
                ),
            )
        )

        decisions.append(
            decision
        )

    positive_size_count = sum(
        decision.final_recommended_dollars
        > 0
        for decision in decisions
    )

    zero_size_count = (
        len(
            decisions
        )
        - positive_size_count
    )

    total_recommended_dollars = sum(
        decision.final_recommended_dollars
        for decision in decisions
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

        "proposal_count": len(
            decisions
        ),

        "positive_size_count": (
            positive_size_count
        ),

        "zero_size_count": (
            zero_size_count
        ),

        "total_recommended_dollars": round(
            total_recommended_dollars,
            2,
        ),

        "total_recommended_percent": round(
            (
                total_recommended_dollars
                / starting_capital
                * 100
            )
            if starting_capital > 0
            else 0.0,
            4,
        ),

        "configuration": {

            "min_position_percent": (
                MIN_POSITION_PERCENT
            ),

            "max_position_percent": (
                MAX_POSITION_PERCENT
            ),

            "max_account_risk_per_trade_percent": (
                MAX_ACCOUNT_RISK_PER_TRADE_PERCENT
            ),

            "max_total_exposure_percent": (
                MAX_TOTAL_EXPOSURE_PERCENT
            ),

            "max_portfolio_heat_percent": (
                MAX_PORTFOLIO_HEAT_PERCENT
            ),

            "min_scanner_score": (
                MIN_SCANNER_SCORE
            ),
        },

        "decisions": [
            asdict(
                decision
            )
            for decision
            in decisions
        ],

        "shadow_mode": True,

        "research_only": True,

        "trade_ledger_modified": False,

        "trading_client_created": False,

        "market_request_made": False,

        "order_submitted": False,
    }

    save_json(
        LATEST_SUMMARY_PATH,
        payload,
    )

    history_path = (
        SIZER_HISTORY_DIRECTORY
        / (
            datetime.now()
            .astimezone()
            .strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        )
    )

    save_json(
        history_path,
        payload,
    )

    csv_rows = []

    for decision in decisions:

        row = asdict(
            decision
        )

        row[
            "reasons"
        ] = " | ".join(
            decision.reasons
        )

        csv_rows.append(
            row
        )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        SIZING_DECISIONS_PATH,
        index=False,
    )

    write_text_report(
        payload
    )

    print(
        "Intelligent Position Sizer v1"
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
        f"Sizing decisions: "
        f"{SIZING_DECISIONS_PATH}"
    )

    print(
        "Research sizing only."
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