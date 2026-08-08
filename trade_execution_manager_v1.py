"""
Trade Execution Manager v1

Research-only trade decision engine for the gold-trading-bot project.

This module:
- reads shadow trade proposals,
- reads Risk Manager v2,
- reads Portfolio Commander,
- reads Position Manager v2,
- checks duplicate open positions,
- checks current risk approval,
- checks proposed allocation,
- approves or rejects proposed shadow trades,
- writes detailed decision reports,
- never modifies the trade ledger,
- never creates a broker client,
- never submits an order.

Run:
    python trade_execution_manager_v1.py
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

SHADOW_DIRECTORY = REPORTS_DIRECTORY / "shadow"
RISK_DIRECTORY = REPORTS_DIRECTORY / "risk_manager"
PORTFOLIO_DIRECTORY = REPORTS_DIRECTORY / "portfolio"
POSITION_MANAGER_DIRECTORY = REPORTS_DIRECTORY / "position_manager"
PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"

EXECUTION_DIRECTORY = REPORTS_DIRECTORY / "trade_execution_manager"
EXECUTION_HISTORY_DIRECTORY = EXECUTION_DIRECTORY / "history"

SHADOW_CONTROLLER_PATH = (
    SHADOW_DIRECTORY
    / "two_bot_shadow_controller_v1.json"
)

RISK_MANAGER_PATH = (
    RISK_DIRECTORY
    / "latest_summary.json"
)

PORTFOLIO_COMMANDER_PATH = (
    PORTFOLIO_DIRECTORY
    / "portfolio_commander_v1.json"
)

POSITION_MANAGER_PATH = (
    POSITION_MANAGER_DIRECTORY
    / "latest_summary.json"
)

TRADE_LEDGER_PATH = (
    PERFORMANCE_DIRECTORY
    / "trade_ledger.json"
)

LATEST_SUMMARY_PATH = (
    EXECUTION_DIRECTORY
    / "latest_summary.json"
)

LATEST_TEXT_PATH = (
    EXECUTION_DIRECTORY
    / "latest_summary.txt"
)

DECISIONS_CSV_PATH = (
    EXECUTION_DIRECTORY
    / "trade_decisions.csv"
)

EXECUTION_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

EXECUTION_HISTORY_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class TradeDecision:
    proposal_index: int
    symbol: str
    side: str
    role: str
    strategy_name: str
    proposed_dollars: float
    proposed_percent_of_capital: float
    scanner_score: float
    suggested_stop: float | None
    suggested_target: float | None
    approved: bool
    decision: str
    reasons: list[str]
    risk_score: int
    risk_level: str
    risk_acceptance_status: str
    duplicate_open_position: bool
    allocation_available: bool
    allocation_limit_dollars: float
    allocation_limit_percent: float
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


# ============================================================
# OPEN POSITION LOOKUP
# ============================================================

def build_open_position_keys(
    trade_ledger: Any,
    position_manager: dict[str, Any],
) -> set[
    tuple[
        str,
        str,
    ]
]:

    keys: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    if isinstance(
        trade_ledger,
        list,
    ):

        for trade in trade_ledger:

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

            symbol = str(
                trade.get(
                    "symbol",
                    "",
                )
            ).upper()

            side = normalize_side(
                trade.get(
                    "side",
                    "",
                )
            )

            if symbol:
                keys.add(
                    (
                        symbol,
                        side,
                    )
                )

    manager_positions = (
        position_manager.get(
            "open_positions",
            [],
        )
    )

    if isinstance(
        manager_positions,
        list,
    ):

        for trade in manager_positions:

            if not isinstance(
                trade,
                dict,
            ):
                continue

            symbol = str(
                trade.get(
                    "symbol",
                    "",
                )
            ).upper()

            side = normalize_side(
                trade.get(
                    "side",
                    "",
                )
            )

            if symbol:
                keys.add(
                    (
                        symbol,
                        side,
                    )
                )

    return keys


# ============================================================
# ALLOCATION LOOKUP
# ============================================================

def build_allocation_lookup(
    portfolio_commander: dict[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:

    lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    allocations = (
        portfolio_commander.get(
            "allocations",
            [],
        )
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

        if not role:
            continue

        lookup[
            role
        ] = allocation

    return lookup


# ============================================================
# DECISION ENGINE
# ============================================================

def evaluate_proposal(
    proposal_index: int,
    proposal: dict[str, Any],
    starting_capital: float,
    risk_summary: dict[str, Any],
    allocation_lookup: dict[
        str,
        dict[str, Any],
    ],
    open_position_keys: set[
        tuple[
            str,
            str,
        ]
    ],
) -> TradeDecision:

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

    proposed_dollars = safe_float(
        proposal.get(
            "proposed_dollars",
            0.0,
        )
    )

    scanner_score = safe_float(
        proposal.get(
            "scanner_score",
            0.0,
        )
    )

    suggested_stop = (
        safe_float(
            proposal.get(
                "suggested_stop"
            )
        )
        if proposal.get(
            "suggested_stop"
        ) is not None
        else None
    )

    suggested_target = (
        safe_float(
            proposal.get(
                "suggested_target"
            )
        )
        if proposal.get(
            "suggested_target"
        ) is not None
        else None
    )

    proposed_percent = (
        proposed_dollars
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
        default=100,
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

    duplicate_open_position = (
        (
            symbol,
            side,
        )
        in open_position_keys
    )

    allocation = (
        allocation_lookup.get(
            role,
            {}
        )
    )

    allocation_limit_dollars = safe_float(
        allocation.get(
            "allocation_dollars",
            0.0,
        )
    )

    allocation_limit_percent = safe_float(
        allocation.get(
            "allocation_percent",
            0.0,
        )
    )

    allocation_available = (
        allocation_limit_dollars > 0
    )

    reasons: list[str] = []

    approved = True

    if not symbol:

        approved = False

        reasons.append(
            "Proposal has no symbol."
        )

    if side not in {
        "LONG",
        "SHORT",
    }:

        approved = False

        reasons.append(
            "Proposal side is not LONG or SHORT."
        )

    if proposed_dollars <= 0:

        approved = False

        reasons.append(
            "Proposed dollars must be greater than zero."
        )

    if duplicate_open_position:

        approved = False

        reasons.append(
            "An open position already exists for this "
            "symbol and side."
        )

    if (
        risk_acceptance_status
        == "BLOCK_NEW_SHADOW_TRADES"
    ):

        approved = False

        reasons.append(
            "Risk Manager is blocking new shadow trades."
        )

    elif (
        risk_acceptance_status
        == "REVIEW_BEFORE_NEW_SHADOW_TRADE"
    ):

        approved = False

        reasons.append(
            "Risk Manager requires review before "
            "new shadow trades."
        )

    if not allocation_available:

        approved = False

        reasons.append(
            "No Portfolio Commander allocation is "
            f"available for role {role}."
        )

    if (
        allocation_limit_dollars > 0
        and proposed_dollars
        > allocation_limit_dollars
    ):

        approved = False

        reasons.append(
            "Proposal exceeds Portfolio Commander "
            f"allocation of "
            f"${allocation_limit_dollars:.2f}."
        )

    if suggested_stop is None:

        approved = False

        reasons.append(
            "Proposal has no suggested stop."
        )

    if suggested_target is None:

        approved = False

        reasons.append(
            "Proposal has no suggested target."
        )

    if risk_score >= 80:

        approved = False

        reasons.append(
            "Risk score is 80 or higher."
        )

    if approved:

        reasons.append(
            "Proposal passed duplicate, risk, allocation, "
            "stop, and target checks."
        )

        decision = (
            "APPROVED_FOR_SHADOW_RESEARCH"
        )

    else:

        decision = (
            "REJECTED"
        )

    return TradeDecision(
        proposal_index=proposal_index,
        symbol=symbol,
        side=side,
        role=role,
        strategy_name=strategy_name,
        proposed_dollars=round(
            proposed_dollars,
            2,
        ),
        proposed_percent_of_capital=round(
            proposed_percent,
            4,
        ),
        scanner_score=round(
            scanner_score,
            4,
        ),
        suggested_stop=(
            round(
                suggested_stop,
                4,
            )
            if suggested_stop is not None
            else None
        ),
        suggested_target=(
            round(
                suggested_target,
                4,
            )
            if suggested_target is not None
            else None
        ),
        approved=approved,
        decision=decision,
        reasons=reasons,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_acceptance_status=(
            risk_acceptance_status
        ),
        duplicate_open_position=(
            duplicate_open_position
        ),
        allocation_available=(
            allocation_available
        ),
        allocation_limit_dollars=round(
            allocation_limit_dollars,
            2,
        ),
        allocation_limit_percent=round(
            allocation_limit_percent,
            4,
        ),
        shadow_only=True,
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    payload: dict[str, Any],
) -> None:

    lines = [
        "TRADE EXECUTION MANAGER V1",
        "=" * 36,

        (
            "Generated at: "
            f"{payload['generated_at']}"
        ),

        (
            "Proposal count: "
            f"{payload['proposal_count']}"
        ),

        (
            "Approved count: "
            f"{payload['approved_count']}"
        ),

        (
            "Rejected count: "
            f"{payload['rejected_count']}"
        ),

        (
            "Risk score: "
            f"{payload['risk_score']}"
        ),

        (
            "Risk level: "
            f"{payload['risk_level']}"
        ),

        (
            "Risk acceptance: "
            f"{payload['risk_acceptance_status']}"
        ),

        "",

        "DECISIONS",
        "-" * 20,
    ]

    if payload[
        "decisions"
    ]:

        for decision in payload[
            "decisions"
        ]:

            lines.append(
                (
                    f"{decision['decision']} | "
                    f"{decision['symbol']} | "
                    f"{decision['side']} | "
                    f"{decision['strategy_name']} | "
                    f"${decision['proposed_dollars']:.2f}"
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
            "No trade proposals were available."
        )

    lines.extend(
        [
            "",
            "SAFETY",
            "-" * 20,

            "Research preview only.",
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

    shadow_controller = load_json(
        SHADOW_CONTROLLER_PATH,
        {},
    )

    risk_summary = load_json(
        RISK_MANAGER_PATH,
        {},
    )

    portfolio_commander = load_json(
        PORTFOLIO_COMMANDER_PATH,
        {},
    )

    position_manager = load_json(
        POSITION_MANAGER_PATH,
        {},
    )

    trade_ledger = load_json(
        TRADE_LEDGER_PATH,
        [],
    )

    starting_capital = safe_float(
        risk_summary.get(
            "starting_capital",
            portfolio_commander.get(
                "starting_capital",
                2000.0,
            ),
        ),
        default=2000.0,
    )

    proposals = (
        shadow_controller.get(
            "proposals",
            [],
        )
    )

    if not isinstance(
        proposals,
        list,
    ):
        proposals = []

    allocation_lookup = (
        build_allocation_lookup(
            portfolio_commander
        )
    )

    open_position_keys = (
        build_open_position_keys(
            trade_ledger=trade_ledger,
            position_manager=(
                position_manager
            ),
        )
    )

    decisions: list[
        TradeDecision
    ] = []

    for (
        proposal_index,
        proposal,
    ) in enumerate(
        proposals,
        start=1,
    ):

        if not isinstance(
            proposal,
            dict,
        ):
            continue

        decision = evaluate_proposal(
            proposal_index=proposal_index,
            proposal=proposal,
            starting_capital=(
                starting_capital
            ),
            risk_summary=(
                risk_summary
            ),
            allocation_lookup=(
                allocation_lookup
            ),
            open_position_keys=(
                open_position_keys
            ),
        )

        decisions.append(
            decision
        )

    approved = [
        decision
        for decision in decisions
        if decision.approved
    ]

    rejected = [
        decision
        for decision in decisions
        if not decision.approved
    ]

    risk_score = safe_int(
        risk_summary.get(
            "risk_score",
            100,
        ),
        default=100,
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

        "approved_count": len(
            approved
        ),

        "rejected_count": len(
            rejected
        ),

        "risk_score": (
            risk_score
        ),

        "risk_level": (
            risk_level
        ),

        "risk_acceptance_status": (
            risk_acceptance_status
        ),

        "open_position_keys": [
            {
                "symbol": symbol,
                "side": side,
            }
            for (
                symbol,
                side,
            ) in sorted(
                open_position_keys
            )
        ],

        "decisions": [
            asdict(
                decision
            )
            for decision
            in decisions
        ],

        "approved_proposals": [
            asdict(
                decision
            )
            for decision
            in approved
        ],

        "rejected_proposals": [
            asdict(
                decision
            )
            for decision
            in rejected
        ],

        "shadow_mode": True,

        "preview_only": True,

        "trade_ledger_modified": False,

        "trading_client_created": False,

        "market_request_made": False,

        "order_submitted": False,
    }

    save_json(
        LATEST_SUMMARY_PATH,
        payload,
    )

    save_json(
        EXECUTION_HISTORY_DIRECTORY
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

    csv_rows: list[
        dict[str, Any]
    ] = []

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
        DECISIONS_CSV_PATH,
        index=False,
    )

    write_text_report(
        payload
    )

    print(
        "Trade Execution Manager v1"
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
        f"Decision CSV: "
        f"{DECISIONS_CSV_PATH}"
    )

    print(
        "Research preview only."
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