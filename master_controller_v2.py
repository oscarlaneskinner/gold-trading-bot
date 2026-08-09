"""
Master Controller v2

Research-only orchestration layer for the gold-trading-bot platform.

Version 2 integrates the newer intelligence modules:

1. Market Regime Lab
2. Strategy Hall of Fame
3. Championship Scanner
4. Two-Bot Shadow Controller
5. Strategy Learning Engine
6. Portfolio Commander
7. Intelligent Position Sizer
8. Shadow Performance Tracker
9. Position Manager v2 Preview
10. Shadow Position Valuation
11. Performance Analytics
12. Risk Manager v2
13. Trade Execution Manager v1 Preview

IMPORTANT SAFETY RULES

This controller:
- stays in shadow/research mode,
- does NOT call Position Manager with --apply,
- does NOT create a trading client,
- does NOT submit paper orders,
- does NOT submit live orders.

Run with existing local data:

    python master_controller_v2.py --skip-data-refresh

Run with data refresh:

    python master_controller_v2.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

MASTER_DIRECTORY = (
    REPORTS_DIRECTORY
    / "master_controller_v2"
)

MASTER_HISTORY_DIRECTORY = (
    MASTER_DIRECTORY
    / "history"
)

LATEST_JSON_PATH = (
    MASTER_DIRECTORY
    / "latest_summary.json"
)

LATEST_TEXT_PATH = (
    MASTER_DIRECTORY
    / "latest_summary.txt"
)

MASTER_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_HISTORY_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ControllerStep:
    name: str
    command: list[str]
    required: bool = True
    enabled: bool = True


@dataclass
class StepResult:
    name: str
    command: list[str]
    required: bool
    status: str
    returncode: int | None
    started_at: str
    finished_at: str
    duration_seconds: float
    stdout: str
    stderr: str
    skipped_reason: str | None = None


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


def script_exists(
    command: list[str],
) -> bool:

    if len(
        command
    ) < 2:
        return True

    script_name = (
        command[
            1
        ]
    )

    if script_name.startswith(
        "-"
    ):
        return True

    return (
        PROJECT_ROOT
        / script_name
    ).exists()


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(
    skip_data_refresh: bool,
) -> list[
    ControllerStep
]:

    python = (
        sys.executable
    )

    return [

        # ----------------------------------------------------
        # 1. MARKET DATA
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Download Market Research Data"
            ),

            command=[
                python,
                "download_multi_market_research_data.py",
            ],

            required=True,

            enabled=(
                not skip_data_refresh
            ),
        ),

        # ----------------------------------------------------
        # 2. MARKET REGIME
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Market Regime Lab"
            ),

            command=[
                python,
                "market_regime_lab_v1.py",
                "--data-dir",
                "data",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 3. STRATEGY HALL OF FAME
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Strategy Hall of Fame"
            ),

            command=[
                python,
                "strategy_hall_of_fame_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 4. MARKET SCANNER
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Championship Market Scanner"
            ),

            command=[
                python,
                "championship_market_scanner_v1.py",
                "--data-dir",
                "data",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 5. SHADOW PROPOSALS
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Two-Bot Shadow Controller"
            ),

            command=[
                python,
                "two_bot_shadow_controller_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 6. STRATEGY LEARNING
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Strategy Learning Engine v1"
            ),

            command=[
                python,
                "strategy_learning_engine_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 7. PORTFOLIO ALLOCATION
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Portfolio Commander"
            ),

            command=[
                python,
                "portfolio_commander_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 8. INTELLIGENT POSITION SIZE
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Intelligent Position Sizer v1"
            ),

            command=[
                python,
                "intelligent_position_sizer_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 9. SHADOW PERFORMANCE TRACKER
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Shadow Performance Tracker"
            ),

            command=[
                python,
                "shadow_performance_tracker_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 10. POSITION MANAGER PREVIEW
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Position Manager v2 Preview"
            ),

            command=[
                python,
                "position_manager_v2.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 11. POSITION VALUATION
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Shadow Position Valuation"
            ),

            command=[
                python,
                "shadow_position_valuation_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 12. PERFORMANCE ANALYTICS
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Performance Analytics v1"
            ),

            command=[
                python,
                "performance_analytics_v1.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 13. RISK MANAGER
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Risk Manager v2"
            ),

            command=[
                python,
                "risk_manager_v2.py",
            ],

            required=True,
        ),

        # ----------------------------------------------------
        # 14. EXECUTION DECISION PREVIEW
        # ----------------------------------------------------

        ControllerStep(
            name=(
                "Trade Execution Manager v1 Preview"
            ),

            command=[
                python,
                "trade_execution_manager_v1.py",
            ],

            required=True,
        ),
    ]


# ============================================================
# RUN SINGLE STEP
# ============================================================

def run_step(
    step: ControllerStep,
) -> StepResult:

    started = (
        datetime.now()
        .astimezone()
    )

    # --------------------------------------------------------
    # SKIPPED
    # --------------------------------------------------------

    if not step.enabled:

        finished = (
            datetime.now()
            .astimezone()
        )

        return StepResult(
            name=(
                step.name
            ),

            command=(
                step.command
            ),

            required=(
                step.required
            ),

            status="SKIPPED",

            returncode=None,

            started_at=(
                started.isoformat()
            ),

            finished_at=(
                finished.isoformat()
            ),

            duration_seconds=0.0,

            stdout="",

            stderr="",

            skipped_reason=(
                "Disabled by command-line option."
            ),
        )

    # --------------------------------------------------------
    # SCRIPT MISSING
    # --------------------------------------------------------

    if not script_exists(
        step.command
    ):

        finished = (
            datetime.now()
            .astimezone()
        )

        return StepResult(
            name=(
                step.name
            ),

            command=(
                step.command
            ),

            required=(
                step.required
            ),

            status="FAILED",

            returncode=None,

            started_at=(
                started.isoformat()
            ),

            finished_at=(
                finished.isoformat()
            ),

            duration_seconds=0.0,

            stdout="",

            stderr=(
                "Script not found: "
                f"{step.command[1]}"
            ),

            skipped_reason=None,
        )

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    completed = (
        subprocess.run(
            step.command,

            cwd=(
                PROJECT_ROOT
            ),

            capture_output=True,

            text=True,

            encoding="utf-8",

            errors="replace",

            check=False,
        )
    )

    finished = (
        datetime.now()
        .astimezone()
    )

    duration = (
        finished
        - started
    ).total_seconds()

    return StepResult(
        name=(
            step.name
        ),

        command=(
            step.command
        ),

        required=(
            step.required
        ),

        status=(
            "PASSED"
            if completed.returncode
            == 0
            else "FAILED"
        ),

        returncode=(
            completed.returncode
        ),

        started_at=(
            started.isoformat()
        ),

        finished_at=(
            finished.isoformat()
        ),

        duration_seconds=round(
            duration,
            3,
        ),

        stdout=(
            completed.stdout
        ),

        stderr=(
            completed.stderr
        ),

        skipped_reason=None,
    )


# ============================================================
# PLATFORM SNAPSHOT
# ============================================================

def build_platform_snapshot() -> dict[
    str,
    Any,
]:

    market_regime = (
        load_json(
            REPORTS_DIRECTORY
            / "market_regime"
            / "market_regime_lab_v1.json",
            {},
        )
    )

    hall_of_fame = (
        load_json(
            REPORTS_DIRECTORY
            / "hall_of_fame"
            / "strategy_hall_of_fame.json",
            {},
        )
    )

    scanner = (
        load_json(
            REPORTS_DIRECTORY
            / "scanner"
            / "championship_scanner_v1.json",
            {},
        )
    )

    shadow = (
        load_json(
            REPORTS_DIRECTORY
            / "shadow"
            / "two_bot_shadow_controller_v1.json",
            {},
        )
    )

    learning = (
        load_json(
            REPORTS_DIRECTORY
            / "strategy_learning"
            / "latest_summary.json",
            {},
        )
    )

    portfolio = (
        load_json(
            REPORTS_DIRECTORY
            / "portfolio"
            / "portfolio_commander_v1.json",
            {},
        )
    )

    position_sizer = (
        load_json(
            REPORTS_DIRECTORY
            / "position_sizer"
            / "latest_summary.json",
            {},
        )
    )

    performance = (
        load_json(
            REPORTS_DIRECTORY
            / "performance"
            / "latest_summary.json",
            {},
        )
    )

    position_manager = (
        load_json(
            REPORTS_DIRECTORY
            / "position_manager"
            / "latest_summary.json",
            {},
        )
    )

    valuation = (
        load_json(
            REPORTS_DIRECTORY
            / "position_valuation"
            / "latest_summary.json",
            {},
        )
    )

    analytics = (
        load_json(
            REPORTS_DIRECTORY
            / "performance_analytics"
            / "latest_summary.json",
            {},
        )
    )

    risk = (
        load_json(
            REPORTS_DIRECTORY
            / "risk_manager"
            / "latest_summary.json",
            {},
        )
    )

    execution = (
        load_json(
            REPORTS_DIRECTORY
            / "trade_execution_manager"
            / "latest_summary.json",
            {},
        )
    )

    # --------------------------------------------------------
    # TOP SCANNER RESULTS
    # --------------------------------------------------------

    top_longs = (
        scanner.get(
            "top_longs",
            [],
        )
    )

    top_shorts = (
        scanner.get(
            "top_shorts",
            [],
        )
    )

    top_long = (
        top_longs[0]
        if isinstance(
            top_longs,
            list,
        )
        and top_longs
        else None
    )

    top_short = (
        top_shorts[0]
        if isinstance(
            top_shorts,
            list,
        )
        and top_shorts
        else None
    )

    # --------------------------------------------------------
    # POSITION SIZER TOP DECISION
    # --------------------------------------------------------

    sizing_decisions = (
        position_sizer.get(
            "decisions",
            [],
        )
    )

    top_sizing_decision = (
        sizing_decisions[0]
        if isinstance(
            sizing_decisions,
            list,
        )
        and sizing_decisions
        else None
    )

    # --------------------------------------------------------
    # LEARNING TOP STRATEGY
    # --------------------------------------------------------

    learning_top_strategy = (
        learning.get(
            "top_strategy"
        )
    )

    learning_top_score = (
        learning.get(
            "top_learning_score",
            0.0,
        )
    )

    # --------------------------------------------------------
    # RETURN SNAPSHOT
    # --------------------------------------------------------

    return {

        "market_regime": (
            market_regime.get(
                "regime",
                "UNKNOWN",
            )
        ),

        "market_permissions": (
            market_regime.get(
                "permissions",
                {},
            )
        ),

        "hall_of_fame_strategy_count": (
            hall_of_fame.get(
                "strategy_count",
                0,
            )
        ),

        "hall_of_fame_scanner_records_excluded": (
            hall_of_fame.get(
                "scanner_records_excluded",
                False,
            )
        ),

        "scanner_top_long": (
            top_long
        ),

        "scanner_top_short": (
            top_short
        ),

        "shadow_proposal_count": (
            shadow.get(
                "proposal_count",
                0,
            )
        ),

        "shadow_proposals": (
            shadow.get(
                "proposals",
                [],
            )
        ),

        "learning_system_status": (
            learning.get(
                "learning_system_status",
                "UNKNOWN",
            )
        ),

        "learning_top_strategy": (
            learning_top_strategy
        ),

        "learning_top_score": (
            learning_top_score
        ),

        "learning_promote_count": (
            learning.get(
                "promote_count",
                0,
            )
        ),

        "learning_hold_count": (
            learning.get(
                "hold_count",
                0,
            )
        ),

        "learning_reduce_count": (
            learning.get(
                "reduce_count",
                0,
            )
        ),

        "learning_retire_count": (
            learning.get(
                "retire_count",
                0,
            )
        ),

        "portfolio_allocations": (
            portfolio.get(
                "allocations",
                [],
            )
        ),

        "cash_reserve_dollars": (
            portfolio.get(
                "cash_reserve_dollars",
                0.0,
            )
        ),

        "position_sizer_proposal_count": (
            position_sizer.get(
                "proposal_count",
                0,
            )
        ),

        "position_sizer_positive_size_count": (
            position_sizer.get(
                "positive_size_count",
                0,
            )
        ),

        "position_sizer_total_recommended_dollars": (
            position_sizer.get(
                "total_recommended_dollars",
                0.0,
            )
        ),

        "position_sizer_total_recommended_percent": (
            position_sizer.get(
                "total_recommended_percent",
                0.0,
            )
        ),

        "position_sizer_top_decision": (
            top_sizing_decision
        ),

        "open_positions": (
            position_manager.get(
                "open_position_count",
                performance.get(
                    "open_positions",
                    0,
                ),
            )
        ),

        "closed_positions": (
            position_manager.get(
                "closed_position_count",
                performance.get(
                    "closed_positions",
                    0,
                ),
            )
        ),

        "realized_pnl_dollars": (
            position_manager.get(
                "realized_pnl_dollars",
                performance.get(
                    "total_pnl_dollars",
                    0.0,
                ),
            )
        ),

        "unrealized_pnl_dollars": (
            position_manager.get(
                "unrealized_pnl_dollars",
                valuation.get(
                    "unrealized_pnl_dollars",
                    0.0,
                ),
            )
        ),

        "estimated_equity": (
            position_manager.get(
                "estimated_equity",
                valuation.get(
                    "estimated_equity",
                    0.0,
                ),
            )
        ),

        "performance_analytics_status": (
            analytics.get(
                "analytics_status",
                "UNKNOWN",
            )
        ),

        "performance_win_rate_percent": (
            analytics.get(
                "win_rate_percent",
                0.0,
            )
        ),

        "performance_profit_factor": (
            analytics.get(
                "profit_factor",
                0.0,
            )
        ),

        "risk_score": (
            risk.get(
                "risk_score",
                100,
            )
        ),

        "risk_level": (
            risk.get(
                "risk_level",
                "UNKNOWN",
            )
        ),

        "risk_trade_acceptance_status": (
            risk.get(
                "trade_acceptance_status",
                "UNKNOWN",
            )
        ),

        "execution_proposal_count": (
            execution.get(
                "proposal_count",
                0,
            )
        ),

        "execution_approved_count": (
            execution.get(
                "approved_count",
                0,
            )
        ),

        "execution_rejected_count": (
            execution.get(
                "rejected_count",
                0,
            )
        ),

        "position_manager_mode": (
            position_manager.get(
                "mode",
                "UNKNOWN",
            )
        ),

        "shadow_mode": True,

        "trading_client_created": False,

        "order_submitted": False,
    }


# ============================================================
# SAFETY VALIDATION
# ============================================================

def build_safety_summary(
    platform_snapshot: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    position_manager_mode = str(
        platform_snapshot.get(
            "position_manager_mode",
            "UNKNOWN",
        )
    ).upper()

    position_manager_apply_used = (
        position_manager_mode
        not in {
            "PREVIEW_ONLY",
            "UNKNOWN",
        }
    )

    trading_client_created = bool(
        platform_snapshot.get(
            "trading_client_created",
            False,
        )
    )

    order_submitted = bool(
        platform_snapshot.get(
            "order_submitted",
            False,
        )
    )

    safe = (
        not position_manager_apply_used
        and not trading_client_created
        and not order_submitted
    )

    return {

        "shadow_mode": True,

        "position_manager_mode": (
            position_manager_mode
        ),

        "position_manager_apply_used": (
            position_manager_apply_used
        ),

        "trading_client_created": (
            trading_client_created
        ),

        "order_submitted": (
            order_submitted
        ),

        "safety_passed": (
            safe
        ),
    }


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    payload: dict[str, Any],
) -> None:

    snapshot = (
        payload[
            "platform_snapshot"
        ]
    )

    safety = (
        payload[
            "safety"
        ]
    )

    lines = [

        "MASTER CONTROLLER V2",

        "=" * 36,

        (
            "Run timestamp: "
            f"{payload['run_timestamp']}"
        ),

        (
            "Overall status: "
            f"{payload['overall_status']}"
        ),

        (
            "Required steps passed: "
            f"{payload['required_steps_passed']}/"
            f"{payload['required_step_count']}"
        ),

        (
            "Failed steps: "
            f"{payload['failed_step_count']}"
        ),

        "",

        "INTELLIGENCE",

        "-" * 24,

        (
            "Learning status: "
            f"{snapshot['learning_system_status']}"
        ),

        (
            "Top learned strategy: "
            f"{snapshot['learning_top_strategy']}"
        ),

        (
            "Top learning score: "
            f"{safe_float(snapshot['learning_top_score']):.2f}"
        ),

        (
            "Promote: "
            f"{snapshot['learning_promote_count']}"
        ),

        (
            "Hold: "
            f"{snapshot['learning_hold_count']}"
        ),

        (
            "Reduce: "
            f"{snapshot['learning_reduce_count']}"
        ),

        (
            "Retire: "
            f"{snapshot['learning_retire_count']}"
        ),

        "",

        "POSITION SIZING",

        "-" * 24,

        (
            "Sizing proposals: "
            f"{snapshot['position_sizer_proposal_count']}"
        ),

        (
            "Positive sizes: "
            f"{snapshot['position_sizer_positive_size_count']}"
        ),

        (
            "Recommended dollars: $"
            f"{safe_float(snapshot['position_sizer_total_recommended_dollars']):.2f}"
        ),

        (
            "Recommended percent: "
            f"{safe_float(snapshot['position_sizer_total_recommended_percent']):.2f}%"
        ),

        "",

        "PORTFOLIO",

        "-" * 24,

        (
            "Market regime: "
            f"{snapshot['market_regime']}"
        ),

        (
            "Hall of Fame strategies: "
            f"{snapshot['hall_of_fame_strategy_count']}"
        ),

        (
            "Scanner records excluded from Hall of Fame: "
            f"{snapshot['hall_of_fame_scanner_records_excluded']}"
        ),

        (
            "Shadow proposals: "
            f"{snapshot['shadow_proposal_count']}"
        ),

        (
            "Open positions: "
            f"{snapshot['open_positions']}"
        ),

        (
            "Closed positions: "
            f"{snapshot['closed_positions']}"
        ),

        (
            "Realized P/L: $"
            f"{safe_float(snapshot['realized_pnl_dollars']):.2f}"
        ),

        (
            "Unrealized P/L: $"
            f"{safe_float(snapshot['unrealized_pnl_dollars']):.2f}"
        ),

        (
            "Estimated equity: $"
            f"{safe_float(snapshot['estimated_equity']):.2f}"
        ),

        "",

        "RISK",

        "-" * 24,

        (
            "Risk score: "
            f"{snapshot['risk_score']}"
        ),

        (
            "Risk level: "
            f"{snapshot['risk_level']}"
        ),

        (
            "Risk trade status: "
            f"{snapshot['risk_trade_acceptance_status']}"
        ),

        "",

        "EXECUTION PREVIEW",

        "-" * 24,

        (
            "Proposals: "
            f"{snapshot['execution_proposal_count']}"
        ),

        (
            "Approved: "
            f"{snapshot['execution_approved_count']}"
        ),

        (
            "Rejected: "
            f"{snapshot['execution_rejected_count']}"
        ),

        "",

        "SAFETY",

        "-" * 24,

        (
            "Shadow mode: "
            f"{safety['shadow_mode']}"
        ),

        (
            "Position Manager mode: "
            f"{safety['position_manager_mode']}"
        ),

        (
            "Position Manager apply used: "
            f"{safety['position_manager_apply_used']}"
        ),

        (
            "Trading client created: "
            f"{safety['trading_client_created']}"
        ),

        (
            "Order submitted: "
            f"{safety['order_submitted']}"
        ),

        (
            "Safety passed: "
            f"{safety['safety_passed']}"
        ),

        "",

        "STEP RESULTS",

        "-" * 24,
    ]

    for result in payload[
        "step_results"
    ]:

        lines.append(
            (
                f"{result['status']} | "
                f"{result['name']} | "
                f"{result['duration_seconds']:.3f}s"
            )
        )

        if (
            result[
                "status"
            ] == "FAILED"
        ):

            error = str(
                result.get(
                    "stderr",
                    "",
                )
            ).strip()

            if error:

                lines.append(
                    (
                        "  Error: "
                        + error.replace(
                            "\n",
                            " | ",
                        )[:800]
                    )
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

def parse_arguments() -> (
    argparse.Namespace
):

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run Master Controller v2 "
                "in research-only mode."
            )
        )
    )

    parser.add_argument(
        "--skip-data-refresh",

        action="store_true",

        help=(
            "Use existing local CSV market data."
        ),
    )

    parser.add_argument(
        "--fail-fast",

        action="store_true",

        help=(
            "Stop immediately if a required "
            "module fails."
        ),
    )

    return (
        parser.parse_args()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    arguments = (
        parse_arguments()
    )

    run_started = (
        datetime.now()
        .astimezone()
    )

    pipeline = (
        build_pipeline(
            skip_data_refresh=(
                arguments.skip_data_refresh
            )
        )
    )

    results: list[
        StepResult
    ] = []

    print(
        "Master Controller v2"
    )

    print(
        "Research / shadow mode only."
    )

    print(
        "Position Manager will run in preview mode."
    )

    print(
        "No trading client will be created."
    )

    print(
        "No order will be submitted."
    )

    print()

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    for step in pipeline:

        print(
            f"Starting: "
            f"{step.name}"
        )

        result = (
            run_step(
                step
            )
        )

        results.append(
            result
        )

        print(
            (
                f"{result.status}: "
                f"{result.name} "
                f"({result.duration_seconds:.3f}s)"
            )
        )

        if (
            result.stdout.strip()
        ):

            print(
                result.stdout.rstrip()
            )

        if (
            result.stderr.strip()
        ):

            print(
                result.stderr.rstrip(),
                file=sys.stderr,
            )

        print()

        if (
            arguments.fail_fast
            and result.required
            and result.status
            == "FAILED"
        ):

            print(
                (
                    "Stopping because "
                    "--fail-fast was requested."
                ),
                file=sys.stderr,
            )

            break

    # --------------------------------------------------------
    # COMPLETE RUN
    # --------------------------------------------------------

    run_finished = (
        datetime.now()
        .astimezone()
    )

    required_results = [

        result

        for result
        in results

        if (
            result.required
            and result.status
            != "SKIPPED"
        )
    ]

    required_passed = sum(

        result.status
        == "PASSED"

        for result
        in required_results
    )

    failed_results = [

        result

        for result
        in results

        if result.status
        == "FAILED"
    ]

    required_failed = [

        result

        for result
        in failed_results

        if result.required
    ]

    if required_failed:

        overall_status = (
            "FAILED"
        )

    elif failed_results:

        overall_status = (
            "COMPLETED_WITH_OPTIONAL_FAILURES"
        )

    else:

        overall_status = (
            "PASSED"
        )

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    platform_snapshot = (
        build_platform_snapshot()
    )

    safety_summary = (
        build_safety_summary(
            platform_snapshot
        )
    )

    if not safety_summary[
        "safety_passed"
    ]:

        overall_status = (
            "FAILED"
        )

    # --------------------------------------------------------
    # FINAL PAYLOAD
    # --------------------------------------------------------

    payload = {

        "controller": (
            "Master Controller v2"
        ),

        "run_timestamp": (
            run_started.isoformat()
        ),

        "finished_timestamp": (
            run_finished.isoformat()
        ),

        "duration_seconds": round(
            (
                run_finished
                - run_started
            ).total_seconds(),
            3,
        ),

        "overall_status": (
            overall_status
        ),

        "required_step_count": (
            len(
                required_results
            )
        ),

        "required_steps_passed": (
            required_passed
        ),

        "failed_step_count": (
            len(
                failed_results
            )
        ),

        "step_results": [

            asdict(
                result
            )

            for result
            in results
        ],

        "platform_snapshot": (
            platform_snapshot
        ),

        "safety": (
            safety_summary
        ),
    }

    # --------------------------------------------------------
    # WRITE REPORTS
    # --------------------------------------------------------

    save_json(
        LATEST_JSON_PATH,
        payload,
    )

    history_path = (
        MASTER_HISTORY_DIRECTORY
        / (
            run_started.strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        )
    )

    save_json(
        history_path,
        payload,
    )

    write_text_report(
        payload
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(
        "MASTER CONTROLLER V2 SUMMARY"
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
        f"{LATEST_JSON_PATH}"
    )

    print(
        f"Text report: "
        f"{LATEST_TEXT_PATH}"
    )

    print()

    print(
        "Research pipeline finished."
    )

    print(
        "Position Manager apply mode "
        "was not used."
    )

    print(
        "No trading client was created."
    )

    print(
        "No order was submitted."
    )

    if (
        overall_status
        == "FAILED"
    ):

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()