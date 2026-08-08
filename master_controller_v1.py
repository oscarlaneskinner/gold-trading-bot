"""
Master Controller v1

Research-only orchestration layer for the gold-trading-bot platform.

This controller runs the major research modules in a controlled order
and produces one combined health report.

It does NOT:
- submit orders,
- create a trading client,
- enable live trading,
- use Position Manager --apply.

Run:
    python master_controller_v1.py

Use existing local market data:
    python master_controller_v1.py --skip-data-refresh
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

PROJECT_ROOT = Path(__file__).resolve().parent

REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

MASTER_DIRECTORY = (
    REPORTS_DIRECTORY / "master_controller"
)

MASTER_HISTORY_DIRECTORY = (
    MASTER_DIRECTORY / "history"
)

LATEST_JSON_PATH = (
    MASTER_DIRECTORY / "latest_summary.json"
)

LATEST_TEXT_PATH = (
    MASTER_DIRECTORY / "latest_summary.txt"
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


def script_exists(
    command: list[str],
) -> bool:

    if len(command) < 2:
        return True

    script_name = command[1]

    if script_name.startswith("-"):
        return True

    return (
        PROJECT_ROOT / script_name
    ).exists()


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(
    skip_data_refresh: bool,
) -> list[ControllerStep]:

    python = sys.executable

    return [

        ControllerStep(
            name=(
                "Download market research data"
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
# RUN ONE STEP
# ============================================================

def run_step(
    step: ControllerStep,
) -> StepResult:

    started = (
        datetime.now()
        .astimezone()
    )

    if not step.enabled:

        finished = (
            datetime.now()
            .astimezone()
        )

        return StepResult(
            name=step.name,
            command=step.command,
            required=step.required,
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

    if not script_exists(
        step.command
    ):

        finished = (
            datetime.now()
            .astimezone()
        )

        return StepResult(
            name=step.name,
            command=step.command,
            required=step.required,
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

    completed = subprocess.run(
        step.command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    finished = (
        datetime.now()
        .astimezone()
    )

    duration = (
        finished - started
    ).total_seconds()

    return StepResult(
        name=step.name,
        command=step.command,
        required=step.required,
        status=(
            "PASSED"
            if completed.returncode == 0
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

def build_platform_snapshot() -> dict[str, Any]:

    market_regime = load_json(
        REPORTS_DIRECTORY
        / "market_regime"
        / "market_regime_lab_v1.json",
        {},
    )

    portfolio = load_json(
        REPORTS_DIRECTORY
        / "portfolio"
        / "portfolio_commander_v1.json",
        {},
    )

    scanner = load_json(
        REPORTS_DIRECTORY
        / "scanner"
        / "championship_scanner_v1.json",
        {},
    )

    shadow = load_json(
        REPORTS_DIRECTORY
        / "shadow"
        / "two_bot_shadow_controller_v1.json",
        {},
    )

    performance = load_json(
        REPORTS_DIRECTORY
        / "performance"
        / "latest_summary.json",
        {},
    )

    position_manager = load_json(
        REPORTS_DIRECTORY
        / "position_manager"
        / "latest_summary.json",
        {},
    )

    valuation = load_json(
        REPORTS_DIRECTORY
        / "position_valuation"
        / "latest_summary.json",
        {},
    )

    analytics = load_json(
        REPORTS_DIRECTORY
        / "performance_analytics"
        / "latest_summary.json",
        {},
    )

    risk = load_json(
        REPORTS_DIRECTORY
        / "risk_manager"
        / "latest_summary.json",
        {},
    )

    execution = load_json(
        REPORTS_DIRECTORY
        / "trade_execution_manager"
        / "latest_summary.json",
        {},
    )

    hall_of_fame = load_json(
        REPORTS_DIRECTORY
        / "hall_of_fame"
        / "strategy_hall_of_fame.json",
        {},
    )

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

    lines = [
        "MASTER CONTROLLER V1",
        "=" * 32,

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

        "PLATFORM SNAPSHOT",
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
            "Unrealized P/L: $"
            f"{safe_float(snapshot['unrealized_pnl_dollars']):.2f}"
        ),

        (
            "Estimated equity: $"
            f"{safe_float(snapshot['estimated_equity']):.2f}"
        ),

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

        (
            "Execution approved: "
            f"{snapshot['execution_approved_count']}"
        ),

        (
            "Execution rejected: "
            f"{snapshot['execution_rejected_count']}"
        ),

        "",

        "SAFETY",
        "-" * 24,

        "Shadow mode: True",

        "Position Manager apply used: False",

        "Trading client created: False",

        "Order submitted: False",

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
                        )[:600]
                    )
                )

    LATEST_TEXT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete research-only "
            "trading platform pipeline."
        )
    )

    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help=(
            "Use existing local market CSV files."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop immediately if a required "
            "step fails."
        ),
    )

    return parser.parse_args()


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

    pipeline = build_pipeline(
        skip_data_refresh=(
            arguments.skip_data_refresh
        )
    )

    results: list[
        StepResult
    ] = []

    print(
        "Master Controller v1"
    )

    print(
        "Research mode only."
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

    for step in pipeline:

        print(
            f"Starting: {step.name}"
        )

        result = run_step(
            step
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

    run_finished = (
        datetime.now()
        .astimezone()
    )

    required_results = [
        result
        for result in results
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
        for result in results
        if result.status
        == "FAILED"
    ]

    required_failed = [
        result
        for result in failed_results
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

    platform_snapshot = (
        build_platform_snapshot()
    )

    payload = {

        "controller": (
            "Master Controller v1"
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
            for result in results
        ],

        "platform_snapshot": (
            platform_snapshot
        ),

        "safety": {

            "shadow_mode": True,

            "position_manager_apply_used": False,

            "trading_client_created": False,

            "order_submitted": False,
        },
    }

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

    print(
        "MASTER CONTROLLER SUMMARY"
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
        "Position Manager apply mode was not used."
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