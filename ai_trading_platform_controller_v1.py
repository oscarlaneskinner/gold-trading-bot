"""
AI Trading Platform Controller v1

Research-only orchestration for the gold-trading-bot project.

This controller runs the platform's existing research modules in a safe order,
captures stdout/stderr, writes one combined controller report, and never submits
a broker order.

Default behavior:
- downloads research market data,
- evaluates the market regime,
- updates the Strategy Hall of Fame,
- creates research-only portfolio allocations,
- scans markets,
- generates shadow proposals,
- updates the existing shadow performance tracker,
- runs Position Manager v2 in PREVIEW mode,
- runs daily position valuation,
- writes dashboard-ready reports.

It does not:
- use Position Manager v2 --apply,
- create a trading client,
- submit an order,
- alter production strategy settings.
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

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
CONTROLLER_REPORT_DIRECTORY = REPORTS_DIRECTORY / "platform_controller"
CONTROLLER_HISTORY_DIRECTORY = CONTROLLER_REPORT_DIRECTORY / "history"
LATEST_JSON_PATH = CONTROLLER_REPORT_DIRECTORY / "latest_summary.json"
LATEST_TEXT_PATH = CONTROLLER_REPORT_DIRECTORY / "latest_summary.txt"

CONTROLLER_REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
CONTROLLER_HISTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineStep:
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


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def command_script_exists(command: list[str]) -> bool:
    if len(command) < 2:
        return True
    executable_name = Path(command[0]).name.lower()
    if executable_name not in {
        "python",
        "python.exe",
        "py",
        "py.exe",
        Path(sys.executable).name.lower(),
    }:
        return True
    script_argument = command[1]
    if script_argument.startswith("-"):
        return True
    return (PROJECT_ROOT / script_argument).exists()


def run_step(step: PipelineStep) -> StepResult:
    started = datetime.now().astimezone()

    if not step.enabled:
        finished = datetime.now().astimezone()
        return StepResult(
            name=step.name,
            command=step.command,
            required=step.required,
            status="SKIPPED",
            returncode=None,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_seconds=0.0,
            stdout="",
            stderr="",
            skipped_reason="Disabled by command-line option.",
        )

    if not command_script_exists(step.command):
        finished = datetime.now().astimezone()
        return StepResult(
            name=step.name,
            command=step.command,
            required=step.required,
            status="FAILED" if step.required else "SKIPPED",
            returncode=None,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_seconds=0.0,
            stdout="",
            stderr=f"Script was not found: {step.command[1]}",
            skipped_reason=None if step.required else "Optional script was not found.",
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

    finished = datetime.now().astimezone()
    return StepResult(
        name=step.name,
        command=step.command,
        required=step.required,
        status="PASSED" if completed.returncode == 0 else "FAILED",
        returncode=completed.returncode,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_seconds=round((finished - started).total_seconds(), 3),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    python = sys.executable
    return [
        PipelineStep(
            "Download multi-market research data",
            [python, "download_multi_market_research_data.py"],
            required=True,
            enabled=not args.skip_data_refresh,
        ),
        PipelineStep(
            "Evaluate market regime",
            [python, "market_regime_lab_v1.py", "--data-dir", "data"],
        ),
        PipelineStep(
            "Update Strategy Hall of Fame",
            [python, "strategy_hall_of_fame_v1.py"],
        ),
        PipelineStep(
            "Create research portfolio allocations",
            [python, "portfolio_commander_v1.py"],
        ),
        PipelineStep(
            "Run championship market scanner",
            [python, "championship_market_scanner_v1.py", "--data-dir", "data"],
        ),
        PipelineStep(
            "Generate two-bot shadow proposals",
            [python, "two_bot_shadow_controller_v1.py"],
        ),
        PipelineStep(
            "Run shadow performance tracker",
            [python, "shadow_performance_tracker_v1.py"],
            enabled=not args.skip_performance_tracker,
        ),
        PipelineStep(
            "Run Position Manager v2 preview",
            [python, "position_manager_v2.py"],
            required=False,
            enabled=not args.skip_position_manager,
        ),
        PipelineStep(
            "Run shadow position valuation",
            [python, "shadow_position_valuation_v1.py"],
            required=False,
            enabled=not args.skip_position_valuation,
        ),
    ]


def gather_platform_snapshot() -> dict[str, Any]:
    market_regime = load_json(REPORTS_DIRECTORY / "market_regime" / "market_regime_lab_v1.json", {})
    portfolio = load_json(REPORTS_DIRECTORY / "portfolio" / "portfolio_commander_v1.json", {})
    scanner = load_json(REPORTS_DIRECTORY / "scanner" / "championship_scanner_v1.json", {})
    shadow = load_json(REPORTS_DIRECTORY / "shadow" / "two_bot_shadow_controller_v1.json", {})
    performance = load_json(REPORTS_DIRECTORY / "performance" / "latest_summary.json", {})
    manager = load_json(REPORTS_DIRECTORY / "position_manager" / "latest_summary.json", {})
    valuation = load_json(REPORTS_DIRECTORY / "position_valuation" / "latest_summary.json", {})
    hall = load_json(REPORTS_DIRECTORY / "hall_of_fame" / "strategy_hall_of_fame.json", {})
    observation = load_json(REPORTS_DIRECTORY / "shadow_observation" / "observation_state.json", {})

    top_longs = scanner.get("top_longs", [])
    top_shorts = scanner.get("top_shorts", [])

    return {
        "market_regime": market_regime.get("regime", "UNKNOWN"),
        "market_permissions": market_regime.get("permissions", {}),
        "hall_of_fame_strategy_count": hall.get("strategy_count", 0),
        "portfolio_allocations": portfolio.get("allocations", []),
        "cash_reserve_dollars": portfolio.get("cash_reserve_dollars", 0.0),
        "scanner_top_long": top_longs[0] if isinstance(top_longs, list) and top_longs else None,
        "scanner_top_short": top_shorts[0] if isinstance(top_shorts, list) and top_shorts else None,
        "shadow_proposals": shadow.get("proposals", []),
        "shadow_proposal_count": shadow.get("proposal_count", 0),
        "open_positions": performance.get("open_positions", 0),
        "closed_positions": performance.get("closed_positions", 0),
        "realized_pnl_dollars": performance.get("total_pnl_dollars", 0.0),
        "position_manager_mode": manager.get("mode", "NOT_AVAILABLE"),
        "position_manager_open_positions": manager.get("open_position_count", 0),
        "position_manager_duplicates_skipped": manager.get("duplicate_proposals_skipped", 0),
        "unrealized_pnl_dollars": valuation.get(
            "unrealized_pnl_dollars",
            manager.get("unrealized_pnl_dollars", 0.0),
        ),
        "estimated_equity": valuation.get(
            "estimated_equity",
            manager.get("estimated_equity", 0.0),
        ),
        "observation_state": observation,
        "shadow_mode": True,
        "trading_client_created": False,
        "order_submitted": False,
    }


def write_text_report(payload: dict[str, Any]) -> None:
    snapshot = payload["platform_snapshot"]
    lines = [
        "AI TRADING PLATFORM CONTROLLER V1",
        "=" * 40,
        f"Run timestamp: {payload['run_timestamp']}",
        f"Overall status: {payload['overall_status']}",
        f"Required steps passed: {payload['required_steps_passed']}/{payload['required_step_count']}",
        f"Failed steps: {payload['failed_step_count']}",
        "",
        "PLATFORM SNAPSHOT",
        "-" * 20,
        f"Market regime: {snapshot['market_regime']}",
        f"Hall of Fame strategies: {snapshot['hall_of_fame_strategy_count']}",
        f"Shadow proposals: {snapshot['shadow_proposal_count']}",
        f"Open positions: {snapshot['open_positions']}",
        f"Unrealized P/L: ${safe_float(snapshot['unrealized_pnl_dollars']):.2f}",
        f"Estimated equity: ${safe_float(snapshot['estimated_equity']):.2f}",
        "",
        "SAFETY",
        "-" * 20,
        "Shadow mode: True",
        "Position Manager apply used: False",
        "Trading client created: False",
        "Order submitted: False",
        "",
        "STEP RESULTS",
        "-" * 20,
    ]

    for result in payload["step_results"]:
        lines.append(
            f"{result['status']}: {result['name']} ({result['duration_seconds']:.3f}s)"
        )
        if result["status"] == "FAILED" and result["stderr"]:
            lines.append("  Error: " + result["stderr"].strip().replace("\n", " | ")[:500])

    LATEST_TEXT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full research-only AI trading platform pipeline."
    )
    parser.add_argument("--skip-data-refresh", action="store_true")
    parser.add_argument("--skip-performance-tracker", action="store_true")
    parser.add_argument("--skip-position-manager", action="store_true")
    parser.add_argument("--skip-position-valuation", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_started = datetime.now().astimezone()
    steps = build_steps(args)
    results: list[StepResult] = []

    print("AI Trading Platform Controller v1")
    print("Research mode only.")
    print("No trading client will be created.")
    print("No order will be submitted.\n")

    for step in steps:
        print(f"Starting: {step.name}")
        result = run_step(step)
        results.append(result)
        print(f"{result.status}: {step.name} ({result.duration_seconds:.3f}s)")

        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        print()

        if args.fail_fast and step.required and result.status == "FAILED":
            print("Stopping because --fail-fast was requested.", file=sys.stderr)
            break

    run_finished = datetime.now().astimezone()
    required_results = [r for r in results if r.required and r.status != "SKIPPED"]
    required_passed = sum(r.status == "PASSED" for r in required_results)
    failed_results = [r for r in results if r.status == "FAILED"]
    required_failed = [r for r in failed_results if r.required]

    if required_failed:
        overall_status = "FAILED"
    elif failed_results:
        overall_status = "COMPLETED_WITH_OPTIONAL_FAILURES"
    else:
        overall_status = "PASSED"

    payload = {
        "controller": "AI Trading Platform Controller v1",
        "run_timestamp": run_started.isoformat(),
        "finished_timestamp": run_finished.isoformat(),
        "duration_seconds": round((run_finished - run_started).total_seconds(), 3),
        "overall_status": overall_status,
        "required_step_count": len(required_results),
        "required_steps_passed": required_passed,
        "failed_step_count": len(failed_results),
        "step_results": [asdict(result) for result in results],
        "platform_snapshot": gather_platform_snapshot(),
        "safety": {
            "shadow_mode": True,
            "position_manager_apply_used": False,
            "trading_client_created": False,
            "order_submitted": False,
        },
    }

    save_json(LATEST_JSON_PATH, payload)
    save_json(
        CONTROLLER_HISTORY_DIRECTORY
        / f"{run_started.strftime('%Y-%m-%d_%H-%M-%S')}.json",
        payload,
    )
    write_text_report(payload)

    print(json.dumps(payload, indent=2))
    print("\nResearch pipeline finished.")
    print("No trading client was created.")
    print("No order was submitted.")

    if overall_status == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()