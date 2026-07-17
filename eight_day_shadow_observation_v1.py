"""Eight-Day Automated Shadow Observation Runner v1.

Runs the existing research pipeline, preserves dated reports, optionally emails
a daily summary, and stops after the configured number of successful observation
days.

This script does not create a trading client and does not submit orders.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import smtplib
import subprocess
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def current_time(timezone_name: str) -> datetime:
    return datetime.now(ZoneInfo(timezone_name))


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
        "status": "passed" if completed.returncode == 0 else "failed",
    }


def inspect_safety(files: list[Path], forbidden: list[str]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []

    for path in files:
        if not path.exists() or path.suffix.lower() != ".py":
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        for token in forbidden:
            if token in text:
                violations.append(
                    {
                        "file": str(path),
                        "token": token,
                    }
                )

    return {
        "passed": not violations,
        "violations": violations,
    }


def copy_report(source: Path, destination_directory: Path) -> str | None:
    if not source.exists():
        return None

    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / source.name
    shutil.copy2(source, destination)
    return str(destination)


def build_summary(
    now: datetime,
    config: dict[str, Any],
    command_results: list[dict[str, Any]],
    archived_reports: dict[str, str | None],
) -> dict[str, Any]:
    paths = config["report_paths"]

    regime = load_json(Path(paths["market_regime"]))
    hall = load_json(Path(paths["hall_of_fame"]))
    portfolio = load_json(Path(paths["portfolio"]))
    scanner = load_json(Path(paths["scanner"]))
    shadow = load_json(Path(paths["shadow_controller"]))

    proposals = shadow.get("proposals", [])
    top_longs = scanner.get("top_longs", [])

    return {
        "run_timestamp": now.isoformat(),
        "run_date": now.date().isoformat(),
        "market_regime": regime.get("regime", "UNKNOWN"),
        "permissions": regime.get("permissions", {}),
        "hall_of_fame_strategy_count": hall.get("strategy_count", 0),
        "portfolio_allocations": portfolio.get("allocations", []),
        "scanner_top_long": top_longs[0] if top_longs else None,
        "shadow_proposal_count": shadow.get("proposal_count", 0),
        "shadow_proposed_total_dollars": shadow.get(
            "proposed_total_dollars",
            0.0,
        ),
        "shadow_proposals": proposals,
        "missing_inputs": shadow.get("missing_inputs", []),
        "command_results": command_results,
        "archived_reports": archived_reports,
        "shadow_mode": True,
        "production_strategy_changed": False,
        "market_request_made": any(
            result["command"] == config.get("market_data_command")
            and result["returncode"] == 0
            for result in command_results
        ),
        "order_submitted": False,
    }


def summary_text(summary: dict[str, Any], state: dict[str, Any]) -> str:
    lines = [
        "EIGHT-DAY SHADOW OBSERVATION",
        "=" * 30,
        f"Observation day: {state.get('completed_days', 0)} / {state.get('target_days', 8)}",
        f"Date: {summary['run_date']}",
        f"Market regime: {summary['market_regime']}",
        f"Proposal count: {summary['shadow_proposal_count']}",
        f"Proposed total: ${float(summary['shadow_proposed_total_dollars']):.2f}",
        "",
        "BOT PERMISSIONS",
    ]

    for key, value in summary.get("permissions", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "SHADOW PROPOSALS"])

    proposals = summary.get("shadow_proposals", [])
    if proposals:
        for index, proposal in enumerate(proposals, start=1):
            lines.append(
                f"{index}. {proposal.get('role')} | "
                f"{proposal.get('side')} {proposal.get('symbol')} | "
                f"${float(proposal.get('proposed_dollars', 0)):.2f} | "
                f"{proposal.get('strategy_name', '')}"
            )
    else:
        lines.append("- No proposals generated.")

    failures = [
        result
        for result in summary.get("command_results", [])
        if result.get("returncode") != 0
    ]

    lines.extend(["", "PIPELINE STATUS"])

    if failures:
        for failure in failures:
            lines.append(
                f"- FAILED: {' '.join(failure['command'])}"
            )
    else:
        lines.append("- All required commands completed successfully.")

    lines.extend(
        [
            "",
            "Shadow mode only.",
            "No order was submitted.",
        ]
    )

    return "\n".join(lines)


def send_email(config: dict[str, Any], subject: str, body: str) -> dict[str, Any]:
    email_config = config["email"]

    if not email_config.get("enabled", False):
        return {"status": "disabled"}

    recipient = os.getenv(email_config["recipient_env"], "").strip()
    sender = os.getenv(email_config["sender_env"], "").strip()
    host = os.getenv(email_config["smtp_host_env"], "").strip()
    username = os.getenv(email_config["smtp_username_env"], "").strip()
    password = os.getenv(email_config["smtp_password_env"], "").strip()
    port_text = os.getenv(email_config["smtp_port_env"], "587").strip()

    if not all([recipient, sender, host, username, password]):
        return {
            "status": "preview_only",
            "reason": "SMTP environment variables are incomplete.",
        }

    try:
        port = int(port_text)
    except ValueError:
        port = 587

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as server:
        if email_config.get("use_tls", True):
            server.starttls()
        server.login(username, password)
        server.send_message(message)

    return {
        "status": "sent",
        "recipient": recipient,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/eight_day_shadow_observation_v1.json",
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)

    timezone_name = config["timezone"]
    now = current_time(timezone_name)

    state_path = Path(config["state_file"])

    if args.reset and state_path.exists():
        state_path.unlink()

    state = load_json(state_path)

    if not state:
        state = {
            "target_days": int(config["observation_days"]),
            "completed_days": 0,
            "successful_dates": [],
            "failed_dates": [],
            "status": "ACTIVE",
        }

    if state.get("completed_days", 0) >= state.get("target_days", 8):
        state["status"] = "COMPLETE"
        save_json(state_path, state)
        print(json.dumps(state, indent=2))
        return

    date_key = now.date().isoformat()

    if date_key in state.get("successful_dates", []):
        print(
            json.dumps(
                {
                    "status": "already_completed_today",
                    "date": date_key,
                    "completed_days": state["completed_days"],
                    "target_days": state["target_days"],
                    "order_submitted": False,
                },
                indent=2,
            )
        )
        return

    pipeline_files = [
        Path("market_regime_lab_v1.py"),
        Path("strategy_hall_of_fame_v1.py"),
        Path("portfolio_commander_v1.py"),
        Path("championship_market_scanner_v1.py"),
        Path("two_bot_shadow_controller_v1.py"),
        Path(__file__),
    ]

    safety = inspect_safety(
        pipeline_files,
        config["safety"]["forbidden_modules"],
    )

    if not safety["passed"]:
        raise SystemExit(
            "Safety inspection failed: "
            + json.dumps(safety["violations"], indent=2)
        )

    command_results: list[dict[str, Any]] = []

    if config.get("refresh_market_data", True) and not args.skip_data_refresh:
        command_results.append(
            run_command(config["market_data_command"])
        )

    for command in config["required_commands"]:
        command_results.append(run_command(command))

    all_required_passed = all(
        item["returncode"] == 0
        for item in command_results
        if item["command"] != config.get("market_data_command")
    )

    archive_root = (
        Path(config["archive_directory"])
        / date_key
    )

    archived_reports: dict[str, str | None] = {}

    for name, path_text in config["report_paths"].items():
        archived_reports[name] = copy_report(
            Path(path_text),
            archive_root,
        )

    summary = build_summary(
        now,
        config,
        command_results,
        archived_reports,
    )

    if all_required_passed:
        state.setdefault("successful_dates", []).append(date_key)
        state["completed_days"] = len(state["successful_dates"])
    else:
        state.setdefault("failed_dates", []).append(date_key)

    state["last_run_timestamp"] = now.isoformat()
    state["status"] = (
        "COMPLETE"
        if state["completed_days"] >= state["target_days"]
        else "ACTIVE"
    )

    summary["observation_state"] = state

    latest_summary_path = Path(config["daily_summary_file"])
    save_json(latest_summary_path, summary)

    dated_summary_path = archive_root / "daily_summary.json"
    save_json(dated_summary_path, summary)

    body = summary_text(summary, state)

    latest_text_path = Path(config["daily_text_file"])
    latest_text_path.parent.mkdir(parents=True, exist_ok=True)
    latest_text_path.write_text(body, encoding="utf-8")
    (archive_root / "daily_summary.txt").write_text(body, encoding="utf-8")

    subject = (
        f"{config['email']['subject_prefix']} "
        f"Day {state['completed_days']}/{state['target_days']} "
        f"- {summary['market_regime']}"
    )

    email_result = send_email(config, subject, body)
    summary["email_result"] = email_result

    preview_path = Path(config["email_preview_file"])
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(
        f"Subject: {subject}\n\n{body}",
        encoding="utf-8",
    )

    save_json(latest_summary_path, summary)
    save_json(dated_summary_path, summary)
    save_json(state_path, state)

    print("Eight-Day Automated Shadow Observation Runner v1")
    print(
        json.dumps(
            {
                "status": state["status"],
                "observation_day": state["completed_days"],
                "target_days": state["target_days"],
                "run_date": date_key,
                "market_regime": summary["market_regime"],
                "proposal_count": summary["shadow_proposal_count"],
                "pipeline_passed": all_required_passed,
                "email_result": email_result,
                "shadow_mode": True,
                "production_strategy_changed": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )
    print("Shadow mode only.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
