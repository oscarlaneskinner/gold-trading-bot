"""Validate the GLD AI Trading Research Platform v1.0 release."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    "VERSION",
    "CHANGELOG.md",
    "RELEASE-NOTES-v1.0.0.md",
    "daily_bot.py",
    "trade_memory.py",
    "trade_intelligence.py",
    "trade_review.py",
    "pattern_discovery.py",
    "research_data_audit.py",
    "strategy_lab.py",
    "experiment_registry.py",
    "strategy_leaderboard.py",
    "baseline_comparison.py",
    "statistical_validation.py",
    "promotion_engine.py",
    "docs/ARCHITECTURE.md",
    "docs/INSTALLATION.md",
    "docs/OPERATING-GUIDE.md",
    "docs/WORKFLOW-REFERENCE.md",
    "docs/BACKUP-AND-RESTORE.md",
    "docs/SAFETY-CHECKLIST.md",
]


def run() -> None:
    version_path = Path("VERSION")
    version = (
        version_path.read_text(encoding="utf-8").strip()
        if version_path.exists()
        else None
    )

    file_status = {
        filename: Path(filename).exists()
        for filename in REQUIRED_FILES
    }

    missing = [
        filename
        for filename, exists in file_status.items()
        if not exists
    ]

    output = {
        "status": (
            "passed"
            if version == "1.0.0" and not missing
            else "failed"
        ),
        "version": version,
        "required_file_count": len(REQUIRED_FILES),
        "missing_files": missing,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    print("GLD Version 1.0 release validation")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed":
        raise SystemExit(
            "Version 1.0 release validation failed."
        )


if __name__ == "__main__":
    run()
