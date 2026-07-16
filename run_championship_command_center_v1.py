"""Run Hall of Fame and Portfolio Commander in sequence."""

from __future__ import annotations

import json
import subprocess
import sys


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(command)}")


def main() -> None:
    run([sys.executable, "strategy_hall_of_fame_v1.py"])
    run([sys.executable, "portfolio_commander_v1.py"])

    print(json.dumps({
        "status": "completed",
        "hall_of_fame_updated": True,
        "portfolio_plan_created": True,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
