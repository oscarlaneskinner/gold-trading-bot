"""Generate the GLD pattern-discovery research report."""

from __future__ import annotations

import json

from config import REPORTS_DIR
from logger import write_json
from pattern_discovery import discover_patterns


REPORT_PATH = REPORTS_DIR / "pattern_discovery.json"


def run() -> None:
    report = discover_patterns()

    write_json(
        REPORT_PATH,
        report,
    )

    print("GLD pattern-discovery report")
    print(
        json.dumps(
            report,
            indent=2,
        )
    )
    print(f"JSON report: {REPORT_PATH}")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
