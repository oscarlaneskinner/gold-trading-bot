"""Generate and save the GLD research-data quality audit."""

from __future__ import annotations

import json

from config import REPORTS_DIR
from logger import write_json
from research_data_audit import audit_database


REPORT_PATH = REPORTS_DIR / "research_data_audit.json"


def run() -> None:
    report = audit_database()

    write_json(
        REPORT_PATH,
        report,
    )

    print("GLD research-data audit")
    print(json.dumps(report, indent=2))
    print(f"JSON report: {REPORT_PATH}")
    print("No market request was made.")
    print("No order was submitted.")

    if report["status"] != "PASSED":
        raise SystemExit(
            "Research-data audit failed. Review the reported issues."
        )


if __name__ == "__main__":
    run()
