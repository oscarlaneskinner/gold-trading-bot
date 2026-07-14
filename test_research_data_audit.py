"""Offline test for the GLD research-data audit."""

from __future__ import annotations

import json

from research_data_audit import audit_database


def run() -> None:
    report = audit_database()

    output = {
        "status": report["status"],
        "database_exists": report["database_exists"],
        "issues": report["issues"],
        "warnings": report["warnings"],
        "checks": report["checks"],
        "production_strategy_changed": False,
        "order_submitted": False,
    }

    print("GLD research-data audit test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if report["status"] != "PASSED":
        raise SystemExit(
            "Audit found integrity problems."
        )


if __name__ == "__main__":
    run()
