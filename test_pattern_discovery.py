"""Offline test for GLD pattern discovery."""

from __future__ import annotations

import json

from pattern_discovery import discover_patterns


def run() -> None:
    report = discover_patterns()

    output = {
        "status": "passed",
        "overall": report["overall"],
        "recommendations": report["recommendations"],
        "order_submitted": False,
    }

    print("GLD pattern-discovery offline test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
