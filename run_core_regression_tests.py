"""Run core read-only regression checks for the GLD platform."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


MODULES_TO_IMPORT = [
    "config",
    "features",
    "model_loader",
    "trade_memory",
    "trade_intelligence",
    "trade_review",
    "pattern_discovery",
    "research_data_audit",
    "strategy_lab",
]


def run() -> None:
    results = {}

    for module_name in MODULES_TO_IMPORT:
        try:
            importlib.import_module(module_name)
            results[module_name] = "passed"
        except Exception as error:
            results[module_name] = f"failed: {error}"

    required_files = [
        "daily_bot.py",
        "trade_memory.py",
        "trade_intelligence.py",
        "trade_review.py",
        "pattern_discovery.py",
        "research_data_audit.py",
        "strategy_lab.py",
    ]

    file_results = {
        filename: Path(filename).exists()
        for filename in required_files
    }

    failures = [
        name
        for name, status in results.items()
        if status != "passed"
    ]

    missing_files = [
        name
        for name, exists in file_results.items()
        if not exists
    ]

    output = {
        "status": (
            "passed"
            if not failures and not missing_files
            else "failed"
        ),
        "module_imports": results,
        "required_files": file_results,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    print("GLD core regression test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed":
        raise SystemExit(
            f"Regression failures: {failures}; "
            f"missing files: {missing_files}"
        )


if __name__ == "__main__":
    run()
