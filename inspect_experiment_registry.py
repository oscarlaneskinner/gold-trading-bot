"""Inspect the GLD Strategy Laboratory experiment registry."""

from __future__ import annotations

import json

from experiment_registry import (
    load_registry,
    registry_summary,
)


def run() -> None:
    output = {
        "summary": registry_summary(),
        "registry": load_registry(),
        "production_strategy_changed": False,
        "order_submitted": False,
    }

    print("GLD experiment-registry summary")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
