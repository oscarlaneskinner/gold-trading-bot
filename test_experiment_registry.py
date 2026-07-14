"""Offline test for the GLD experiment registry."""

from __future__ import annotations

import json

from experiment_registry import (
    find_experiment,
    register_experiment,
    registry_summary,
    update_experiment,
)


def run() -> None:
    experiment = register_experiment(
        name="Synthetic confidence threshold test",
        hypothesis=(
            "A higher entry threshold may reduce weak trades "
            "without materially reducing return."
        ),
        category="entry_threshold",
        candidate_settings={
            "entry_threshold": 0.85,
            "position_percent": 0.15,
            "holding_days": 20,
            "stop_loss_percent": 0.10,
            "take_profit_percent": 0.20,
        },
        notes="Synthetic registry test only.",
        status="DRAFT",
    )

    updated = update_experiment(
        experiment["experiment_id"],
        status="READY",
        metrics={
            "test_metric": 1.0,
        },
        comparison={
            "baseline_checked": True,
        },
        decision="No production change.",
    )

    loaded = find_experiment(
        experiment["experiment_id"]
    )

    output = {
        "status": "passed",
        "experiment_id": experiment["experiment_id"],
        "experiment_status": updated["status"],
        "experiment_found": loaded is not None,
        "registry_summary": registry_summary(),
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    print("GLD experiment-registry test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
