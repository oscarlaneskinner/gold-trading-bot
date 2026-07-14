"""Register a manual GLD Strategy Laboratory experiment."""

from __future__ import annotations

import argparse
import json

from experiment_registry import (
    register_experiment,
    registry_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True,
    )

    parser.add_argument(
        "--hypothesis",
        required=True,
    )

    parser.add_argument(
        "--category",
        required=True,
    )

    parser.add_argument(
        "--settings-json",
        required=True,
        help="JSON object containing candidate settings.",
    )

    parser.add_argument(
        "--notes",
        default="",
    )

    parser.add_argument(
        "--status",
        default="DRAFT",
    )

    return parser.parse_args()


def run() -> None:
    args = parse_args()

    settings = json.loads(
        args.settings_json
    )

    if not isinstance(settings, dict):
        raise ValueError(
            "--settings-json must decode to a JSON object."
        )

    experiment = register_experiment(
        name=args.name,
        hypothesis=args.hypothesis,
        category=args.category,
        candidate_settings=settings,
        notes=args.notes,
        status=args.status,
    )

    print("GLD experiment registered")
    print(
        json.dumps(
            {
                "experiment": experiment,
                "registry_summary": registry_summary(),
                "production_strategy_changed": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
