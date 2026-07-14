"""Persistent experiment registry for the GLD Strategy Laboratory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path("reports/strategy_experiments.json")


VALID_STATUSES = {
    "DRAFT",
    "READY",
    "RUNNING",
    "COMPLETED",
    "REJECTED",
    "PROMOTED_TO_PAPER",
    "ARCHIVED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "baseline": {
            "name": "LightGBM Production",
            "model_name": "lightgbm",
            "model_version": "lgbm_d",
            "symbol": "GLD",
            "entry_threshold": 0.50,
            "position_percent": 0.10,
            "paper_research_position_percent": 0.15,
            "holding_days": 20,
            "stop_loss_percent": 0.10,
            "take_profit_percent": 0.20,
            "status": "ACTIVE",
        },
        "next_experiment_number": 1,
        "experiments": [],
        "updated_at_utc": utc_now(),
    }


def load_registry() -> dict[str, Any]:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_PATH.exists():
        registry = default_registry()
        save_registry(registry)
        return registry

    registry = json.loads(
        REGISTRY_PATH.read_text(encoding="utf-8")
    )

    registry.setdefault("schema_version", 1)
    registry.setdefault("baseline", default_registry()["baseline"])
    registry.setdefault("next_experiment_number", 1)
    registry.setdefault("experiments", [])
    registry.setdefault("updated_at_utc", utc_now())

    return registry


def save_registry(registry: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at_utc"] = utc_now()

    temporary_path = REGISTRY_PATH.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(
            registry,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(REGISTRY_PATH)


def next_experiment_id(registry: dict[str, Any]) -> str:
    number = int(registry["next_experiment_number"])
    return f"EXP-{number:04d}"


def register_experiment(
    *,
    name: str,
    hypothesis: str,
    category: str,
    candidate_settings: dict[str, Any],
    notes: str = "",
    status: str = "DRAFT",
) -> dict[str, Any]:
    normalized_status = status.upper()

    if normalized_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid experiment status: {status!r}. "
            f"Expected one of {sorted(VALID_STATUSES)}."
        )

    if not name.strip():
        raise ValueError("Experiment name cannot be empty.")

    if not hypothesis.strip():
        raise ValueError("Experiment hypothesis cannot be empty.")

    registry = load_registry()

    experiment_id = next_experiment_id(registry)

    experiment = {
        "experiment_id": experiment_id,
        "name": name.strip(),
        "hypothesis": hypothesis.strip(),
        "category": category.strip() or "general",
        "status": normalized_status,
        "baseline_name": registry["baseline"]["name"],
        "candidate_settings": candidate_settings,
        "metrics": {},
        "comparison": {},
        "decision": None,
        "notes": notes.strip(),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }

    registry["experiments"].append(experiment)
    registry["next_experiment_number"] = (
        int(registry["next_experiment_number"]) + 1
    )

    save_registry(registry)
    return experiment


def update_experiment(
    experiment_id: str,
    *,
    status: str | None = None,
    metrics: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    decision: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    registry = load_registry()

    experiment = next(
        (
            item
            for item in registry["experiments"]
            if item["experiment_id"] == experiment_id
        ),
        None,
    )

    if experiment is None:
        raise KeyError(
            f"Experiment {experiment_id!r} was not found."
        )

    if status is not None:
        normalized_status = status.upper()

        if normalized_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid experiment status: {status!r}."
            )

        experiment["status"] = normalized_status

    if metrics is not None:
        experiment["metrics"] = metrics

    if comparison is not None:
        experiment["comparison"] = comparison

    if decision is not None:
        experiment["decision"] = decision

    if notes is not None:
        experiment["notes"] = notes

    experiment["updated_at_utc"] = utc_now()
    save_registry(registry)
    return experiment


def find_experiment(
    experiment_id: str,
) -> dict[str, Any] | None:
    registry = load_registry()

    return next(
        (
            item
            for item in registry["experiments"]
            if item["experiment_id"] == experiment_id
        ),
        None,
    )


def registry_summary() -> dict[str, Any]:
    registry = load_registry()

    status_counts: dict[str, int] = {}

    for experiment in registry["experiments"]:
        status = experiment["status"]
        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    return {
        "registry_path": str(REGISTRY_PATH),
        "baseline": registry["baseline"],
        "experiment_count": len(registry["experiments"]),
        "status_counts": status_counts,
        "next_experiment_id": next_experiment_id(registry),
    }
