"""CSV and JSON logging helpers."""

from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from config import DECISION_LOG_PATH, TRADE_LOG_PATH, create_project_directories

def append_csv(path: Path, row: dict[str, Any]) -> None:
    create_project_directories()
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def log_decision(**values: Any) -> None:
    append_csv(DECISION_LOG_PATH, {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **values})

def log_trade(**values: Any) -> None:
    append_csv(TRADE_LOG_PATH, {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **values})

def write_json(path: Path, payload: dict[str, Any]) -> None:
    create_project_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
