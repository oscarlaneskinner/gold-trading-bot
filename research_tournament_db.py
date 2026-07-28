"""Research database and tournament manager for strategy results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATABASE_PATH = Path("data/research_tournament.sqlite3")
REPORT_PATH = Path("reports/research_tournament_leaderboard.json")


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_code TEXT NOT NULL UNIQUE,
    asset TEXT NOT NULL,
    venue TEXT,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    starting_capital REAL NOT NULL,
    commission_percent REAL NOT NULL,
    slippage_units REAL NOT NULL,
    position_percent REAL NOT NULL,
    net_profit_amount REAL,
    net_profit_percent REAL NOT NULL,
    max_drawdown_amount REAL,
    max_drawdown_percent REAL NOT NULL,
    profitable_trades_percent REAL NOT NULL,
    profit_factor REAL NOT NULL,
    closed_trades INTEGER NOT NULL,
    market_regime TEXT,
    source_platform TEXT NOT NULL,
    strategy_version TEXT,
    notes TEXT,
    created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_asset_timeframe
ON experiments(asset, timeframe);

CREATE INDEX IF NOT EXISTS idx_experiments_strategy
ON experiments(strategy_name, strategy_variant);

CREATE INDEX IF NOT EXISTS idx_experiments_platform
ON experiments(source_platform);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def next_experiment_code(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM experiments"
    ).fetchone()

    return f"RND-{int(row['count']) + 1:05d}"


def score_result(record: dict[str, Any]) -> float:
    net_profit_percent = float(record["net_profit_percent"])
    profit_factor = min(float(record["profit_factor"]), 5.0)
    drawdown_percent = abs(float(record["max_drawdown_percent"]))
    win_rate = float(record["profitable_trades_percent"])
    closed_trades = int(record["closed_trades"])

    sample_multiplier = min(closed_trades / 30.0, 1.0)
    risk_adjusted = (
        net_profit_percent / drawdown_percent
        if drawdown_percent > 0
        else 0.0
    )

    raw_score = (
        net_profit_percent * 1.5
        + profit_factor * 8.0
        + win_rate * 0.15
        + min(risk_adjusted, 10.0) * 2.0
        - drawdown_percent * 1.5
    )

    return round(raw_score * sample_multiplier, 4)


def register_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    connection = connect()

    try:
        experiment_code = payload.get(
            "experiment_code"
        ) or next_experiment_code(connection)

        values = {
            "experiment_code": experiment_code,
            "asset": payload["asset"].upper(),
            "venue": payload.get("venue"),
            "strategy_name": payload["strategy_name"],
            "strategy_variant": payload["strategy_variant"],
            "timeframe": payload["timeframe"],
            "date_start": payload["date_start"],
            "date_end": payload["date_end"],
            "starting_capital": float(payload["starting_capital"]),
            "commission_percent": float(payload["commission_percent"]),
            "slippage_units": float(payload["slippage_units"]),
            "position_percent": float(payload["position_percent"]),
            "net_profit_amount": (
                float(payload["net_profit_amount"])
                if payload.get("net_profit_amount") is not None
                else None
            ),
            "net_profit_percent": float(payload["net_profit_percent"]),
            "max_drawdown_amount": (
                float(payload["max_drawdown_amount"])
                if payload.get("max_drawdown_amount") is not None
                else None
            ),
            "max_drawdown_percent": float(payload["max_drawdown_percent"]),
            "profitable_trades_percent": float(
                payload["profitable_trades_percent"]
            ),
            "profit_factor": float(payload["profit_factor"]),
            "closed_trades": int(payload["closed_trades"]),
            "market_regime": payload.get("market_regime"),
            "source_platform": payload["source_platform"],
            "strategy_version": payload.get("strategy_version"),
            "notes": payload.get("notes", ""),
            "created_at_utc": utc_now(),
        }

        connection.execute(
            """
            INSERT INTO experiments (
                experiment_code,
                asset,
                venue,
                strategy_name,
                strategy_variant,
                timeframe,
                date_start,
                date_end,
                starting_capital,
                commission_percent,
                slippage_units,
                position_percent,
                net_profit_amount,
                net_profit_percent,
                max_drawdown_amount,
                max_drawdown_percent,
                profitable_trades_percent,
                profit_factor,
                closed_trades,
                market_regime,
                source_platform,
                strategy_version,
                notes,
                created_at_utc
            )
            VALUES (
                :experiment_code,
                :asset,
                :venue,
                :strategy_name,
                :strategy_variant,
                :timeframe,
                :date_start,
                :date_end,
                :starting_capital,
                :commission_percent,
                :slippage_units,
                :position_percent,
                :net_profit_amount,
                :net_profit_percent,
                :max_drawdown_amount,
                :max_drawdown_percent,
                :profitable_trades_percent,
                :profit_factor,
                :closed_trades,
                :market_regime,
                :source_platform,
                :strategy_version,
                :notes,
                :created_at_utc
            )
            """,
            values,
        )

        connection.commit()

        row = connection.execute(
            """
            SELECT *
            FROM experiments
            WHERE experiment_code = ?
            """,
            (experiment_code,),
        ).fetchone()

        record = dict(row)
        record["score"] = score_result(record)
        return record

    finally:
        connection.close()


def load_experiments(
    *,
    asset: str | None = None,
    timeframe: str | None = None,
    source_platform: str | None = None,
) -> list[dict[str, Any]]:
    connection = connect()

    try:
        clauses = []
        parameters: list[Any] = []

        if asset:
            clauses.append("asset = ?")
            parameters.append(asset.upper())

        if timeframe:
            clauses.append("timeframe = ?")
            parameters.append(timeframe)

        if source_platform:
            clauses.append("source_platform = ?")
            parameters.append(source_platform)

        where = (
            "WHERE " + " AND ".join(clauses)
            if clauses
            else ""
        )

        rows = connection.execute(
            f"""
            SELECT *
            FROM experiments
            {where}
            ORDER BY created_at_utc ASC
            """,
            parameters,
        ).fetchall()

        results = []

        for row in rows:
            record = dict(row)
            record["score"] = score_result(record)
            results.append(record)

        return results

    finally:
        connection.close()


def build_leaderboard(
    *,
    asset: str | None = None,
    timeframe: str | None = None,
    source_platform: str | None = None,
    minimum_trades: int = 0,
) -> dict[str, Any]:
    records = [
        record
        for record in load_experiments(
            asset=asset,
            timeframe=timeframe,
            source_platform=source_platform,
        )
        if int(record["closed_trades"]) >= minimum_trades
    ]

    records.sort(
        key=lambda item: (
            item["score"],
            item["net_profit_percent"],
            -abs(item["max_drawdown_percent"]),
        ),
        reverse=True,
    )

    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    report = {
        "filters": {
            "asset": asset,
            "timeframe": timeframe,
            "source_platform": source_platform,
            "minimum_trades": minimum_trades,
        },
        "experiment_count": len(records),
        "leaderboard": records,
        "generated_at_utc": utc_now(),
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def database_summary() -> dict[str, Any]:
    connection = connect()

    try:
        total = connection.execute(
            "SELECT COUNT(*) AS count FROM experiments"
        ).fetchone()["count"]

        assets = connection.execute(
            """
            SELECT asset, COUNT(*) AS count
            FROM experiments
            GROUP BY asset
            ORDER BY asset
            """
        ).fetchall()

        timeframes = connection.execute(
            """
            SELECT timeframe, COUNT(*) AS count
            FROM experiments
            GROUP BY timeframe
            ORDER BY timeframe
            """
        ).fetchall()

        strategies = connection.execute(
            """
            SELECT strategy_name, strategy_variant, COUNT(*) AS count
            FROM experiments
            GROUP BY strategy_name, strategy_variant
            ORDER BY strategy_name, strategy_variant
            """
        ).fetchall()

        return {
            "database_path": str(DATABASE_PATH),
            "experiment_count": int(total),
            "asset_counts": {
                row["asset"]: row["count"]
                for row in assets
            },
            "timeframe_counts": {
                row["timeframe"]: row["count"]
                for row in timeframes
            },
            "strategy_counts": [
                dict(row)
                for row in strategies
            ],
        }

    finally:
        connection.close()
