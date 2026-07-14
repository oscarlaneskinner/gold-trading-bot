"""SQLite trade-memory storage for the GLD paper-trading bot."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import ROOT_DIR, create_project_directories


DATABASE_PATH = ROOT_DIR / "data" / "trade_memory.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    create_project_directories()
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database() -> None:
    with closing(connect()) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                market_timestamp TEXT,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT,
                prediction INTEGER NOT NULL,
                probability_up REAL NOT NULL,
                price REAL NOT NULL,
                action TEXT NOT NULL,
                reason TEXT,
                position_percent REAL,
                notional REAL,
                market_open INTEGER NOT NULL,
                existing_position INTEGER NOT NULL,
                open_buy_order_count INTEGER NOT NULL,
                order_id TEXT,
                paper_trading INTEGER NOT NULL,
                features_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_symbol_time
            ON decisions(symbol, timestamp_utc);

            CREATE INDEX IF NOT EXISTS idx_decisions_order_id
            ON decisions(order_id);

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_order_id TEXT UNIQUE,
                exit_order_id TEXT,
                entry_timestamp_utc TEXT,
                exit_timestamp_utc TEXT,
                entry_price REAL,
                exit_price REAL,
                notional REAL,
                quantity REAL,
                entry_probability_up REAL,
                position_percent REAL,
                exit_reason TEXT,
                gross_return REAL,
                gross_profit_loss REAL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                entry_features_json TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
            ON trades(symbol, status);
            """
        )
        connection.commit()


def record_decision(
    *,
    symbol: str,
    model_name: str,
    model_version: str | None,
    prediction: int,
    probability_up: float,
    price: float,
    action: str,
    reason: str,
    position_percent: float | None,
    notional: float | None,
    market_open: bool,
    existing_position: bool,
    open_buy_order_count: int,
    order_id: str | None,
    paper_trading: bool,
    market_timestamp: str | None,
    features: dict[str, float],
) -> int:
    initialize_database()

    with closing(connect()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO decisions (
                timestamp_utc,
                market_timestamp,
                symbol,
                model_name,
                model_version,
                prediction,
                probability_up,
                price,
                action,
                reason,
                position_percent,
                notional,
                market_open,
                existing_position,
                open_buy_order_count,
                order_id,
                paper_trading,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                market_timestamp,
                symbol,
                model_name,
                model_version,
                int(prediction),
                float(probability_up),
                float(price),
                action,
                reason,
                position_percent,
                notional,
                int(market_open),
                int(existing_position),
                int(open_buy_order_count),
                order_id,
                int(paper_trading),
                json.dumps(features, sort_keys=True),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def record_trade_entry(
    *,
    symbol: str,
    entry_order_id: str,
    entry_timestamp_utc: str | None,
    entry_price: float,
    notional: float,
    quantity: float | None,
    probability_up: float,
    position_percent: float,
    entry_features: dict[str, float],
) -> int:
    initialize_database()
    now = utc_now()

    with closing(connect()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO trades (
                symbol,
                entry_order_id,
                entry_timestamp_utc,
                entry_price,
                notional,
                quantity,
                entry_probability_up,
                position_percent,
                entry_features_json,
                status,
                created_at_utc,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            ON CONFLICT(entry_order_id) DO UPDATE SET
                entry_timestamp_utc = excluded.entry_timestamp_utc,
                entry_price = excluded.entry_price,
                notional = excluded.notional,
                quantity = excluded.quantity,
                entry_probability_up = excluded.entry_probability_up,
                position_percent = excluded.position_percent,
                entry_features_json = excluded.entry_features_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                symbol,
                entry_order_id,
                entry_timestamp_utc,
                float(entry_price),
                float(notional),
                quantity,
                float(probability_up),
                float(position_percent),
                json.dumps(entry_features, sort_keys=True),
                now,
                now,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid or 0)


def close_latest_open_trade(
    *,
    symbol: str,
    exit_order_id: str,
    exit_timestamp_utc: str | None,
    exit_price: float,
    exit_reason: str,
) -> bool:
    initialize_database()

    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM trades
            WHERE symbol = ? AND status = 'OPEN'
            ORDER BY id DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()

        if row is None:
            return False

        entry_price = float(row["entry_price"])
        quantity = row["quantity"]
        notional = float(row["notional"])

        gross_return = (
            float(exit_price) / entry_price - 1
            if entry_price > 0
            else None
        )

        gross_profit_loss = (
            float(quantity) * (float(exit_price) - entry_price)
            if quantity is not None
            else notional * float(gross_return or 0.0)
        )

        connection.execute(
            """
            UPDATE trades
            SET
                exit_order_id = ?,
                exit_timestamp_utc = ?,
                exit_price = ?,
                exit_reason = ?,
                gross_return = ?,
                gross_profit_loss = ?,
                status = 'CLOSED',
                updated_at_utc = ?
            WHERE id = ?
            """,
            (
                exit_order_id,
                exit_timestamp_utc,
                float(exit_price),
                exit_reason,
                gross_return,
                gross_profit_loss,
                utc_now(),
                int(row["id"]),
            ),
        )
        connection.commit()
        return True


def database_summary() -> dict[str, Any]:
    initialize_database()

    with closing(connect()) as connection:
        decision_count = connection.execute(
            "SELECT COUNT(*) FROM decisions"
        ).fetchone()[0]

        trade_count = connection.execute(
            "SELECT COUNT(*) FROM trades"
        ).fetchone()[0]

        open_trade_count = connection.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'OPEN'"
        ).fetchone()[0]

        closed_trade_count = connection.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()[0]

    return {
        "database_path": str(DATABASE_PATH),
        "decision_count": int(decision_count),
        "trade_count": int(trade_count),
        "open_trade_count": int(open_trade_count),
        "closed_trade_count": int(closed_trade_count),
    }
