"""Read-only integrity audit for the GLD research database."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trade_memory import DATABASE_PATH


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def audit_database() -> dict[str, Any]:
    database_path = Path(DATABASE_PATH)

    if not database_path.exists():
        return {
            "status": "FAILED",
            "database_exists": False,
            "database_path": str(database_path),
            "issues": ["The trade-memory database does not exist."],
            "warnings": [],
            "checks": {},
            "order_submitted": False,
        }

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    issues: list[str] = []
    warnings: list[str] = []

    try:
        integrity_result = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        foreign_key_rows = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        checks: dict[str, Any] = {
            "sqlite_integrity_check": integrity_result,
            "foreign_key_violation_count": len(foreign_key_rows),
            "decision_count": scalar(
                connection,
                "SELECT COUNT(*) FROM decisions",
            ),
            "trade_count": scalar(
                connection,
                "SELECT COUNT(*) FROM trades",
            ),
            "open_trade_count": scalar(
                connection,
                "SELECT COUNT(*) FROM trades WHERE status = 'OPEN'",
            ),
            "closed_trade_count": scalar(
                connection,
                "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'",
            ),
            "decision_intelligence_count": scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type = 'table'
                AND name = 'decision_intelligence'
                """,
            ),
        }

        intelligence_table_exists = (
            checks["decision_intelligence_count"] == 1
        )

        if intelligence_table_exists:
            checks["intelligence_record_count"] = scalar(
                connection,
                "SELECT COUNT(*) FROM decision_intelligence",
            )
            checks["orphan_intelligence_count"] = scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM decision_intelligence di
                LEFT JOIN decisions d
                    ON d.id = di.decision_id
                WHERE d.id IS NULL
                """,
            )
        else:
            checks["intelligence_record_count"] = 0
            checks["orphan_intelligence_count"] = 0
            warnings.append(
                "The decision_intelligence table has not been created yet."
            )

        checks["duplicate_entry_order_ids"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT entry_order_id
                FROM trades
                WHERE entry_order_id IS NOT NULL
                GROUP BY entry_order_id
                HAVING COUNT(*) > 1
            )
            """,
        )

        checks["duplicate_exit_order_ids"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT exit_order_id
                FROM trades
                WHERE exit_order_id IS NOT NULL
                GROUP BY exit_order_id
                HAVING COUNT(*) > 1
            )
            """,
        )

        checks["closed_trades_missing_exit_price"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM trades
            WHERE status = 'CLOSED'
            AND exit_price IS NULL
            """,
        )

        checks["closed_trades_missing_return"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM trades
            WHERE status = 'CLOSED'
            AND gross_return IS NULL
            """,
        )

        checks["open_trades_with_exit_data"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM trades
            WHERE status = 'OPEN'
            AND (
                exit_order_id IS NOT NULL
                OR exit_price IS NOT NULL
                OR exit_timestamp_utc IS NOT NULL
            )
            """,
        )

        checks["decisions_missing_features"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM decisions
            WHERE features_json IS NULL
            OR TRIM(features_json) = ''
            """,
        )

        checks["buy_decisions_without_order_id"] = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM decisions
            WHERE action = 'BUY'
            AND order_id IS NULL
            """,
        )

        if integrity_result != "ok":
            issues.append(
                f"SQLite integrity check returned: {integrity_result}"
            )

        if foreign_key_rows:
            issues.append(
                f"{len(foreign_key_rows)} foreign-key violation(s) found."
            )

        for field in (
            "orphan_intelligence_count",
            "duplicate_entry_order_ids",
            "duplicate_exit_order_ids",
            "closed_trades_missing_exit_price",
            "closed_trades_missing_return",
            "open_trades_with_exit_data",
        ):
            if checks[field] > 0:
                issues.append(
                    f"{field} = {checks[field]}"
                )

        if checks["open_trade_count"] > 1:
            warnings.append(
                "More than one OPEN trade exists, but the current GLD system "
                "is designed for one position at a time."
            )

        if checks["decisions_missing_features"] > 0:
            warnings.append(
                f"{checks['decisions_missing_features']} decision(s) are "
                "missing feature snapshots."
            )

        if checks["buy_decisions_without_order_id"] > 0:
            warnings.append(
                f"{checks['buy_decisions_without_order_id']} BUY decision(s) "
                "have no order ID."
            )

        status = "PASSED" if not issues else "FAILED"

        return {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": status,
            "database_exists": True,
            "database_path": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "checks": checks,
            "issues": issues,
            "warnings": warnings,
            "production_strategy_changed": False,
            "market_request_made": False,
            "order_submitted": False,
        }

    finally:
        connection.close()
