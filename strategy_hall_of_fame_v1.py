"""
Strategy Hall of Fame v1

Research-only strategy ranking database for the gold-trading-bot platform.

IMPORTANT:
The Hall of Fame contains STRATEGY RESEARCH results only.

Scanner opportunities such as:
- top_longs
- top_shorts
- individual symbols such as AMZN or SPY

are NOT strategies and are intentionally excluded.

This version also removes legacy scanner-derived
"Unknown Strategy" records from the Hall of Fame database.

The script:
- reads strategy research arena reports,
- normalizes candidate strategy results,
- stores them in SQLite,
- ranks strategies,
- writes CSV and JSON reports,
- never changes a production strategy,
- never creates a trading client,
- never submits an order.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "hall_of_fame"

DATABASE_PATH = (
    DATA_DIRECTORY
    / "strategy_hall_of_fame.sqlite3"
)

CSV_REPORT_PATH = (
    REPORT_DIRECTORY
    / "strategy_hall_of_fame.csv"
)

JSON_REPORT_PATH = (
    REPORT_DIRECTORY
    / "strategy_hall_of_fame.json"
)


# ============================================================
# DEFAULT STRATEGY RESEARCH SOURCES
# ============================================================

DEFAULT_REPORTS = [
    "reports/arena_v3_results.json",
    "reports/multi_market_strategy_arena_v2.json",
    "reports/strategy_research_arena.json",
]

# Scanner output is intentionally NOT included.
#
# DO NOT add:
#
# reports/scanner/championship_scanner_v1.json
#
# Scanner symbols are opportunities, not tested strategies.


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class StrategyRecord:
    strategy_key: str
    strategy_name: str
    role: str
    symbol: str
    source_report: str

    score: float

    return_percent: float
    drawdown_percent: float
    profit_factor: float
    consistency_percent: float
    trade_count: int

    status: str
    parameters_json: str


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any] | None:

    if not path.exists():
        return None

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    return payload


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:

        if value is None:
            return default

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def clean_strategy_name(
    value: Any,
) -> str:

    name = str(
        value or ""
    ).strip()

    if not name:
        return ""

    if name.lower() == "unknown strategy":
        return ""

    return name


# ============================================================
# ROLE INFERENCE
# ============================================================

def infer_role(
    strategy_name: str,
    symbol: str,
) -> str:

    name_upper = (
        strategy_name.upper()
    )

    symbol_upper = (
        symbol.upper()
    )

    if (
        "SHORT" in name_upper
        or "BEAR" in name_upper
    ):
        return "SHORT"

    if (
        "LONG" in name_upper
        or "BULL" in name_upper
    ):
        return "LONG"

    if symbol_upper == "GLD":
        return "GLD"

    return "LONG"


# ============================================================
# CANDIDATE NORMALIZATION
# ============================================================

def candidate_to_record(
    candidate: dict[str, Any],
    source: str,
) -> StrategyRecord | None:

    name = clean_strategy_name(
        candidate.get(
            "candidate_id"
        )
        or candidate.get(
            "strategy_name"
        )
        or candidate.get(
            "filter_set"
        )
        or candidate.get(
            "strategy_variant"
        )
    )

    # A real strategy must have an actual strategy identifier.
    # Do not manufacture "Unknown Strategy" records.
    if not name:
        return None

    symbol = str(
        candidate.get(
            "symbol"
        )
        or candidate.get(
            "asset"
        )
        or "MULTI"
    ).upper()

    score = safe_float(
        candidate.get(
            "score",
            0.0,
        )
    )

    return_percent = safe_float(
        candidate.get(
            "median_test_return_percent",
            candidate.get(
                "test_return_percent",
                candidate.get(
                    "net_profit_percent",
                    0.0,
                ),
            ),
        )
    )

    drawdown_percent = safe_float(
        candidate.get(
            "median_drawdown_percent",
            candidate.get(
                "test_drawdown_percent",
                candidate.get(
                    "max_drawdown_percent",
                    0.0,
                ),
            ),
        )
    )

    profit_factor = safe_float(
        candidate.get(
            "median_profit_factor",
            candidate.get(
                "test_profit_factor",
                candidate.get(
                    "profit_factor",
                    0.0,
                ),
            ),
        )
    )

    consistency_percent = safe_float(
        candidate.get(
            "consistency_percent",
            0.0,
        )
    )

    trade_count = safe_int(
        candidate.get(
            "median_trade_count",
            candidate.get(
                "test_trade_count",
                candidate.get(
                    "closed_trades",
                    0,
                ),
            ),
        )
    )

    status = str(
        candidate.get(
            "status",
            "RESEARCH",
        )
    ).upper()

    role = str(
        candidate.get(
            "role",
            "",
        )
    ).upper()

    if role not in {
        "LONG",
        "SHORT",
        "GLD",
    }:
        role = infer_role(
            name,
            symbol,
        )

    parameters = {
        key: candidate.get(
            key
        )
        for key in [
            "filter_set",
            "sensitivity",
            "atr_period",
            "max_bars_held",
            "stop_loss_percent",
            "take_profit_percent",
            "risk_profile",
            "strategy_variant",
            "timeframe",
        ]
        if key in candidate
    }

    parameters_json = json.dumps(
        parameters,
        sort_keys=True,
    )

    strategy_key = (
        f"{source}|"
        f"{name}|"
        f"{symbol}|"
        f"{parameters_json}"
    )

    return StrategyRecord(
        strategy_key=(
            strategy_key
        ),

        strategy_name=(
            name
        ),

        role=(
            role
        ),

        symbol=(
            symbol
        ),

        source_report=(
            source
        ),

        score=round(
            score,
            4,
        ),

        return_percent=round(
            return_percent,
            4,
        ),

        drawdown_percent=round(
            drawdown_percent,
            4,
        ),

        profit_factor=round(
            profit_factor,
            4,
        ),

        consistency_percent=round(
            consistency_percent,
            4,
        ),

        trade_count=(
            trade_count
        ),

        status=(
            status
        ),

        parameters_json=(
            parameters_json
        ),
    )


# ============================================================
# STRATEGY EXTRACTION
# ============================================================

def extract_candidates(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    """
    Extract STRATEGY candidates only.

    Scanner keys such as top_longs and top_shorts
    are intentionally excluded.
    """

    candidate_keys = [
        "top_finalists",
        "leaderboard",
        "strategies",
        "candidates",
    ]

    for key in candidate_keys:

        value = payload.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return [
                candidate
                for candidate in value
                if isinstance(
                    candidate,
                    dict,
                )
            ]

    return []


# ============================================================
# DATABASE
# ============================================================

def initialize_database() -> None:

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_key TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                role TEXT NOT NULL,
                symbol TEXT NOT NULL,
                source_report TEXT NOT NULL,
                score REAL NOT NULL,
                return_percent REAL NOT NULL,
                drawdown_percent REAL NOT NULL,
                profit_factor REAL NOT NULL,
                consistency_percent REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                parameters_json TEXT NOT NULL
            )
            """
        )

        connection.commit()


# ============================================================
# LEGACY CLEANUP
# ============================================================

def remove_invalid_legacy_records() -> dict[str, int]:

    """
    Remove records incorrectly imported by earlier versions.

    Specifically:
    - Championship Scanner records
    - Unknown Strategy records
    """

    initialize_database()

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        scanner_count = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM strategies
                WHERE source_report LIKE '%championship_scanner%'
                """
            ).fetchone()[0]
        )

        unknown_count = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM strategies
                WHERE LOWER(TRIM(strategy_name))
                      = 'unknown strategy'
                """
            ).fetchone()[0]
        )

        connection.execute(
            """
            DELETE FROM strategies
            WHERE source_report LIKE '%championship_scanner%'
            """
        )

        connection.execute(
            """
            DELETE FROM strategies
            WHERE LOWER(TRIM(strategy_name))
                  = 'unknown strategy'
            """
        )

        connection.commit()

    return {
        "legacy_scanner_records_removed": int(
            scanner_count
        ),

        "unknown_strategy_records_removed": int(
            unknown_count
        ),
    }


# ============================================================
# SAVE / UPSERT
# ============================================================

def save_records(
    records: list[StrategyRecord],
) -> None:

    initialize_database()

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        for record in records:

            connection.execute(
                """
                INSERT INTO strategies (
                    strategy_key,
                    strategy_name,
                    role,
                    symbol,
                    source_report,
                    score,
                    return_percent,
                    drawdown_percent,
                    profit_factor,
                    consistency_percent,
                    trade_count,
                    status,
                    parameters_json
                )
                VALUES (
                    :strategy_key,
                    :strategy_name,
                    :role,
                    :symbol,
                    :source_report,
                    :score,
                    :return_percent,
                    :drawdown_percent,
                    :profit_factor,
                    :consistency_percent,
                    :trade_count,
                    :status,
                    :parameters_json
                )
                ON CONFLICT(strategy_key)
                DO UPDATE SET
                    strategy_name =
                        excluded.strategy_name,

                    role =
                        excluded.role,

                    symbol =
                        excluded.symbol,

                    source_report =
                        excluded.source_report,

                    score =
                        excluded.score,

                    return_percent =
                        excluded.return_percent,

                    drawdown_percent =
                        excluded.drawdown_percent,

                    profit_factor =
                        excluded.profit_factor,

                    consistency_percent =
                        excluded.consistency_percent,

                    trade_count =
                        excluded.trade_count,

                    status =
                        excluded.status,

                    parameters_json =
                        excluded.parameters_json
                """,
                asdict(
                    record
                ),
            )

        connection.commit()


# ============================================================
# REPORT BUILDING
# ============================================================

def build_reports(
    records_imported: int,
    cleanup_summary: dict[str, int],
) -> dict[str, Any]:

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialize_database()

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        frame = pd.read_sql_query(
            """
            SELECT *
            FROM strategies
            ORDER BY
                score DESC,
                profit_factor DESC,
                drawdown_percent ASC
            """,
            connection,
        )

    if frame.empty:

        frame = pd.DataFrame(
            columns=[
                "strategy_key",
                "strategy_name",
                "role",
                "symbol",
                "source_report",
                "score",
                "return_percent",
                "drawdown_percent",
                "profit_factor",
                "consistency_percent",
                "trade_count",
                "status",
                "parameters_json",
            ]
        )

    frame.insert(
        0,
        "rank",
        range(
            1,
            len(
                frame
            )
            + 1,
        ),
    )

    frame.to_csv(
        CSV_REPORT_PATH,
        index=False,
    )

    top_strategies = (
        frame.head(
            100
        ).to_dict(
            orient="records"
        )
    )

    payload = {
        "records_imported": (
            records_imported
        ),

        "strategy_count": len(
            frame
        ),

        "top_strategies": (
            top_strategies
        ),

        "database_path": str(
            DATABASE_PATH
        ),

        "scanner_records_excluded": True,

        "legacy_cleanup": (
            cleanup_summary
        ),

        "production_strategy_changed": False,

        "market_request_made": False,

        "order_submitted": False,
    }

    JSON_REPORT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return payload


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build the research-only Strategy Hall of Fame."
        )
    )

    parser.add_argument(
        "--reports",
        nargs="*",
        default=(
            DEFAULT_REPORTS
        ),
    )

    arguments = (
        parser.parse_args()
    )

    cleanup_summary = (
        remove_invalid_legacy_records()
    )

    records: list[
        StrategyRecord
    ] = []

    skipped_candidates = 0

    scanner_reports_skipped = 0

    for report_name in arguments.reports:

        normalized_report_name = (
            str(
                report_name
            )
            .replace(
                "\\",
                "/",
            )
            .lower()
        )

        # Additional protection:
        # never allow scanner output into the Hall of Fame.
        if (
            "scanner/"
            in normalized_report_name
            or "championship_scanner"
            in normalized_report_name
        ):

            scanner_reports_skipped += 1

            continue

        path = Path(
            report_name
        )

        payload = load_json(
            path
        )

        if payload is None:
            continue

        candidates = (
            extract_candidates(
                payload
            )
        )

        for candidate in candidates:

            record = (
                candidate_to_record(
                    candidate=(
                        candidate
                    ),
                    source=(
                        str(
                            path
                        )
                    ),
                )
            )

            if record is None:

                skipped_candidates += 1

                continue

            records.append(
                record
            )

    save_records(
        records
    )

    output = build_reports(
        records_imported=(
            len(
                records
            )
        ),
        cleanup_summary=(
            cleanup_summary
        ),
    )

    print(
        "Strategy Hall of Fame v1"
    )

    print(
        json.dumps(
            {
                "records_imported": (
                    len(
                        records
                    )
                ),

                "strategy_count": (
                    output[
                        "strategy_count"
                    ]
                ),

                "skipped_invalid_candidates": (
                    skipped_candidates
                ),

                "scanner_reports_skipped": (
                    scanner_reports_skipped
                ),

                "legacy_cleanup": (
                    cleanup_summary
                ),

                "top_strategies": (
                    output[
                        "top_strategies"
                    ][
                        :10
                    ]
                ),

                "scanner_records_excluded": True,

                "production_strategy_changed": False,

                "market_request_made": False,

                "order_submitted": False,
            },
            indent=2,
        )
    )

    print(
        "Scanner opportunities were excluded "
        "from the Strategy Hall of Fame."
    )

    print(
        "Production strategy was not changed."
    )

    print(
        "No market request was made."
    )

    print(
        "No order was submitted."
    )


if __name__ == "__main__":
    main()