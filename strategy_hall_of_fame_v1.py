"""Strategy Hall of Fame v1.

Collects candidates from existing research reports and creates one permanent,
deduplicated ranking table.

Research only. No market request or order submission.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


DATABASE_PATH = Path("data/strategy_hall_of_fame.sqlite3")
REPORT_DIR = Path("reports/hall_of_fame")


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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_role(name: str, symbol: str) -> str:
    text = f"{name} {symbol}".lower()
    if "short" in text or "bear" in text or "breakdown" in text:
        return "SHORT"
    if symbol.upper() == "GLD" or "gold" in text:
        return "GLD"
    return "LONG"


def candidate_to_record(candidate: dict[str, Any], source: str) -> StrategyRecord:
    name = (
        candidate.get("candidate_id")
        or candidate.get("strategy_name")
        or candidate.get("filter_set")
        or candidate.get("strategy_variant")
        or "Unknown Strategy"
    )
    symbol = str(candidate.get("symbol") or candidate.get("asset") or "MULTI")
    score = float(candidate.get("score", 0.0) or 0.0)
    ret = float(
        candidate.get("median_test_return_percent",
        candidate.get("test_return_percent",
        candidate.get("net_profit_percent", 0.0))) or 0.0
    )
    drawdown = float(
        candidate.get("median_drawdown_percent",
        candidate.get("test_drawdown_percent",
        candidate.get("max_drawdown_percent", 0.0))) or 0.0
    )
    pf = float(
        candidate.get("median_profit_factor",
        candidate.get("test_profit_factor",
        candidate.get("profit_factor", 0.0))) or 0.0
    )
    consistency = float(candidate.get("consistency_percent", 0.0) or 0.0)
    trades = int(
        candidate.get("median_trade_count",
        candidate.get("test_trade_count",
        candidate.get("closed_trades", 0))) or 0
    )
    status = str(candidate.get("status", "RESEARCH"))
    role = infer_role(str(name), symbol)

    params = {
        key: candidate.get(key)
        for key in [
            "filter_set", "sensitivity", "atr_period", "max_bars_held",
            "stop_loss_percent", "take_profit_percent", "risk_profile",
            "strategy_variant", "timeframe",
        ]
        if key in candidate
    }

    strategy_key = f"{source}|{name}|{symbol}|{json.dumps(params, sort_keys=True)}"

    return StrategyRecord(
        strategy_key=strategy_key,
        strategy_name=str(name),
        role=role,
        symbol=symbol,
        source_report=source,
        score=round(score, 4),
        return_percent=round(ret, 4),
        drawdown_percent=round(drawdown, 4),
        profit_factor=round(pf, 4),
        consistency_percent=round(consistency, 4),
        trade_count=trades,
        status=status,
        parameters_json=json.dumps(params, sort_keys=True),
    )


def extract_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ["top_finalists", "leaderboard", "top_longs", "top_shorts"]:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
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


def save_records(records: list[StrategyRecord]) -> None:
    initialize_database()
    with sqlite3.connect(DATABASE_PATH) as connection:
        for record in records:
            connection.execute(
                """
                INSERT INTO strategies VALUES (
                    :strategy_key, :strategy_name, :role, :symbol, :source_report,
                    :score, :return_percent, :drawdown_percent, :profit_factor,
                    :consistency_percent, :trade_count, :status, :parameters_json
                )
                ON CONFLICT(strategy_key) DO UPDATE SET
                    score=excluded.score,
                    return_percent=excluded.return_percent,
                    drawdown_percent=excluded.drawdown_percent,
                    profit_factor=excluded.profit_factor,
                    consistency_percent=excluded.consistency_percent,
                    trade_count=excluded.trade_count,
                    status=excluded.status
                """,
                asdict(record),
            )


def build_reports() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        frame = pd.read_sql_query(
            """
            SELECT * FROM strategies
            ORDER BY score DESC, profit_factor DESC, drawdown_percent ASC
            """,
            connection,
        )

    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.to_csv(REPORT_DIR / "strategy_hall_of_fame.csv", index=False)

    top = frame.head(100).to_dict(orient="records")
    payload = {
        "strategy_count": len(frame),
        "top_strategies": top,
        "database_path": str(DATABASE_PATH),
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }
    (REPORT_DIR / "strategy_hall_of_fame.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        nargs="*",
        default=[
            "reports/arena_v3_results.json",
            "reports/multi_market_strategy_arena_v2.json",
            "reports/strategy_research_arena.json",
            "reports/scanner/championship_scanner_v1.json",
        ],
    )
    args = parser.parse_args()

    records: list[StrategyRecord] = []

    for report_name in args.reports:
        path = Path(report_name)
        payload = load_json(path)
        if payload is None:
            continue
        for candidate in extract_candidates(payload):
            records.append(candidate_to_record(candidate, str(path)))

    save_records(records)
    output = build_reports()

    print("Strategy Hall of Fame v1")
    print(json.dumps({
        "records_imported": len(records),
        "strategy_count": output["strategy_count"],
        "top_strategies": output["top_strategies"][:10],
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
