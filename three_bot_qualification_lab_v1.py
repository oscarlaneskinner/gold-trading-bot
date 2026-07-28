"""Three-Bot Qualification Lab v1.

Qualifies one strategy for each role:
- GLD specialist,
- long opportunity strategy,
- short opportunity strategy.

This package:
- imports the verified GLD finalist,
- imports the strongest eligible long strategy from the Hall of Fame,
- runs a dedicated short-side research arena,
- updates the Hall of Fame database,
- creates a three-role qualification report.

Research only:
- no trading client,
- no market request,
- no order submission,
- no production-strategy changes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QualifiedStrategy:
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


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_hall_table(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
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


def upsert_strategy(database_path: Path, strategy: QualifiedStrategy) -> None:
    ensure_hall_table(database_path)
    with sqlite3.connect(database_path) as connection:
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
                status=excluded.status,
                parameters_json=excluded.parameters_json
            """,
            asdict(strategy),
        )


def import_verified_gld(config: dict[str, Any], database_path: Path) -> QualifiedStrategy:
    item = config["verified_gld"]
    params = item["parameters"]

    strategy = QualifiedStrategy(
        strategy_key=f"verified-gld|{item['strategy_name']}|GLD",
        strategy_name=item["strategy_name"],
        role="GLD",
        symbol="GLD",
        source_report="TradingView verified GLD finalist",
        score=float(item["score"]),
        return_percent=float(item["return_percent"]),
        drawdown_percent=float(item["drawdown_percent"]),
        profit_factor=float(item["profit_factor"]),
        consistency_percent=float(item["consistency_percent"]),
        trade_count=int(item["trade_count"]),
        status="QUALIFIED_GLD",
        parameters_json=json.dumps(params, sort_keys=True),
    )

    upsert_strategy(database_path, strategy)
    return strategy


def load_best_long(database_path: Path, rules: dict[str, Any]) -> QualifiedStrategy | None:
    if not database_path.exists():
        return None

    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT * FROM strategies
            WHERE role = 'LONG'
            ORDER BY score DESC, profit_factor DESC, drawdown_percent ASC
            """,
            connection,
        )

    if frame.empty:
        return None

    eligible = frame[
        (frame["score"] >= float(rules["minimum_score"]))
        & (frame["profit_factor"] >= float(rules["minimum_profit_factor"]))
        & (frame["drawdown_percent"] <= float(rules["maximum_drawdown_percent"]))
        & (frame["consistency_percent"] >= float(rules["minimum_consistency_percent"]))
        & (frame["trade_count"] >= int(rules["minimum_trade_count"]))
    ]

    if eligible.empty:
        return None

    row = eligible.iloc[0]

    strategy = QualifiedStrategy(
        strategy_key=str(row["strategy_key"]),
        strategy_name=str(row["strategy_name"]),
        role="LONG",
        symbol=str(row["symbol"]),
        source_report=str(row["source_report"]),
        score=float(row["score"]),
        return_percent=float(row["return_percent"]),
        drawdown_percent=float(row["drawdown_percent"]),
        profit_factor=float(row["profit_factor"]),
        consistency_percent=float(row["consistency_percent"]),
        trade_count=int(row["trade_count"]),
        status="QUALIFIED_LONG",
        parameters_json=str(row["parameters_json"]),
    )

    upsert_strategy(database_path, strategy)
    return strategy


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for column in frame.columns:
        key = column.strip().lower()
        if key in {"date", "time", "datetime", "timestamp"}:
            rename[column] = "date"
        elif key in {"open", "high", "low", "close", "volume"}:
            rename[column] = key

    frame = frame.rename(columns=rename)

    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close, high, low = df["close"], df["high"], df["low"]

    previous_close = close.shift(1)
    df["true_range"] = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    return df


def short_signals(
    df: pd.DataFrame,
    sensitivity: float,
    atr_period: int,
) -> tuple[pd.Series, pd.Series]:
    atr = rma(df["true_range"], atr_period)
    trailing = np.full(len(df), np.nan)
    close = df["close"]

    for index in range(len(df)):
        source = float(close.iloc[index])
        distance = float(atr.iloc[index]) * sensitivity if np.isfinite(atr.iloc[index]) else 0.0
        previous = source if index == 0 or not np.isfinite(trailing[index - 1]) else trailing[index - 1]
        previous_source = source if index == 0 else float(close.iloc[index - 1])

        if source < previous and previous_source < previous:
            trailing[index] = min(previous, source + distance)
        elif source > previous and previous_source > previous:
            trailing[index] = max(previous, source - distance)
        elif source < previous:
            trailing[index] = source + distance
        else:
            trailing[index] = source - distance

    stop = pd.Series(trailing, index=df.index)

    entry = (
        (close < stop)
        & (close.shift(1) >= stop.shift(1))
        & (close < df["ema200"])
        & (df["ema20"] < df["ema50"])
        & (df["rsi14"] < 50)
    )

    exit_signal = (
        (close > stop)
        & (close.shift(1) <= stop.shift(1))
    )

    return entry.fillna(False), exit_signal.fillna(False)


def simulate_short(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    start_capital: float,
    position_percent: float,
    hold: int,
    stop_percent: float,
    target_percent: float,
) -> dict[str, float]:
    cash = start_capital
    entry_price = 0.0
    shares = 0.0
    entry_index = -1
    trades: list[float] = []
    equity_curve: list[float] = []

    for index, row in df.iterrows():
        close = float(row["close"])

        if shares == 0 and bool(entries.iloc[index]):
            allocation = cash * position_percent / 100
            shares = allocation / close
            entry_price = close
            entry_index = index

        elif shares > 0:
            stop_price = entry_price * (1 + stop_percent / 100)
            target_price = entry_price * (1 - target_percent / 100)
            exit_price = None

            if float(row["high"]) >= stop_price:
                exit_price = stop_price
            elif float(row["low"]) <= target_price:
                exit_price = target_price
            elif bool(exits.iloc[index]) or index - entry_index >= hold:
                exit_price = close

            if exit_price is not None:
                pnl = shares * (entry_price - exit_price)
                cash += pnl
                trades.append(pnl)
                shares = 0.0
                entry_price = 0.0
                entry_index = -1

        unrealized = shares * (entry_price - close) if shares > 0 else 0.0
        equity_curve.append(cash + unrealized)

    if shares > 0:
        final_close = float(df.iloc[-1]["close"])
        pnl = shares * (entry_price - final_close)
        cash += pnl
        trades.append(pnl)
        equity_curve[-1] = cash

    returns = 100 * (cash - start_capital) / start_capital

    equity = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity) if len(equity) else np.asarray([start_capital])
    drawdowns = peaks - equity if len(equity) else np.asarray([0.0])
    index = int(drawdowns.argmax())
    drawdown_percent = 100 * float(drawdowns.max()) / peaks[index] if peaks[index] else 0.0

    wins = [trade for trade in trades if trade > 0]
    losses = [trade for trade in trades if trade < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        "return_percent": returns,
        "drawdown_percent": drawdown_percent,
        "profit_factor": profit_factor,
        "trade_count": len(trades),
    }


def evaluate_short_candidate(
    datasets: dict[str, pd.DataFrame],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    per_symbol = []

    for symbol, full in datasets.items():
        split = int(len(full) * float(config["short_arena"]["train_fraction"]))
        test = full.iloc[split:].reset_index(drop=True)

        entries, exits = short_signals(
            test,
            candidate["sensitivity"],
            candidate["atr_period"],
        )

        result = simulate_short(
            test,
            entries,
            exits,
            float(config["starting_capital"]),
            10.0,
            candidate["max_bars_held"],
            candidate["stop_loss_percent"],
            candidate["take_profit_percent"],
        )

        passed = (
            result["return_percent"] > 0
            and result["profit_factor"] > 1.05
            and result["trade_count"] >= 10
        )

        per_symbol.append({
            "symbol": symbol,
            **result,
            "passed": passed,
        })

    returns = [row["return_percent"] for row in per_symbol]
    drawdowns = [row["drawdown_percent"] for row in per_symbol]
    profit_factors = [min(row["profit_factor"], 5.0) for row in per_symbol]
    trade_counts = [row["trade_count"] for row in per_symbol]

    symbols_passed = sum(row["passed"] for row in per_symbol)
    consistency = 100 * symbols_passed / len(per_symbol) if per_symbol else 0.0

    median_return = float(np.median(returns)) if returns else 0.0
    median_drawdown = float(np.median(drawdowns)) if drawdowns else 0.0
    median_profit_factor = float(np.median(profit_factors)) if profit_factors else 0.0
    median_trade_count = float(np.median(trade_counts)) if trade_counts else 0.0
    worst_return = min(returns) if returns else 0.0

    score = (
        median_return * 3.0
        + median_profit_factor * 8.0
        + consistency * 0.20
        + max(worst_return, -10.0) * 0.75
        - median_drawdown * 1.5
    )

    return {
        **candidate,
        "symbols_tested": len(per_symbol),
        "symbols_passed": symbols_passed,
        "median_return_percent": round(median_return, 4),
        "median_drawdown_percent": round(median_drawdown, 4),
        "median_profit_factor": round(median_profit_factor, 4),
        "median_trade_count": round(median_trade_count, 2),
        "consistency_percent": round(consistency, 2),
        "score": round(score, 4),
        "per_symbol": per_symbol,
    }


def run_short_arena(config: dict[str, Any], data_dir: Path) -> dict[str, Any] | None:
    arena = config["short_arena"]
    datasets = {}

    for symbol in arena["symbols"]:
        path = data_dir / f"{symbol}_1D.csv"
        if path.exists():
            datasets[symbol] = prepare(normalize_columns(pd.read_csv(path)))

    if not datasets:
        return None

    candidates = []
    counter = 1

    for values in itertools.product(
        arena["sensitivities"],
        arena["atr_periods"],
        arena["max_bars_held"],
        arena["stop_loss_percent"],
        arena["take_profit_percent"],
    ):
        sensitivity, atr_period, hold, stop, target = values
        candidate = {
            "candidate_id": f"SHORT-{counter:04d}",
            "sensitivity": float(sensitivity),
            "atr_period": int(atr_period),
            "max_bars_held": int(hold),
            "stop_loss_percent": float(stop),
            "take_profit_percent": float(target),
        }
        candidates.append(evaluate_short_candidate(datasets, candidate, config))
        counter += 1

    candidates.sort(key=lambda row: row["score"], reverse=True)

    qualified = [
        row for row in candidates
        if row["symbols_passed"] >= int(arena["minimum_symbols_passed"])
        and row["median_return_percent"] > 0
        and row["median_profit_factor"] > 1.05
    ]

    return {
        "candidate_count": len(candidates),
        "qualified_count": len(qualified),
        "top_candidate": qualified[0] if qualified else None,
        "leaderboard": candidates[:100],
    }


def register_short(
    short_result: dict[str, Any] | None,
    database_path: Path,
) -> QualifiedStrategy | None:
    if not short_result or not short_result.get("top_candidate"):
        return None

    top = short_result["top_candidate"]

    strategy = QualifiedStrategy(
        strategy_key=f"qualified-short|{top['candidate_id']}",
        strategy_name=top["candidate_id"],
        role="SHORT",
        symbol="MULTI",
        source_report="Three-Bot Qualification Lab short arena",
        score=float(top["score"]),
        return_percent=float(top["median_return_percent"]),
        drawdown_percent=float(top["median_drawdown_percent"]),
        profit_factor=float(top["median_profit_factor"]),
        consistency_percent=float(top["consistency_percent"]),
        trade_count=int(top["median_trade_count"]),
        status="QUALIFIED_SHORT",
        parameters_json=json.dumps(
            {
                "sensitivity": top["sensitivity"],
                "atr_period": top["atr_period"],
                "max_bars_held": top["max_bars_held"],
                "stop_loss_percent": top["stop_loss_percent"],
                "take_profit_percent": top["take_profit_percent"],
            },
            sort_keys=True,
        ),
    )

    upsert_strategy(database_path, strategy)
    return strategy


def strategy_summary(strategy: QualifiedStrategy | None) -> dict[str, Any] | None:
    return asdict(strategy) if strategy is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/three_bot_qualification_lab_v1.json")
    parser.add_argument("--data-dir", default="data/scanner")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    database_path = Path(config["hall_of_fame_database"])

    gld = import_verified_gld(config, database_path)
    long_strategy = load_best_long(database_path, config["qualification_rules"])
    short_arena = run_short_arena(config, Path(args.data_dir))
    short_strategy = register_short(short_arena, database_path)

    roles = {
        "GLD": strategy_summary(gld),
        "LONG": strategy_summary(long_strategy),
        "SHORT": strategy_summary(short_strategy),
    }

    qualified_role_count = sum(value is not None for value in roles.values())

    output = {
        "qualified_role_count": qualified_role_count,
        "all_three_roles_qualified": qualified_role_count == 3,
        "roles": roles,
        "short_arena": short_arena,
        "next_stage": (
            "THREE_BOT_SHADOW_CONTROLLER"
            if qualified_role_count == 3
            else "CONTINUE_QUALIFICATION"
        ),
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    report_dir = Path("reports/qualification")
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "three_bot_qualification_lab_v1.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    role_rows = [
        value
        for value in roles.values()
        if value is not None
    ]
    pd.DataFrame(role_rows).to_csv(
        report_dir / "qualified_three_bot_roles.csv",
        index=False,
    )

    if short_arena is not None:
        pd.DataFrame(short_arena["leaderboard"]).drop(
            columns=["per_symbol"],
            errors="ignore",
        ).to_csv(
            report_dir / "short_arena_leaderboard.csv",
            index=False,
        )

    print("Three-Bot Qualification Lab v1")
    print(json.dumps({
        "qualified_role_count": qualified_role_count,
        "all_three_roles_qualified": output["all_three_roles_qualified"],
        "roles": roles,
        "short_candidate_count": (
            short_arena["candidate_count"]
            if short_arena
            else 0
        ),
        "short_qualified_count": (
            short_arena["qualified_count"]
            if short_arena
            else 0
        ),
        "next_stage": output["next_stage"],
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }, indent=2))
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
