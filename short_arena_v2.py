"""Short Arena v2 Championship Edition.

Research-only short strategy tournament.

Features:
- multiple short-entry families,
- cross-market train/test evaluation,
- relative weakness versus SPY,
- realistic short P/L accounting,
- commissions and slippage,
- top-100 leaderboard,
- automatic Hall of Fame promotion for qualified candidates,
- no trading client and no order submission.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    entry_model: str
    lookback_period: int
    ema_fast_period: int
    ema_slow_period: int
    rsi_threshold: int
    relative_weakness_period: int
    atr_period: int
    atr_entry_multiplier: float
    max_bars_held: int
    stop_loss_percent: float
    take_profit_percent: float


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def prepare_indicators(
    frame: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    candidate: Candidate,
) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["ema_fast"] = close.ewm(span=candidate.ema_fast_period, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=candidate.ema_slow_period, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = rma(true_range, candidate.atr_period)

    df["support"] = low.rolling(candidate.lookback_period).min().shift(1)
    df["resistance"] = high.rolling(candidate.lookback_period).max().shift(1)
    df["previous_high"] = high.shift(1)
    df["previous_low"] = low.shift(1)

    df["volume_ma20"] = volume.rolling(20).mean()
    df["relative_volume"] = volume / df["volume_ma20"].replace(0, np.nan)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["return_n"] = close.pct_change(candidate.relative_weakness_period)
    if benchmark_returns is not None:
        aligned = benchmark_returns.reindex(df.index)
        df["relative_weakness"] = df["return_n"] - aligned
    else:
        df["relative_weakness"] = df["return_n"]

    return df


def build_entry_signal(df: pd.DataFrame, candidate: Candidate) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    ema_fast = df["ema_fast"]
    ema_slow = df["ema_slow"]
    rsi = df["rsi14"]
    atr = df["atr"]

    bearish_regime = (close < ema_slow) & (ema_fast < ema_slow)

    if candidate.entry_model == "Breakdown":
        signal = bearish_regime & (close < df["support"])

    elif candidate.entry_model == "FailedRally":
        touched_fast = high >= ema_fast
        rejected = close < ema_fast
        signal = bearish_regime & touched_fast & rejected & (close < open_)

    elif candidate.entry_model == "EMARejection":
        touched_slow = high >= ema_slow
        rejected = close < ema_slow
        signal = (close < ema_slow) & touched_slow & rejected & (rsi < candidate.rsi_threshold)

    elif candidate.entry_model == "RelativeWeakness":
        signal = (
            bearish_regime
            & (df["relative_weakness"] < 0)
            & (rsi < candidate.rsi_threshold)
        )

    elif candidate.entry_model == "LowerHigh":
        lower_high = high < df["previous_high"]
        lower_low = low < df["previous_low"]
        signal = bearish_regime & lower_high & lower_low

    elif candidate.entry_model == "DonchianBreakdown":
        signal = close < df["support"]

    elif candidate.entry_model == "ATRBreakdown":
        threshold = df["previous_low"] - atr * candidate.atr_entry_multiplier
        signal = bearish_regime & (close < threshold)

    elif candidate.entry_model == "VolumeReversal":
        wide_red = (close < open_) & ((open_ - close) > atr * 0.5)
        signal = bearish_regime & wide_red & (df["relative_volume"] > 1.2)

    elif candidate.entry_model == "RSIFailure":
        failure = (rsi < candidate.rsi_threshold) & (rsi.shift(1) >= candidate.rsi_threshold)
        signal = bearish_regime & failure

    elif candidate.entry_model == "MACDBearCross":
        cross = (df["macd"] < df["macd_signal"]) & (
            df["macd"].shift(1) >= df["macd_signal"].shift(1)
        )
        signal = bearish_regime & cross

    else:
        raise ValueError(f"Unsupported entry model: {candidate.entry_model}")

    return signal.fillna(False)


def build_exit_signal(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["ema_fast"])
        | (
            (df["macd"] > df["macd_signal"])
            & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        )
    ).fillna(False)


def simulate_short(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    candidate: Candidate,
    config: dict[str, Any],
) -> dict[str, Any]:
    initial = float(config["starting_capital"])
    cash = initial
    shares = 0.0
    entry_price = 0.0
    entry_index = -1
    entry_fee = 0.0

    trades: list[float] = []
    equity_curve: list[float] = []

    position_fraction = float(config["position_percent"]) / 100.0
    commission_rate = float(config["commission_percent"]) / 100.0
    slippage = float(config["slippage_dollars"])

    for index, row in df.iterrows():
        close = float(row["close"])

        if shares == 0 and bool(entries.iloc[index]):
            fill = close - slippage
            allocation = cash * position_fraction
            shares = allocation / fill if fill > 0 else 0.0
            entry_price = fill
            entry_index = index
            entry_fee = allocation * commission_rate

        elif shares > 0:
            stop_price = entry_price * (1 + candidate.stop_loss_percent / 100.0)
            target_price = entry_price * (1 - candidate.take_profit_percent / 100.0)
            exit_price = None

            if float(row["high"]) >= stop_price:
                exit_price = stop_price + slippage
            elif float(row["low"]) <= target_price:
                exit_price = target_price + slippage
            elif bool(exits.iloc[index]) or index - entry_index >= candidate.max_bars_held:
                exit_price = close + slippage

            if exit_price is not None:
                gross_pnl = shares * (entry_price - exit_price)
                exit_fee = shares * exit_price * commission_rate
                net_pnl = gross_pnl - entry_fee - exit_fee
                cash += net_pnl
                trades.append(float(net_pnl))
                shares = 0.0
                entry_price = 0.0
                entry_index = -1
                entry_fee = 0.0

        unrealized = shares * (entry_price - close) if shares > 0 else 0.0
        equity_curve.append(cash + unrealized)

    if shares > 0:
        exit_price = float(df.iloc[-1]["close"]) + slippage
        gross_pnl = shares * (entry_price - exit_price)
        exit_fee = shares * exit_price * commission_rate
        net_pnl = gross_pnl - entry_fee - exit_fee
        cash += net_pnl
        trades.append(float(net_pnl))
        if equity_curve:
            equity_curve[-1] = cash

    equity = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity) if len(equity) else np.asarray([initial])
    drawdowns = peaks - equity if len(equity) else np.asarray([0.0])
    max_index = int(drawdowns.argmax())
    max_drawdown_percent = (
        100 * float(drawdowns.max()) / peaks[max_index]
        if peaks[max_index]
        else 0.0
    )

    wins = [trade for trade in trades if trade > 0]
    losses = [trade for trade in trades if trade < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (99.0 if gross_profit > 0 else 0.0)
    )

    return {
        "return_percent": 100 * (cash - initial) / initial,
        "drawdown_percent": max_drawdown_percent,
        "profit_factor": profit_factor,
        "win_rate": 100 * len(wins) / len(trades) if trades else 0.0,
        "trade_count": len(trades),
    }


def load_datasets(data_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    datasets = {}
    for symbol in symbols:
        path = data_dir / f"{symbol}_1D.csv"
        if path.exists():
            datasets[symbol] = normalize_columns(pd.read_csv(path))
    return datasets


def evaluate_candidate(
    candidate: Candidate,
    datasets: dict[str, pd.DataFrame],
    benchmark_returns: pd.Series | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    per_symbol = []

    for symbol, raw in datasets.items():
        prepared = prepare_indicators(raw, benchmark_returns, candidate)
        split = int(len(prepared) * float(config["train_fraction"]))
        test = prepared.iloc[split:].reset_index(drop=True)

        entries = build_entry_signal(test, candidate)
        exits = build_exit_signal(test)
        result = simulate_short(test, entries, exits, candidate, config)

        passed = (
            result["trade_count"] >= int(config["minimum_trades_per_symbol"])
            and result["return_percent"] > 0
            and result["profit_factor"] >= float(config["minimum_profit_factor"])
        )

        per_symbol.append(
            {
                "symbol": symbol,
                **{key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()},
                "passed": passed,
            }
        )

    returns = [row["return_percent"] for row in per_symbol]
    drawdowns = [row["drawdown_percent"] for row in per_symbol]
    profit_factors = [min(row["profit_factor"], 5.0) for row in per_symbol]
    trade_counts = [row["trade_count"] for row in per_symbol]

    symbols_passed = sum(row["passed"] for row in per_symbol)
    consistency = 100 * symbols_passed / len(per_symbol) if per_symbol else 0.0

    median_return = float(np.median(returns)) if returns else 0.0
    mean_return = float(np.mean(returns)) if returns else 0.0
    worst_return = float(np.min(returns)) if returns else 0.0
    median_drawdown = float(np.median(drawdowns)) if drawdowns else 0.0
    worst_drawdown = float(np.max(drawdowns)) if drawdowns else 0.0
    median_profit_factor = float(np.median(profit_factors)) if profit_factors else 0.0
    median_trade_count = float(np.median(trade_counts)) if trade_counts else 0.0

    score = (
        median_return * 3.0
        + mean_return * 1.5
        + median_profit_factor * 8.0
        + consistency * 0.20
        + max(worst_return, -10.0) * 0.75
        - median_drawdown * 1.5
        - worst_drawdown * 0.5
    )

    status = (
        "QUALIFIED_SHORT"
        if (
            symbols_passed >= int(config["minimum_symbols_passed"])
            and median_return > 0
            and median_profit_factor >= float(config["minimum_profit_factor"])
            and median_drawdown <= float(config["maximum_median_drawdown_percent"])
        )
        else "REJECT"
    )

    return {
        **asdict(candidate),
        "symbols_tested": len(per_symbol),
        "symbols_passed": symbols_passed,
        "median_return_percent": round(median_return, 4),
        "mean_return_percent": round(mean_return, 4),
        "worst_return_percent": round(worst_return, 4),
        "median_drawdown_percent": round(median_drawdown, 4),
        "worst_drawdown_percent": round(worst_drawdown, 4),
        "median_profit_factor": round(median_profit_factor, 4),
        "median_trade_count": round(median_trade_count, 2),
        "consistency_percent": round(consistency, 2),
        "score": round(score, 4),
        "status": status,
        "per_symbol": per_symbol,
    }


def build_candidates(config: dict[str, Any]) -> list[Candidate]:
    candidates = []
    counter = 1

    for values in itertools.product(
        config["entry_models"],
        config["lookback_periods"],
        config["ema_fast_periods"],
        config["ema_slow_periods"],
        config["rsi_thresholds"],
        config["relative_weakness_periods"],
        config["atr_periods"],
        config["atr_entry_multipliers"],
        config["max_bars_held"],
        config["stop_loss_percent"],
        config["take_profit_percent"],
    ):
        (
            entry_model,
            lookback,
            ema_fast,
            ema_slow,
            rsi_threshold,
            relative_weakness_period,
            atr_period,
            atr_multiplier,
            max_bars,
            stop,
            target,
        ) = values

        if ema_fast >= ema_slow:
            continue

        candidates.append(
            Candidate(
                candidate_id=f"SHORTV2-{counter:06d}",
                entry_model=str(entry_model),
                lookback_period=int(lookback),
                ema_fast_period=int(ema_fast),
                ema_slow_period=int(ema_slow),
                rsi_threshold=int(rsi_threshold),
                relative_weakness_period=int(relative_weakness_period),
                atr_period=int(atr_period),
                atr_entry_multiplier=float(atr_multiplier),
                max_bars_held=int(max_bars),
                stop_loss_percent=float(stop),
                take_profit_percent=float(target),
            )
        )
        counter += 1

    return candidates


def ensure_hall_of_fame_table(database_path: Path) -> None:
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


def promote_best_candidate(best: dict[str, Any] | None) -> bool:
    if best is None or best["status"] != "QUALIFIED_SHORT":
        return False

    database_path = Path("data/strategy_hall_of_fame.sqlite3")
    ensure_hall_of_fame_table(database_path)

    params = {
        key: best[key]
        for key in [
            "entry_model",
            "lookback_period",
            "ema_fast_period",
            "ema_slow_period",
            "rsi_threshold",
            "relative_weakness_period",
            "atr_period",
            "atr_entry_multiplier",
            "max_bars_held",
            "stop_loss_percent",
            "take_profit_percent",
        ]
    }

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO strategies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            (
                f"short-arena-v2|{best['candidate_id']}",
                best["candidate_id"],
                "SHORT",
                "MULTI",
                "reports/short_arena_v2/short_arena_v2_results.json",
                float(best["score"]),
                float(best["median_return_percent"]),
                float(best["median_drawdown_percent"]),
                float(best["median_profit_factor"]),
                float(best["consistency_percent"]),
                int(best["median_trade_count"]),
                "QUALIFIED_SHORT",
                json.dumps(params, sort_keys=True),
            ),
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/short_arena_v2_championship.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    data_dir = Path(args.data_dir)

    datasets = load_datasets(data_dir, config["symbols"])
    if not datasets:
        raise SystemExit("No symbol CSV files were found.")

    benchmark = datasets.get(config["benchmark_symbol"])
    benchmark_returns = (
        benchmark["close"].pct_change(20)
        if benchmark is not None
        else None
    )

    candidates = build_candidates(config)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    results = []

    for index, candidate in enumerate(candidates, start=1):
        results.append(
            evaluate_candidate(candidate, datasets, benchmark_returns, config)
        )

        if index % 100 == 0 or index == len(candidates):
            print(f"Completed {index}/{len(candidates)} candidates...")

    results.sort(
        key=lambda row: (row["status"] == "QUALIFIED_SHORT", row["score"]),
        reverse=True,
    )

    qualified = [row for row in results if row["status"] == "QUALIFIED_SHORT"]
    top = qualified[: int(config["top_n"])]
    best = top[0] if top else None
    promoted = promote_best_candidate(best)

    report_dir = Path("reports/short_arena_v2")
    report_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "symbols_loaded": sorted(datasets),
        "candidate_count": len(results),
        "qualified_count": len(qualified),
        "best_candidate": best,
        "top_100": top,
        "promoted_to_hall_of_fame": promoted,
        "next_stage": (
            "THREE_BOT_SHADOW_CONTROLLER"
            if promoted
            else "CONTINUE_SHORT_RESEARCH"
        ),
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    (report_dir / "short_arena_v2_results.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    summary_columns = [
        "candidate_id",
        "entry_model",
        "lookback_period",
        "ema_fast_period",
        "ema_slow_period",
        "rsi_threshold",
        "relative_weakness_period",
        "atr_period",
        "atr_entry_multiplier",
        "max_bars_held",
        "stop_loss_percent",
        "take_profit_percent",
        "symbols_passed",
        "median_return_percent",
        "mean_return_percent",
        "worst_return_percent",
        "median_drawdown_percent",
        "worst_drawdown_percent",
        "median_profit_factor",
        "median_trade_count",
        "consistency_percent",
        "score",
        "status",
    ]

    leaderboard = pd.DataFrame(results)
    if leaderboard.empty:
        leaderboard = pd.DataFrame(columns=summary_columns)
    else:
        leaderboard = leaderboard.reindex(columns=summary_columns)

    leaderboard.to_csv(
        report_dir / "short_arena_v2_leaderboard.csv",
        index=False,
    )

    top_frame = pd.DataFrame(top)
    if top_frame.empty:
        top_frame = pd.DataFrame(columns=summary_columns)
    else:
        top_frame = top_frame.reindex(columns=summary_columns)

    top_frame.to_csv(
        report_dir / "short_arena_v2_top_100.csv",
        index=False,
    )

    print("Short Arena v2 Championship Edition")
    print(
        json.dumps(
            {
                "symbols_loaded": output["symbols_loaded"],
                "candidate_count": output["candidate_count"],
                "qualified_count": output["qualified_count"],
                "best_candidate": output["best_candidate"],
                "promoted_to_hall_of_fame": output["promoted_to_hall_of_fame"],
                "next_stage": output["next_stage"],
                "production_strategy_changed": False,
                "market_request_made": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
