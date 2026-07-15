"""Automated UT strategy research arena.

Runs hundreds of parameter/filter combinations against a local OHLCV CSV,
uses a chronological train/test split, and ranks candidates using out-of-sample
performance. It does not place orders or alter the production strategy.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ArenaResult:
    candidate_id: str
    filter_set: str
    sensitivity: float
    atr_period: int
    max_bars_held: int
    risk_profile: str
    stop_loss_percent: float
    take_profit_percent: float
    train_return_percent: float
    test_return_percent: float
    test_drawdown_percent: float
    test_profit_factor: float
    test_win_rate: float
    test_trade_count: int
    robustness_ratio: float
    score: float
    status: str


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for column in frame.columns:
        key = column.strip().lower()
        if key in {"time", "date", "datetime", "timestamp"}:
            mapping[column] = "date"
        elif key in {"open", "high", "low", "close", "volume"}:
            mapping[column] = key
    frame = frame.rename(columns=mapping)

    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if "volume" not in frame:
        frame["volume"] = 0.0

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return frame


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close, high, low, volume = result["close"], result["high"], result["low"], result["volume"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["true_range"] = true_range
    result["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    rs = rma(gains, 14) / rma(losses, 14).replace(0, np.nan)
    result["rsi14"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr14 = rma(true_range, 14).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, 14) / atr14
    minus_di = 100 * rma(minus_dm, 14) / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    result["adx"] = rma(dx, 14)
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di

    result["volume_ma20"] = volume.rolling(20).mean()
    result["relative_volume"] = volume / result["volume_ma20"].replace(0, np.nan)

    result["linreg50"] = close.rolling(50).apply(
        lambda values: np.polyval(np.polyfit(np.arange(len(values)), values, 1), len(values) - 1),
        raw=True,
    )
    result["linreg50_prev"] = result["linreg50"].shift(1)

    smooth_open = result["open"].ewm(span=10, adjust=False).mean()
    smooth_high = high.ewm(span=10, adjust=False).mean()
    smooth_low = low.ewm(span=10, adjust=False).mean()
    smooth_close = close.ewm(span=10, adjust=False).mean()
    result["sha_close"] = (smooth_open + smooth_high + smooth_low + smooth_close) / 4
    sha_open = pd.Series(index=result.index, dtype=float)
    for index in result.index:
        if index == 0:
            sha_open.iloc[index] = (smooth_open.iloc[index] + smooth_close.iloc[index]) / 2
        else:
            sha_open.iloc[index] = (sha_open.iloc[index - 1] + result["sha_close"].iloc[index - 1]) / 2
    result["sha_open"] = sha_open

    fast_rsi = rma(gains, 5) / rma(losses, 5).replace(0, np.nan)
    fast_rsi = 100 - 100 / (1 + fast_rsi)
    ema_momentum = 100 * (
        close.ewm(span=5, adjust=False).mean() - close.ewm(span=20, adjust=False).mean()
    ) / close.replace(0, np.nan)
    result["bx_style"] = (fast_rsi - 50).ewm(span=5, adjust=False).mean() + ema_momentum
    return result


def ut_signals(df: pd.DataFrame, sensitivity: float, atr_period: int) -> tuple[pd.Series, pd.Series]:
    atr = rma(df["true_range"], atr_period)
    stop = np.full(len(df), np.nan)
    close = df["close"]

    for i in range(len(df)):
        source = float(close.iloc[i])
        distance = float(atr.iloc[i]) * sensitivity if np.isfinite(atr.iloc[i]) else 0.0
        previous_stop = source if i == 0 or not np.isfinite(stop[i - 1]) else stop[i - 1]
        previous_source = source if i == 0 else float(close.iloc[i - 1])

        if source > previous_stop and previous_source > previous_stop:
            stop[i] = max(previous_stop, source - distance)
        elif source < previous_stop and previous_source < previous_stop:
            stop[i] = min(previous_stop, source + distance)
        elif source > previous_stop:
            stop[i] = source - distance
        else:
            stop[i] = source + distance

    stop_series = pd.Series(stop, index=df.index)
    buy = (close > stop_series) & (close.shift(1) <= stop_series.shift(1))
    sell = (close < stop_series) & (close.shift(1) >= stop_series.shift(1))
    return buy.fillna(False), sell.fillna(False)


def filter_mask(df: pd.DataFrame, name: str) -> pd.Series:
    masks = {
        "Original": pd.Series(True, index=df.index),
        "EMA200": df["close"] > df["ema200"],
        "RSI": (df["rsi14"] > 50) & (df["rsi14"] < 75),
        "MACD": df["macd"] > df["macd_signal"],
        "ADX": (df["adx"] > 20) & (df["plus_di"] > df["minus_di"]),
        "RelativeVolume": df["relative_volume"] > 1,
        "LinearRegression": (df["close"] > df["linreg50"]) & (df["linreg50"] > df["linreg50_prev"]),
        "SmoothedHeikenAshi": (df["sha_close"] > df["sha_open"]) & (df["sha_close"] > df["sha_close"].shift(1)),
        "BXStyle": (df["bx_style"] > 0) & (df["bx_style"] > df["bx_style"].shift(1)),
    }

    parts = name.split("+")
    output = pd.Series(True, index=df.index)
    for part in parts:
        output &= masks[part]
    return output.fillna(False)


def simulate(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    mask: pd.Series,
    config: dict[str, Any],
    max_bars: int,
    stop_percent: float,
    target_percent: float,
) -> dict[str, float]:
    initial = float(config["starting_capital"])
    cash = initial
    quantity = 0.0
    entry_price = 0.0
    entry_index = -1
    trades: list[float] = []
    equity_curve: list[float] = []

    position_fraction = float(config["position_percent"]) / 100
    commission = float(config["commission_percent"]) / 100
    slippage = float(config["slippage_dollars"])

    for i, row in df.iterrows():
        close = float(row["close"])

        if quantity == 0 and bool(entries.iloc[i]) and bool(mask.iloc[i]):
            fill = close + slippage
            notional = cash * position_fraction
            quantity = notional / fill
            cash -= notional + notional * commission
            entry_price = fill
            entry_index = i
        elif quantity > 0:
            stop_price = entry_price * (1 - stop_percent / 100)
            target_price = entry_price * (1 + target_percent / 100)
            exit_price = None

            if float(row["low"]) <= stop_price:
                exit_price = stop_price - slippage
            elif float(row["high"]) >= target_price:
                exit_price = target_price - slippage
            elif bool(exits.iloc[i]) or i - entry_index >= max_bars:
                exit_price = close - slippage

            if exit_price is not None:
                proceeds = quantity * exit_price
                fee = proceeds * commission
                pnl = proceeds - fee - quantity * entry_price
                cash += proceeds - fee
                trades.append(pnl)
                quantity = 0.0
                entry_price = 0.0
                entry_index = -1

        equity_curve.append(cash + quantity * close)

    if quantity > 0:
        proceeds = quantity * (float(df.iloc[-1]["close"]) - slippage)
        fee = proceeds * commission
        trades.append(proceeds - fee - quantity * entry_price)
        cash += proceeds - fee
        equity_curve[-1] = cash

    net = cash - initial
    returns = 100 * net / initial

    equity = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([initial])
    drawdowns = peaks - equity if len(equity) else np.array([0.0])
    max_dd_amount = float(drawdowns.max())
    dd_index = int(drawdowns.argmax())
    max_dd_percent = 100 * max_dd_amount / peaks[dd_index] if peaks[dd_index] else 0.0

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        "return_percent": returns,
        "drawdown_percent": max_dd_percent,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trade_count": len(trades),
    }


def candidate_score(test: dict[str, float], train_return: float, minimum_trades: int) -> tuple[float, float, str]:
    test_return = test["return_percent"]
    drawdown = test["drawdown_percent"]
    trades = int(test["trade_count"])
    pf = min(float(test["profit_factor"]), 5.0)
    robustness = test_return / train_return if train_return > 0 else 0.0
    sample_multiplier = min(trades / minimum_trades, 1.0)

    score = (
        test_return * 2.0
        - drawdown * 1.75
        + pf * 8.0
        + float(test["win_rate"]) * 0.1
        + min(max(robustness, 0.0), 1.5) * 10.0
    ) * sample_multiplier

    status = "FINALIST" if trades >= minimum_trades and test_return > 0 and pf > 1.05 else "REJECT"
    return round(score, 4), round(robustness, 4), status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", default="config/strategy_arena.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    frame = normalize_columns(pd.read_csv(args.csv))
    frame = prepare_indicators(frame)

    split = int(len(frame) * float(config["train_fraction"]))
    train = frame.iloc[:split].reset_index(drop=True)
    test = frame.iloc[split:].reset_index(drop=True)

    results: list[ArenaResult] = []
    counter = 1

    combinations = itertools.product(
        config["filter_sets"],
        config["ut_sensitivities"],
        config["atr_periods"],
        config["max_bars_held"],
        config["risk_profiles"],
    )

    for filter_set, sensitivity, atr_period, max_bars, risk in combinations:
        train_buy, train_sell = ut_signals(train, float(sensitivity), int(atr_period))
        test_buy, test_sell = ut_signals(test, float(sensitivity), int(atr_period))

        train_metrics = simulate(
            train,
            train_buy,
            train_sell,
            filter_mask(train, filter_set),
            config,
            int(max_bars),
            float(risk["stop_loss_percent"]),
            float(risk["take_profit_percent"]),
        )
        test_metrics = simulate(
            test,
            test_buy,
            test_sell,
            filter_mask(test, filter_set),
            config,
            int(max_bars),
            float(risk["stop_loss_percent"]),
            float(risk["take_profit_percent"]),
        )

        score, robustness, status = candidate_score(
            test_metrics,
            float(train_metrics["return_percent"]),
            int(config["minimum_trades"]),
        )

        results.append(
            ArenaResult(
                candidate_id=f"ARENA-{counter:04d}",
                filter_set=filter_set,
                sensitivity=float(sensitivity),
                atr_period=int(atr_period),
                max_bars_held=int(max_bars),
                risk_profile=str(risk["name"]),
                stop_loss_percent=float(risk["stop_loss_percent"]),
                take_profit_percent=float(risk["take_profit_percent"]),
                train_return_percent=round(float(train_metrics["return_percent"]), 4),
                test_return_percent=round(float(test_metrics["return_percent"]), 4),
                test_drawdown_percent=round(float(test_metrics["drawdown_percent"]), 4),
                test_profit_factor=round(float(test_metrics["profit_factor"]), 4),
                test_win_rate=round(float(test_metrics["win_rate"]), 4),
                test_trade_count=int(test_metrics["trade_count"]),
                robustness_ratio=robustness,
                score=score,
                status=status,
            )
        )
        counter += 1

    results.sort(key=lambda item: (item.status == "FINALIST", item.score), reverse=True)

    leaderboard = [{"rank": index, **asdict(item)} for index, item in enumerate(results, start=1)]
    output = {
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "bars": len(frame),
        "train_bars": len(train),
        "test_bars": len(test),
        "candidate_count": len(leaderboard),
        "finalist_count": sum(item["status"] == "FINALIST" for item in leaderboard),
        "top_finalists": [item for item in leaderboard if item["status"] == "FINALIST"][:10],
        "leaderboard": leaderboard,
        "tradingview_validation_required": True,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    report_path = Path("reports/strategy_research_arena.json")
    csv_path = Path("reports/strategy_research_arena.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(leaderboard).to_csv(csv_path, index=False)

    print("UT Strategy Research Arena")
    print(json.dumps({
        "candidate_count": output["candidate_count"],
        "finalist_count": output["finalist_count"],
        "top_finalists": output["top_finalists"],
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }, indent=2))
    print(f"JSON report: {report_path}")
    print(f"CSV report: {csv_path}")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
