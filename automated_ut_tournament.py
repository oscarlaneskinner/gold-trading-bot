"""Batch-run UT research variants from one TradingView OHLCV CSV export.

This is a research approximation of the Pine strategies. TradingView's broker
emulator can differ in order-fill details, so finalists should still be
validated in TradingView before paper-trading promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np
import pandas as pd


REPORT_JSON = Path("reports/automated_ut_tournament.json")
REPORT_CSV = Path("reports/automated_ut_tournament.csv")


@dataclass
class Result:
    strategy_name: str
    net_profit_amount: float
    net_profit_percent: float
    max_drawdown_amount: float
    max_drawdown_percent: float
    profitable_trades_percent: float
    profit_factor: float
    closed_trades: int
    score: float


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "time": "date",
        "datetime": "date",
        "timestamp": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    renamed = {}
    for column in frame.columns:
        key = column.strip().lower()
        if key in aliases:
            renamed[column] = aliases[key]
    frame = frame.rename(columns=renamed)

    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "volume" not in frame:
        frame["volume"] = 0.0

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return frame


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr10"] = rma(true_range, 10)
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema100"] = close.ewm(span=100, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    atr14 = rma(true_range, 14).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, 14) / atr14
    minus_di = 100 * rma(minus_dm, 14) / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = rma(dx, 14)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["volume_ma20"] = volume.rolling(20).mean()
    df["relative_volume"] = volume / df["volume_ma20"].replace(0, np.nan)

    df["linreg50"] = close.rolling(50).apply(
        lambda values: np.polyval(np.polyfit(np.arange(len(values)), values, 1), len(values) - 1),
        raw=True,
    )
    df["linreg50_prev"] = df["linreg50"].shift(1)

    smooth_open = df["open"].ewm(span=10, adjust=False).mean()
    smooth_high = high.ewm(span=10, adjust=False).mean()
    smooth_low = low.ewm(span=10, adjust=False).mean()
    smooth_close = close.ewm(span=10, adjust=False).mean()
    sha_close = (smooth_open + smooth_high + smooth_low + smooth_close) / 4
    sha_open = pd.Series(index=df.index, dtype=float)
    for index in df.index:
        if index == 0:
            sha_open.iloc[index] = (smooth_open.iloc[index] + smooth_close.iloc[index]) / 2
        else:
            sha_open.iloc[index] = (sha_open.iloc[index - 1] + sha_close.iloc[index - 1]) / 2
    df["sha_close"] = sha_close
    df["sha_open"] = sha_open

    daily_range = high - low
    df["adr20"] = daily_range.rolling(20).mean()
    df["adr_percent"] = 100 * df["adr20"] / close.replace(0, np.nan)

    ut_stop = np.full(len(df), np.nan)
    sensitivity = 1.0
    for i in range(len(df)):
        source = close.iloc[i]
        distance = sensitivity * df["atr10"].iloc[i]
        if not np.isfinite(distance):
            ut_stop[i] = source
            continue
        previous_stop = source if i == 0 or not np.isfinite(ut_stop[i - 1]) else ut_stop[i - 1]
        previous_source = source if i == 0 else close.iloc[i - 1]
        if source > previous_stop and previous_source > previous_stop:
            ut_stop[i] = max(previous_stop, source - distance)
        elif source < previous_stop and previous_source < previous_stop:
            ut_stop[i] = min(previous_stop, source + distance)
        elif source > previous_stop:
            ut_stop[i] = source - distance
        else:
            ut_stop[i] = source + distance

    df["ut_stop"] = ut_stop
    df["ut_buy"] = (close > df["ut_stop"]) & (close.shift(1) <= df["ut_stop"].shift(1))
    df["ut_sell"] = (close < df["ut_stop"]) & (close.shift(1) >= df["ut_stop"].shift(1))
    return df


def filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    ema50 = df["close"] > df["ema50"]
    ema100 = df["close"] > df["ema100"]
    ema200 = df["close"] > df["ema200"]
    rsi = (df["rsi14"] > 50) & (df["rsi14"] < 75)
    macd = df["macd"] > df["macd_signal"]
    adx = (df["adx"] > 20) & (df["plus_di"] > df["minus_di"])
    relvol = df["relative_volume"] > 1
    spike = df["volume"] > 1.5 * df["volume_ma20"]
    linreg = (df["close"] > df["linreg50"]) & (df["linreg50"] > df["linreg50_prev"])
    sha = (df["sha_close"] > df["sha_open"]) & (df["sha_close"] > df["sha_close"].shift(1))
    adr = df["adr_percent"] >= 0.8

    fast_momentum = (df["rsi14"] - 50).ewm(span=5, adjust=False).mean()
    ema_momentum = 100 * (
        df["close"].ewm(span=5, adjust=False).mean()
        - df["close"].ewm(span=20, adjust=False).mean()
    ) / df["close"].replace(0, np.nan)
    bx = (fast_momentum + ema_momentum > 0) & (
        fast_momentum + ema_momentum > (fast_momentum + ema_momentum).shift(1)
    )

    true = pd.Series(True, index=df.index)
    return {
        "UT Original": true,
        "UT + EMA50": ema50,
        "UT + EMA100": ema100,
        "UT + EMA200": ema200,
        "UT + RSI": rsi,
        "UT + MACD": macd,
        "UT + ADX": adx,
        "UT + Relative Volume": relvol,
        "UT + Volume Spike": spike,
        "UT + EMA200 + RSI": ema200 & rsi,
        "UT + EMA200 + MACD": ema200 & macd,
        "UT + EMA200 + ADX": ema200 & adx,
        "UT + EMA200 + Relative Volume": ema200 & relvol,
        "UT + EMA200 + Volume Spike": ema200 & spike,
        "UT + RSI + MACD": rsi & macd,
        "UT + RSI + Relative Volume": rsi & relvol,
        "UT + B-Xtrender Style": bx,
        "UT + EMA200 + B-Xtrender Style": ema200 & bx,
        "UT + Linear Regression Trend": linreg,
        "UT + EMA200 + Linear Regression": ema200 & linreg,
        "UT + Smoothed Heiken Ashi": sha,
        "UT + EMA200 + Smoothed Heiken Ashi": ema200 & sha,
        "UT + ADR Volatility": adr,
        "UT + B-Xtrender Style + Linear Regression": bx & linreg,
        "UT + B-Xtrender Style + Smoothed Heiken Ashi": bx & sha,
        "UT + All Alternative Filters": ema200 & bx & linreg & sha & adr,
    }


def score_result(return_percent: float, drawdown_percent: float, win_rate: float, profit_factor: float, trades: int) -> float:
    sample_multiplier = min(trades / 30.0, 1.0)
    risk_adjusted = return_percent / drawdown_percent if drawdown_percent > 0 else 0.0
    raw = (
        return_percent * 1.5
        + min(profit_factor, 5.0) * 8.0
        + win_rate * 0.15
        + min(risk_adjusted, 10.0) * 2.0
        - drawdown_percent * 1.5
    )
    return round(raw * sample_multiplier, 4)


def run_variant(df: pd.DataFrame, name: str, mask: pd.Series) -> Result:
    initial_capital = 100000.0
    position_percent = 15.0
    commission_rate = 0.0001
    slippage = 0.01
    hard_stop = 0.03
    target = 0.08
    max_bars = 20

    cash = initial_capital
    quantity = 0.0
    entry_price = 0.0
    entry_index = -1
    trades: list[float] = []
    equity_curve: list[float] = []

    for i, row in df.iterrows():
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])

        if quantity == 0 and bool(row["ut_buy"]) and bool(mask.iloc[i]):
            fill = close + slippage
            notional = cash * position_percent / 100.0
            quantity = notional / fill
            cash -= notional + notional * commission_rate
            entry_price = fill
            entry_index = i

        elif quantity > 0:
            stop_price = entry_price * (1 - hard_stop)
            target_price = entry_price * (1 + target)
            exit_price = None

            if low <= stop_price:
                exit_price = stop_price - slippage
            elif high >= target_price:
                exit_price = target_price - slippage
            elif bool(row["ut_sell"]) or i - entry_index >= max_bars:
                exit_price = close - slippage

            if exit_price is not None:
                proceeds = quantity * exit_price
                commission = proceeds * commission_rate
                pnl = proceeds - commission - quantity * entry_price
                cash += proceeds - commission
                trades.append(pnl)
                quantity = 0.0
                entry_price = 0.0
                entry_index = -1

        equity = cash + quantity * close
        equity_curve.append(equity)

    if quantity > 0:
        exit_price = float(df.iloc[-1]["close"]) - slippage
        proceeds = quantity * exit_price
        commission = proceeds * commission_rate
        pnl = proceeds - commission - quantity * entry_price
        cash += proceeds - commission
        trades.append(pnl)
        equity_curve[-1] = cash

    net_profit = cash - initial_capital
    net_percent = 100 * net_profit / initial_capital

    peaks = np.maximum.accumulate(np.asarray(equity_curve, dtype=float))
    drawdowns = peaks - np.asarray(equity_curve, dtype=float)
    max_dd_amount = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    peak_at_dd = float(peaks[int(np.argmax(drawdowns))]) if len(drawdowns) else initial_capital
    max_dd_percent = 100 * max_dd_amount / peak_at_dd if peak_at_dd else 0.0

    wins = [value for value in trades if value > 0]
    losses = [value for value in trades if value < 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    score = score_result(net_percent, max_dd_percent, win_rate, profit_factor, len(trades))

    return Result(
        strategy_name=name,
        net_profit_amount=round(net_profit, 2),
        net_profit_percent=round(net_percent, 4),
        max_drawdown_amount=round(max_dd_amount, 2),
        max_drawdown_percent=round(max_dd_percent, 4),
        profitable_trades_percent=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        closed_trades=len(trades),
        score=score,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="TradingView OHLCV CSV export")
    args = parser.parse_args()

    source = Path(args.csv)
    if not source.exists():
        raise SystemExit(f"CSV file not found: {source}")

    frame = normalize_columns(pd.read_csv(source))
    prepared = indicators(frame)
    variants = filters(prepared)

    results = [run_variant(prepared, name, mask.fillna(False)) for name, mask in variants.items()]
    results.sort(key=lambda result: (result.score, result.net_profit_percent), reverse=True)

    payload = {
        "source_csv": str(source),
        "bars": len(prepared),
        "competitor_count": len(results),
        "leaderboard": [
            {"rank": rank, **asdict(result)}
            for rank, result in enumerate(results, start=1)
        ],
        "tradingview_validation_required_for_finalists": True,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["leaderboard"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["leaderboard"])

    print("Automated UT Tournament")
    print(json.dumps(payload, indent=2))
    print(f"JSON report: {REPORT_JSON}")
    print(f"CSV report: {REPORT_CSV}")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
