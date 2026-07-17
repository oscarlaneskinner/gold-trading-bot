"""Market Regime Lab v1.

Classifies the current benchmark environment as:
- BULL
- BEAR
- SIDEWAYS
- HIGH_VOLATILITY

Uses local daily OHLCV CSV data only.

Research only:
- no market-data request,
- no trading client,
- no order submission,
- no production strategy change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}

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
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.dropna(
        subset=["open", "high", "low", "close"]
    ).reset_index(drop=True)


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    df = frame.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    fast_period = int(config["ema_fast_period"])
    slow_period = int(config["ema_slow_period"])
    atr_period = int(config["atr_period"])

    df["ema_fast"] = close.ewm(
        span=fast_period,
        adjust=False,
    ).mean()

    df["ema_slow"] = close.ewm(
        span=slow_period,
        adjust=False,
    ).mean()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = rma(true_range, atr_period)

    df["atr_percent"] = (
        100
        * df["atr"]
        / close.replace(0, np.nan)
    )

    df["price_vs_slow_percent"] = (
        100
        * (close - df["ema_slow"])
        / df["ema_slow"].replace(0, np.nan)
    )

    df["fast_vs_slow_percent"] = (
        100
        * (df["ema_fast"] - df["ema_slow"])
        / df["ema_slow"].replace(0, np.nan)
    )

    df["slow_slope_20d_percent"] = (
        100
        * (
            df["ema_slow"]
            / df["ema_slow"].shift(20)
            - 1
        )
    )

    return df


def classify_regime(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    minimum_rows = int(config["minimum_rows"])

    if len(frame) < minimum_rows:
        raise ValueError(
            f"Not enough rows. Need at least {minimum_rows}, found {len(frame)}."
        )

    df = add_indicators(frame, config)
    latest = df.iloc[-1]

    close = float(latest["close"])
    ema_fast = float(latest["ema_fast"])
    ema_slow = float(latest["ema_slow"])
    atr_percent = float(latest["atr_percent"])
    price_vs_slow = float(latest["price_vs_slow_percent"])
    fast_vs_slow = float(latest["fast_vs_slow_percent"])
    slow_slope = float(latest["slow_slope_20d_percent"])

    high_volatility_threshold = float(
        config["atr_percent_high_threshold"]
    )

    trend_strength = float(
        config["trend_strength_percent"]
    )

    sideways_band = float(
        config["sideways_band_percent"]
    )

    reasons: list[str] = []

    if atr_percent >= high_volatility_threshold:
        regime = "HIGH_VOLATILITY"
        reasons.append(
            f"ATR percent {atr_percent:.2f}% is above "
            f"{high_volatility_threshold:.2f}%."
        )

    elif (
        close > ema_slow
        and ema_fast > ema_slow
        and price_vs_slow >= trend_strength
        and slow_slope > 0
    ):
        regime = "BULL"
        reasons.extend(
            [
                "Price is above the slow EMA.",
                "Fast EMA is above the slow EMA.",
                "Slow EMA slope is positive.",
            ]
        )

    elif (
        close < ema_slow
        and ema_fast < ema_slow
        and price_vs_slow <= -trend_strength
        and slow_slope < 0
    ):
        regime = "BEAR"
        reasons.extend(
            [
                "Price is below the slow EMA.",
                "Fast EMA is below the slow EMA.",
                "Slow EMA slope is negative.",
            ]
        )

    elif (
        abs(price_vs_slow) <= sideways_band
        or abs(fast_vs_slow) <= sideways_band
    ):
        regime = "SIDEWAYS"
        reasons.append(
            "Price and moving averages are clustered near the long-term trend."
        )

    else:
        regime = "SIDEWAYS"
        reasons.append(
            "Trend conditions are mixed and do not meet bull or bear thresholds."
        )

    permissions = {
        "allow_gld_bot": True,
        "allow_long_bot": regime in {"BULL", "SIDEWAYS"},
        "allow_short_bot": regime in {"BEAR", "HIGH_VOLATILITY"},
        "reduce_position_size": regime in {"SIDEWAYS", "HIGH_VOLATILITY"},
    }

    return {
        "regime": regime,
        "close": round(close, 4),
        "ema_fast": round(ema_fast, 4),
        "ema_slow": round(ema_slow, 4),
        "atr_percent": round(atr_percent, 4),
        "price_vs_slow_percent": round(price_vs_slow, 4),
        "fast_vs_slow_percent": round(fast_vs_slow, 4),
        "slow_slope_20d_percent": round(slow_slope, 4),
        "reasons": reasons,
        "permissions": permissions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/market_regime_lab_v1.json",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
    )

    parser.add_argument(
        "--symbol",
        default=None,
    )

    arguments = parser.parse_args()

    config = load_config(Path(arguments.config))

    data_directory = Path(
        arguments.data_dir
        or config["data_directory"]
    )

    symbol = str(
        arguments.symbol
        or config["benchmark_symbol"]
    ).upper()

    csv_path = data_directory / f"{symbol}_1D.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {csv_path}"
        )

    frame = normalize_columns(
        pd.read_csv(csv_path)
    )

    result = classify_regime(
        frame,
        config,
    )

    output = {
        "benchmark_symbol": symbol,
        **result,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    report_directory = Path(
        "reports/market_regime"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        report_directory
        / "market_regime_lab_v1.json"
    ).write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "MARKET REGIME LAB V1",
        "=" * 20,
        f"Benchmark: {symbol}",
        f"Regime: {output['regime']}",
        f"Close: {output['close']}",
        f"ATR %: {output['atr_percent']}",
        "",
        "Permissions:",
        f"GLD bot: {output['permissions']['allow_gld_bot']}",
        f"Long bot: {output['permissions']['allow_long_bot']}",
        f"Short bot: {output['permissions']['allow_short_bot']}",
        f"Reduce size: {output['permissions']['reduce_position_size']}",
        "",
        "Reasons:",
    ]

    summary_lines.extend(
        f"- {reason}"
        for reason in output["reasons"]
    )

    (
        report_directory
        / "market_regime_lab_v1_summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("Market Regime Lab v1")
    print(
        json.dumps(
            output,
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
