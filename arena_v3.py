"""UT Bot Championship Arena v3.

Features:
- large configurable parameter grid,
- cross-market scoring,
- walk-forward validation,
- multi-process execution,
- Monte Carlo trade-order robustness checks,
- finalist CSV/JSON reports,
- heatmap-ready pivot tables,
- TradingView finalist configuration export.

Research only:
- no trading client is created,
- no orders are submitted,
- the production strategy is not changed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    filter_set: str
    sensitivity: float
    atr_period: int
    max_bars_held: int
    stop_loss_percent: float
    take_profit_percent: float


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}

    for column in frame.columns:
        key = column.strip().lower()

        if key in {"time", "date", "datetime", "timestamp"}:
            rename[column] = "date"
        elif key in {"open", "high", "low", "close", "volume"}:
            rename[column] = key

    frame = frame.rename(columns=rename)

    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing)}"
        )

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=["open", "high", "low", "close"]
    ).reset_index(drop=True)

    return frame


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(
        alpha=1 / length,
        adjust=False,
    ).mean()


def prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
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

    df["true_range"] = true_range
    df["ema200"] = close.ewm(
        span=200,
        adjust=False,
    ).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = (
        rma(gain, 14)
        / rma(loss, 14).replace(0, np.nan)
    )

    df["rsi14"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr14 = rma(
        true_range,
        14,
    ).replace(0, np.nan)

    plus_di = 100 * rma(
        plus_dm,
        14,
    ) / atr14

    minus_di = 100 * rma(
        minus_dm,
        14,
    ) / atr14

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["adx"] = rma(dx, 14)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["volume_ma20"] = volume.rolling(20).mean()

    df["relative_volume"] = (
        volume
        / df["volume_ma20"].replace(0, np.nan)
    )

    df["linreg50"] = close.rolling(50).apply(
        lambda values: np.polyval(
            np.polyfit(
                np.arange(len(values)),
                values,
                1,
            ),
            len(values) - 1,
        ),
        raw=True,
    )

    df["linreg50_prev"] = df["linreg50"].shift(1)

    smooth_open = df["open"].ewm(
        span=10,
        adjust=False,
    ).mean()

    smooth_high = high.ewm(
        span=10,
        adjust=False,
    ).mean()

    smooth_low = low.ewm(
        span=10,
        adjust=False,
    ).mean()

    smooth_close = close.ewm(
        span=10,
        adjust=False,
    ).mean()

    df["sha_close"] = (
        smooth_open
        + smooth_high
        + smooth_low
        + smooth_close
    ) / 4

    sha_open = pd.Series(
        index=df.index,
        dtype=float,
    )

    for index in df.index:
        if index == 0:
            sha_open.iloc[index] = (
                smooth_open.iloc[index]
                + smooth_close.iloc[index]
            ) / 2
        else:
            sha_open.iloc[index] = (
                sha_open.iloc[index - 1]
                + df["sha_close"].iloc[index - 1]
            ) / 2

    df["sha_open"] = sha_open

    fast_rs = (
        rma(gain, 5)
        / rma(loss, 5).replace(0, np.nan)
    )

    fast_rsi = 100 - 100 / (1 + fast_rs)

    ema_momentum = (
        100
        * (
            close.ewm(
                span=5,
                adjust=False,
            ).mean()
            - close.ewm(
                span=20,
                adjust=False,
            ).mean()
        )
        / close.replace(0, np.nan)
    )

    df["bx_style"] = (
        (fast_rsi - 50).ewm(
            span=5,
            adjust=False,
        ).mean()
        + ema_momentum
    )

    return df


def build_mask(
    df: pd.DataFrame,
    filter_set: str,
) -> pd.Series:
    masks = {
        "Original": pd.Series(
            True,
            index=df.index,
        ),
        "EMA200": df["close"] > df["ema200"],
        "RSI": (
            (df["rsi14"] > 50)
            & (df["rsi14"] < 75)
        ),
        "MACD": (
            df["macd"]
            > df["macd_signal"]
        ),
        "ADX": (
            (df["adx"] > 20)
            & (df["plus_di"] > df["minus_di"])
        ),
        "RelativeVolume": (
            df["relative_volume"] > 1
        ),
        "LinearRegression": (
            (df["close"] > df["linreg50"])
            & (
                df["linreg50"]
                > df["linreg50_prev"]
            )
        ),
        "SmoothedHeikenAshi": (
            (df["sha_close"] > df["sha_open"])
            & (
                df["sha_close"]
                > df["sha_close"].shift(1)
            )
        ),
        "BXStyle": (
            (df["bx_style"] > 0)
            & (
                df["bx_style"]
                > df["bx_style"].shift(1)
            )
        ),
    }

    output = pd.Series(
        True,
        index=df.index,
    )

    for component in filter_set.split("+"):
        if component not in masks:
            raise KeyError(
                f"Unknown filter component: {component}"
            )

        output &= masks[component]

    return output.fillna(False)


def ut_signals(
    df: pd.DataFrame,
    sensitivity: float,
    atr_period: int,
) -> tuple[pd.Series, pd.Series]:
    atr = rma(
        df["true_range"],
        atr_period,
    )

    stop = np.full(
        len(df),
        np.nan,
    )

    close = df["close"]

    for index in range(len(df)):
        source = float(close.iloc[index])

        distance = (
            float(atr.iloc[index]) * sensitivity
            if np.isfinite(atr.iloc[index])
            else 0.0
        )

        previous_stop = (
            source
            if index == 0
            or not np.isfinite(stop[index - 1])
            else stop[index - 1]
        )

        previous_source = (
            source
            if index == 0
            else float(close.iloc[index - 1])
        )

        if (
            source > previous_stop
            and previous_source > previous_stop
        ):
            stop[index] = max(
                previous_stop,
                source - distance,
            )
        elif (
            source < previous_stop
            and previous_source < previous_stop
        ):
            stop[index] = min(
                previous_stop,
                source + distance,
            )
        elif source > previous_stop:
            stop[index] = source - distance
        else:
            stop[index] = source + distance

    stop_series = pd.Series(
        stop,
        index=df.index,
    )

    buy = (
        (close > stop_series)
        & (
            close.shift(1)
            <= stop_series.shift(1)
        )
    )

    sell = (
        (close < stop_series)
        & (
            close.shift(1)
            >= stop_series.shift(1)
        )
    )

    return (
        buy.fillna(False),
        sell.fillna(False),
    )


def simulate(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    mask: pd.Series,
    config: dict[str, Any],
    candidate: Candidate,
) -> dict[str, Any]:
    initial_capital = float(
        config["starting_capital"]
    )

    cash = initial_capital
    quantity = 0.0
    entry_price = 0.0
    entry_index = -1

    trades: list[float] = []
    equity_curve: list[float] = []

    position_fraction = (
        float(config["position_percent"])
        / 100
    )

    commission_rate = (
        float(config["commission_percent"])
        / 100
    )

    slippage = float(
        config["slippage_dollars"]
    )

    for index, row in df.iterrows():
        close = float(row["close"])

        if (
            quantity == 0
            and bool(entries.iloc[index])
            and bool(mask.iloc[index])
        ):
            fill = close + slippage
            notional = cash * position_fraction

            quantity = (
                notional
                / fill
            )

            cash -= (
                notional
                + notional * commission_rate
            )

            entry_price = fill
            entry_index = index

        elif quantity > 0:
            stop_price = (
                entry_price
                * (
                    1
                    - candidate.stop_loss_percent
                    / 100
                )
            )

            target_price = (
                entry_price
                * (
                    1
                    + candidate.take_profit_percent
                    / 100
                )
            )

            exit_price: float | None = None

            if float(row["low"]) <= stop_price:
                exit_price = (
                    stop_price
                    - slippage
                )
            elif float(row["high"]) >= target_price:
                exit_price = (
                    target_price
                    - slippage
                )
            elif (
                bool(exits.iloc[index])
                or index - entry_index
                >= candidate.max_bars_held
            ):
                exit_price = (
                    close
                    - slippage
                )

            if exit_price is not None:
                proceeds = (
                    quantity
                    * exit_price
                )

                fee = (
                    proceeds
                    * commission_rate
                )

                pnl = (
                    proceeds
                    - fee
                    - quantity * entry_price
                )

                cash += (
                    proceeds
                    - fee
                )

                trades.append(
                    float(pnl)
                )

                quantity = 0.0
                entry_price = 0.0
                entry_index = -1

        equity_curve.append(
            cash
            + quantity * close
        )

    if quantity > 0:
        exit_price = (
            float(df.iloc[-1]["close"])
            - slippage
        )

        proceeds = (
            quantity
            * exit_price
        )

        fee = (
            proceeds
            * commission_rate
        )

        trades.append(
            float(
                proceeds
                - fee
                - quantity * entry_price
            )
        )

        cash += (
            proceeds
            - fee
        )

        if equity_curve:
            equity_curve[-1] = cash

    equity = np.asarray(
        equity_curve,
        dtype=float,
    )

    peaks = (
        np.maximum.accumulate(equity)
        if len(equity)
        else np.asarray([initial_capital])
    )

    drawdowns = (
        peaks - equity
        if len(equity)
        else np.asarray([0.0])
    )

    max_drawdown_amount = float(
        drawdowns.max()
    )

    max_drawdown_index = int(
        drawdowns.argmax()
    )

    max_drawdown_percent = (
        100
        * max_drawdown_amount
        / peaks[max_drawdown_index]
        if peaks[max_drawdown_index]
        else 0.0
    )

    wins = [
        trade
        for trade in trades
        if trade > 0
    ]

    losses = [
        trade
        for trade in trades
        if trade < 0
    ]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            99.0
            if gross_profit > 0
            else 0.0
        )
    )

    return {
        "return_percent": (
            100
            * (
                cash
                - initial_capital
            )
            / initial_capital
        ),
        "drawdown_percent":
            max_drawdown_percent,
        "profit_factor":
            profit_factor,
        "win_rate": (
            100
            * len(wins)
            / len(trades)
            if trades
            else 0.0
        ),
        "trade_count":
            len(trades),
        "trades":
            trades,
    }


def walk_forward_slices(
    length: int,
    folds: int,
) -> list[tuple[int, int]]:
    warmup = max(
        250,
        length // 3,
    )

    remaining = max(
        length - warmup,
        1,
    )

    fold_size = max(
        remaining // folds,
        1,
    )

    slices: list[tuple[int, int]] = []
    start = warmup

    for fold_index in range(folds):
        end = (
            length
            if fold_index == folds - 1
            else min(
                length,
                start + fold_size,
            )
        )

        if end > start:
            slices.append(
                (start, end)
            )

        start = end

    return slices


def evaluate_candidate(
    candidate: Candidate,
    datasets: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    all_trades: list[float] = []

    for symbol, full_data in datasets.items():
        entries, exits = ut_signals(
            full_data,
            candidate.sensitivity,
            candidate.atr_period,
        )

        mask = build_mask(
            full_data,
            candidate.filter_set,
        )

        slices = walk_forward_slices(
            len(full_data),
            int(config["walk_forward_folds"]),
        )

        for fold_number, (start, end) in enumerate(
            slices,
            start=1,
        ):
            fold_data = full_data.iloc[
                start:end
            ].reset_index(drop=True)

            fold_entries = entries.iloc[
                start:end
            ].reset_index(drop=True)

            fold_exits = exits.iloc[
                start:end
            ].reset_index(drop=True)

            fold_mask = mask.iloc[
                start:end
            ].reset_index(drop=True)

            result = simulate(
                fold_data,
                fold_entries,
                fold_exits,
                fold_mask,
                config,
                candidate,
            )

            all_trades.extend(
                result["trades"]
            )

            passed = (
                result["trade_count"]
                >= int(
                    config[
                        "minimum_trades_per_fold"
                    ]
                )
                and result["return_percent"] > 0
                and result["profit_factor"] > 1.05
            )

            fold_rows.append(
                {
                    "symbol":
                        symbol,
                    "fold":
                        fold_number,
                    "return_percent":
                        round(
                            result[
                                "return_percent"
                            ],
                            4,
                        ),
                    "drawdown_percent":
                        round(
                            result[
                                "drawdown_percent"
                            ],
                            4,
                        ),
                    "profit_factor":
                        round(
                            min(
                                result[
                                    "profit_factor"
                                ],
                                10.0,
                            ),
                            4,
                        ),
                    "win_rate":
                        round(
                            result[
                                "win_rate"
                            ],
                            4,
                        ),
                    "trade_count":
                        result[
                            "trade_count"
                        ],
                    "passed":
                        passed,
                }
            )

    returns = [
        row["return_percent"]
        for row in fold_rows
    ]

    drawdowns = [
        row["drawdown_percent"]
        for row in fold_rows
    ]

    profit_factors = [
        row["profit_factor"]
        for row in fold_rows
    ]

    trade_counts = [
        row["trade_count"]
        for row in fold_rows
    ]

    symbols_passed = len(
        {
            row["symbol"]
            for row in fold_rows
            if row["passed"]
        }
    )

    consistency = (
        100
        * sum(
            row["passed"]
            for row in fold_rows
        )
        / len(fold_rows)
        if fold_rows
        else 0.0
    )

    median_return = (
        median(returns)
        if returns
        else 0.0
    )

    mean_return = (
        mean(returns)
        if returns
        else 0.0
    )

    worst_return = (
        min(returns)
        if returns
        else 0.0
    )

    median_drawdown = (
        median(drawdowns)
        if drawdowns
        else 0.0
    )

    worst_drawdown = (
        max(drawdowns)
        if drawdowns
        else 0.0
    )

    median_profit_factor = (
        median(profit_factors)
        if profit_factors
        else 0.0
    )

    median_trade_count = (
        median(trade_counts)
        if trade_counts
        else 0.0
    )

    score = (
        median_return * 3.0
        + mean_return * 1.5
        + median_profit_factor * 8.0
        + consistency * 0.20
        + max(
            worst_return,
            -10.0,
        ) * 0.75
        - median_drawdown * 1.5
        - worst_drawdown * 0.5
    )

    status = (
        "WALK_FORWARD_FINALIST"
        if (
            symbols_passed
            >= int(
                config[
                    "minimum_symbols_passed"
                ]
            )
            and median_return > 0
            and median_profit_factor > 1.05
        )
        else "REJECT"
    )

    return {
        **asdict(candidate),
        "symbols_tested":
            len(datasets),
        "symbols_passed":
            symbols_passed,
        "fold_count":
            len(fold_rows),
        "median_test_return_percent":
            round(
                median_return,
                4,
            ),
        "mean_test_return_percent":
            round(
                mean_return,
                4,
            ),
        "worst_test_return_percent":
            round(
                worst_return,
                4,
            ),
        "median_drawdown_percent":
            round(
                median_drawdown,
                4,
            ),
        "worst_drawdown_percent":
            round(
                worst_drawdown,
                4,
            ),
        "median_profit_factor":
            round(
                median_profit_factor,
                4,
            ),
        "median_trade_count":
            round(
                median_trade_count,
                2,
            ),
        "consistency_percent":
            round(
                consistency,
                2,
            ),
        "score":
            round(
                score,
                4,
            ),
        "status":
            status,
        "trade_returns":
            all_trades,
        "fold_results":
            fold_rows,
    }


def monte_carlo(
    trades: list[float],
    initial_capital: float,
    runs: int,
    seed: int,
) -> dict[str, float]:
    if not trades:
        return {
            "mc_runs":
                runs,
            "median_final_equity":
                initial_capital,
            "fifth_percentile_final_equity":
                initial_capital,
            "median_max_drawdown_percent":
                0.0,
            "ninety_fifth_percentile_drawdown_percent":
                0.0,
            "probability_of_loss_percent":
                100.0,
        }

    rng = np.random.default_rng(seed)

    final_equities: list[float] = []
    max_drawdowns: list[float] = []

    trades_array = np.asarray(
        trades,
        dtype=float,
    )

    for _ in range(runs):
        shuffled = rng.permutation(
            trades_array
        )

        equity = (
            initial_capital
            + np.cumsum(shuffled)
        )

        full_equity = np.concatenate(
            [
                [initial_capital],
                equity,
            ]
        )

        peaks = np.maximum.accumulate(
            full_equity
        )

        drawdowns = (
            100
            * (
                peaks
                - full_equity
            )
            / np.where(
                peaks == 0,
                1,
                peaks,
            )
        )

        final_equities.append(
            float(full_equity[-1])
        )

        max_drawdowns.append(
            float(drawdowns.max())
        )

    final_array = np.asarray(
        final_equities,
        dtype=float,
    )

    drawdown_array = np.asarray(
        max_drawdowns,
        dtype=float,
    )

    return {
        "mc_runs":
            runs,
        "median_final_equity":
            round(
                float(
                    np.median(final_array)
                ),
                2,
            ),
        "fifth_percentile_final_equity":
            round(
                float(
                    np.percentile(
                        final_array,
                        5,
                    )
                ),
                2,
            ),
        "median_max_drawdown_percent":
            round(
                float(
                    np.median(
                        drawdown_array
                    )
                ),
                4,
            ),
        "ninety_fifth_percentile_drawdown_percent":
            round(
                float(
                    np.percentile(
                        drawdown_array,
                        95,
                    )
                ),
                4,
            ),
        "probability_of_loss_percent":
            round(
                100
                * float(
                    np.mean(
                        final_array
                        < initial_capital
                    )
                ),
                4,
            ),
    }


def build_candidates(
    config: dict[str, Any],
) -> list[Candidate]:
    grid = config["parameter_grid"]

    combinations = itertools.product(
        grid["filter_sets"],
        grid["sensitivities"],
        grid["atr_periods"],
        grid["max_bars_held"],
        grid["stop_loss_percent"],
        grid["take_profit_percent"],
    )

    candidates: list[Candidate] = []

    for index, values in enumerate(
        combinations,
        start=1,
    ):
        (
            filter_set,
            sensitivity,
            atr_period,
            max_bars_held,
            stop_loss,
            take_profit,
        ) = values

        candidates.append(
            Candidate(
                candidate_id=f"V3-{index:06d}",
                filter_set=str(filter_set),
                sensitivity=float(sensitivity),
                atr_period=int(atr_period),
                max_bars_held=int(max_bars_held),
                stop_loss_percent=float(stop_loss),
                take_profit_percent=float(take_profit),
            )
        )

    return candidates


def load_datasets(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        path = data_dir / f"{symbol}_1D.csv"

        if not path.exists():
            continue

        datasets[symbol] = prepare_indicators(
            normalize_columns(
                pd.read_csv(path)
            )
        )

    return datasets


def export_heatmaps(
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if frame.empty:
        return

    for metric in [
        "score",
        "median_test_return_percent",
        "median_profit_factor",
    ]:
        pivot = frame.pivot_table(
            index="sensitivity",
            columns="atr_period",
            values=metric,
            aggfunc="mean",
        )

        pivot.to_csv(
            output_dir
            / f"heatmap_{metric}.csv"
        )


def write_reports(
    results: list[dict[str, Any]],
    finalists: list[dict[str, Any]],
    datasets: dict[str, pd.DataFrame],
    config: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    reports = Path("reports")

    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_limit = int(
        config[
            "top_candidates_for_reports"
        ]
    )

    top = finalists[:report_limit]

    output = {
        "symbols_loaded":
            sorted(datasets),
        "candidate_count":
            len(results),
        "finalist_count":
            len(finalists),
        "worker_count":
            workers,
        "walk_forward_folds":
            int(
                config[
                    "walk_forward_folds"
                ]
            ),
        "monte_carlo_runs":
            int(
                config[
                    "monte_carlo_runs"
                ]
            ),
        "top_finalists":
            top,
        "production_strategy_changed":
            False,
        "market_request_made":
            False,
        "order_submitted":
            False,
    }

    (
        reports
        / "arena_v3_results.json"
    ).write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_columns = [
        "rank",
        "candidate_id",
        "filter_set",
        "sensitivity",
        "atr_period",
        "max_bars_held",
        "stop_loss_percent",
        "take_profit_percent",
        "symbols_passed",
        "median_test_return_percent",
        "mean_test_return_percent",
        "worst_test_return_percent",
        "median_drawdown_percent",
        "worst_drawdown_percent",
        "median_profit_factor",
        "consistency_percent",
        "score",
        "status",
    ]

    results_frame = pd.DataFrame(results)

    if results_frame.empty:
        results_frame = pd.DataFrame(
            columns=summary_columns
        )
    else:
        results_frame = results_frame.reindex(
            columns=summary_columns
        )

    results_frame.to_csv(
        reports
        / "arena_v3_leaderboard.csv",
        index=False,
    )

    top_frame = pd.DataFrame(top)

    if top_frame.empty:
        top_frame = pd.DataFrame(
            columns=summary_columns
        )
    else:
        top_frame = top_frame.reindex(
            columns=summary_columns
        )

    top_frame.to_csv(
        reports
        / "arena_v3_top_100.csv",
        index=False,
    )

    export_heatmaps(
        results_frame,
        reports / "heatmaps",
    )

    tradingview_columns = [
        "rank",
        "candidate_id",
        "competitor",
        "ut_sensitivity",
        "ut_atr_period",
        "maximum_bars_held",
        "hard_stop_percent",
        "take_profit_percent",
        "score",
    ]

    tradingview_rows: list[dict[str, Any]] = []

    for row in top:
        tradingview_rows.append(
            {
                "rank":
                    row["rank"],
                "candidate_id":
                    row["candidate_id"],
                "competitor":
                    row["filter_set"],
                "ut_sensitivity":
                    row["sensitivity"],
                "ut_atr_period":
                    row["atr_period"],
                "maximum_bars_held":
                    row["max_bars_held"],
                "hard_stop_percent":
                    row["stop_loss_percent"],
                "take_profit_percent":
                    row["take_profit_percent"],
                "score":
                    row["score"],
            }
        )

    tradingview_frame = pd.DataFrame(
        tradingview_rows
    )

    if tradingview_frame.empty:
        tradingview_frame = pd.DataFrame(
            columns=tradingview_columns
        )
    else:
        tradingview_frame = (
            tradingview_frame.reindex(
                columns=tradingview_columns
            )
        )

    tradingview_frame.to_csv(
        reports
        / "arena_v3_tradingview_finalists.csv",
        index=False,
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data",
    )

    parser.add_argument(
        "--config",
        default="config/arena_v3.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    arguments = parser.parse_args()

    config = load_config(
        Path(arguments.config)
    )

    datasets = load_datasets(
        Path(arguments.data_dir),
        config["symbols"],
    )

    if not datasets:
        raise SystemExit(
            "No symbol CSV files found."
        )

    candidates = build_candidates(config)

    if arguments.limit > 0:
        candidates = candidates[
            : arguments.limit
        ]

    configured_workers = int(
        config.get(
            "max_workers",
            0,
        )
    )

    workers = (
        configured_workers
        if configured_workers > 0
        else max(
            1,
            (os.cpu_count() or 2) - 1,
        )
    )

    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                evaluate_candidate,
                candidate,
                datasets,
                config,
            ): candidate.candidate_id
            for candidate in candidates
        }

        completed = 0

        for future in as_completed(futures):
            results.append(
                future.result()
            )

            completed += 1

            if (
                completed % 100 == 0
                or completed == len(candidates)
            ):
                print(
                    f"Completed "
                    f"{completed}/"
                    f"{len(candidates)} "
                    f"candidates..."
                )

    results.sort(
        key=lambda row: (
            row["status"]
            == "WALK_FORWARD_FINALIST",
            row["score"],
        ),
        reverse=True,
    )

    monte_carlo_count = min(
        int(
            config[
                "top_candidates_for_monte_carlo"
            ]
        ),
        len(results),
    )

    for index, row in enumerate(
        results[:monte_carlo_count]
    ):
        row["monte_carlo"] = monte_carlo(
            row.pop(
                "trade_returns",
                [],
            ),
            float(
                config[
                    "starting_capital"
                ]
            ),
            int(
                config[
                    "monte_carlo_runs"
                ]
            ),
            seed=1000 + index,
        )

    for row in results[
        monte_carlo_count:
    ]:
        row.pop(
            "trade_returns",
            None,
        )

        row["monte_carlo"] = None

    for rank, row in enumerate(
        results,
        start=1,
    ):
        row["rank"] = rank

    finalists = [
        row
        for row in results
        if row["status"]
        == "WALK_FORWARD_FINALIST"
    ]

    output = write_reports(
        results=results,
        finalists=finalists,
        datasets=datasets,
        config=config,
        workers=workers,
    )

    print(
        "UT Bot Championship Arena v3"
    )

    print(
        json.dumps(
            {
                "symbols_loaded":
                    output[
                        "symbols_loaded"
                    ],
                "candidate_count":
                    output[
                        "candidate_count"
                    ],
                "finalist_count":
                    output[
                        "finalist_count"
                    ],
                "worker_count":
                    output[
                        "worker_count"
                    ],
                "top_finalists":
                    output[
                        "top_finalists"
                    ][:10],
                "production_strategy_changed":
                    False,
                "market_request_made":
                    False,
                "order_submitted":
                    False,
            },
            indent=2,
        )
    )

    print(
        "No market request was made."
    )

    print(
        "No order was submitted."
    )


if __name__ == "__main__":
    main()
