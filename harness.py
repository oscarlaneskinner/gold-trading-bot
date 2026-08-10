"""
Shared, mandatory testing harness for any trading strategy experiment.

RULE: every experiment run through this harness automatically gets compared
against buy-and-hold on the SAME data windows. There is no code path that
produces a result without that comparison - this is deliberate, because the
single biggest mistake found across this project's history was strategies
that looked good in isolation but had never actually been checked against
simply holding the asset.

This also logs every experiment run to experiment_log.json, so that anyone
looking at results later can see how MANY strategy variants have been tried
in total. The more variants tried, the more likely one looks good purely by
chance (the "multiple comparisons problem") - this log exists so that risk
stays visible rather than hidden.

USAGE:
    from harness import run_experiment, ExperimentConfig

    def train_and_predict(train_df, test_df, feature_cols, target_col):
        model = YourModelHere()
        model.fit(train_df[feature_cols], train_df[target_col])
        return model.predict_proba(test_df[feature_cols])[:, 1]

    result = run_experiment(
        experiment_name="my_new_idea",
        symbol="GLD",
        data=your_dataframe,  # columns: timestamp, open, high, low, close, volume
        feature_cols=[...],
        target_col="target",
        train_and_predict_fn=train_and_predict,
        config=ExperimentConfig(),
    )
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

EXPERIMENT_LOG_PATH = Path("reports/experiment_log.json")


@dataclass
class ExperimentConfig:
    """
    Defaults here match the ONLY configuration validated as of today across
    a full 20-year GLD backtest: 20-day model window, no confidence
    threshold, no trend/RSI/volatility filters, 10% stop-loss, 20%
    take-profit, 20-day max hold, realistic slippage. Change these
    deliberately, not by accident - each field has a note on what changing
    it did the last time it was tested, where known.
    """
    min_train_rows: int = 750          # ~3 years minimum before the first fold
    n_folds: int = 5                   # more folds = more robust, noisier per-fold
    entry_threshold: float = 0.50      # tested: raising this (e.g. 0.70) consistently HURT results
    stop_loss_pct: float = 0.10        # tested: 3-9% cut off good trades early; 10% was the sweet spot
    take_profit_pct: float = 0.20
    max_hold_days: int = 20
    slippage_pct: float = 0.0005       # per-side; realistic for a liquid ETF, may be too low for illiquid instruments
    position_percent: float = 0.10     # fraction of capital per trade
    initial_capital: float = 10_000.0

    # The bar a strategy must clear to be called "promising" - decided in
    # advance, not adjusted after seeing results. A strategy must beat
    # buy-and-hold by this many percentage points, on average across folds,
    # AND in the majority of individual folds, to pass.
    min_outperformance_vs_buy_hold_pct: float = 5.0
    min_fraction_of_folds_beating_buy_hold: float = 0.6


def build_walk_forward_folds(data: pd.DataFrame, target_col: str, config: ExperimentConfig):
    """Build N walk-forward folds. Each fold trains only on data strictly
    before its own test window - no fold's model ever sees its own test data."""
    n = len(data)
    if n <= config.min_train_rows:
        raise ValueError(
            f"Only {n} rows available, need more than min_train_rows={config.min_train_rows}."
        )
    size = (n - config.min_train_rows) // config.n_folds
    if size < 20:
        raise ValueError(
            f"Fold size would be only {size} rows - reduce n_folds or increase data."
        )

    folds = []
    for i in range(config.n_folds):
        start = config.min_train_rows + i * size
        end = n if i == config.n_folds - 1 else start + size
        train = data.iloc[:start]
        test = data.iloc[start:end]
        if len(test) < 10:
            continue
        if train[target_col].nunique() < 2 or test[target_col].nunique() < 2:
            continue
        folds.append((train, test))
    return folds


def calculate_buy_hold_return(test_df: pd.DataFrame) -> float:
    start_price = float(test_df.iloc[0]["close"])
    end_price = float(test_df.iloc[-1]["close"])
    return (end_price - start_price) / start_price


def simulate_strategy(test_df: pd.DataFrame, probability_up: np.ndarray, config: ExperimentConfig) -> dict:
    """
    Simple long-only simulation: enter when probability_up >= entry_threshold,
    exit on stop-loss, take-profit, or max_hold_days, whichever comes first.
    Applies slippage on both entry and exit.
    """
    cash = config.initial_capital
    shares = 0.0
    entry_price = entry_value = entry_index = None
    trades = []
    curve = []
    invested_days = 0

    for i, ((_, row), probability) in enumerate(zip(test_df.iterrows(), probability_up)):
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])

        if shares == 0:
            if probability >= config.entry_threshold:
                allocation = cash * config.position_percent
                execution_price = close * (1 + config.slippage_pct)
                shares = allocation / execution_price
                cash -= allocation
                entry_price = execution_price
                entry_value = allocation
                entry_index = i
        else:
            invested_days += 1
            held = i - int(entry_index)
            stop_price = float(entry_price) * (1 - config.stop_loss_pct)
            target_price = float(entry_price) * (1 + config.take_profit_pct)
            exit_price = None

            if low <= stop_price:
                exit_price = stop_price
            elif high >= target_price:
                exit_price = target_price
            elif held >= config.max_hold_days:
                exit_price = close

            if exit_price is not None:
                proceeds = shares * exit_price * (1 - config.slippage_pct)
                cash += proceeds
                trades.append((proceeds - float(entry_value)) / float(entry_value))
                shares = 0.0
                entry_price = entry_value = entry_index = None

        curve.append(cash + shares * close)

    if shares > 0:
        proceeds = shares * float(test_df.iloc[-1]["close"]) * (1 - config.slippage_pct)
        cash += proceeds
        trades.append((proceeds - float(entry_value)) / float(entry_value))
        curve[-1] = cash

    arr = np.asarray(curve, dtype=float)
    max_drawdown = float((arr / np.maximum.accumulate(arr) - 1).min()) if arr.size else 0.0

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss if gross_loss > 0
                      else (math.inf if gross_profit > 0 else 0.0))

    return {
        "total_return_pct": float(cash / config.initial_capital - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "trades": len(trades),
        "win_rate_pct": (float(np.mean(np.asarray(trades) > 0)) * 100) if trades else 0.0,
        "profit_factor": float(profit_factor),
        "exposure_pct": float(invested_days / max(1, len(test_df))) * 100,
    }


def run_experiment(
    experiment_name: str,
    symbol: str,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_and_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, list[str], str], np.ndarray],
    config: Optional[ExperimentConfig] = None,
    notes: str = "",
) -> dict:
    """
    Run a full walk-forward experiment. train_and_predict_fn is called once
    per fold with (train_df, test_df, feature_cols, target_col) and must
    return an array of predicted "probability of up" values, one per row of
    test_df, using ONLY train_df to fit the model.

    Returns a report dict. ALWAYS includes buy-and-hold comparison - there is
    no way to call this function and get a result without it.
    """
    config = config or ExperimentConfig()
    data = data.sort_values("timestamp").reset_index(drop=True)
    folds = build_walk_forward_folds(data, target_col, config)

    if not folds:
        raise RuntimeError("No valid folds could be built - check data length and config.")

    fold_reports = []
    for fold_index, (train_df, test_df) in enumerate(folds):
        probability_up = train_and_predict_fn(train_df, test_df, feature_cols, target_col)
        strategy_result = simulate_strategy(test_df, np.asarray(probability_up), config)
        buy_hold_return_pct = calculate_buy_hold_return(test_df) * 100

        fold_reports.append({
            "fold": fold_index,
            "test_start": str(test_df["timestamp"].iloc[0]),
            "test_end": str(test_df["timestamp"].iloc[-1]),
            "test_rows": len(test_df),
            "strategy": strategy_result,
            "buy_hold_return_pct": buy_hold_return_pct,
            "strategy_beat_buy_hold": strategy_result["total_return_pct"] > buy_hold_return_pct,
            "outperformance_pct": strategy_result["total_return_pct"] - buy_hold_return_pct,
        })

    strategy_returns = [f["strategy"]["total_return_pct"] for f in fold_reports]
    buy_hold_returns = [f["buy_hold_return_pct"] for f in fold_reports]
    outperformances = [f["outperformance_pct"] for f in fold_reports]
    fraction_beating = float(np.mean([f["strategy_beat_buy_hold"] for f in fold_reports]))

    mean_outperformance = float(np.mean(outperformances))
    passed = (
        mean_outperformance >= config.min_outperformance_vs_buy_hold_pct
        and fraction_beating >= config.min_fraction_of_folds_beating_buy_hold
    )

    report = {
        "experiment_name": experiment_name,
        "symbol": symbol,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "config": asdict(config),
        "data_window": {
            "start": str(data["timestamp"].iloc[0]),
            "end": str(data["timestamp"].iloc[-1]),
            "rows": len(data),
        },
        "folds": fold_reports,
        "summary": {
            "mean_strategy_return_pct": float(np.mean(strategy_returns)),
            "median_strategy_return_pct": float(np.median(strategy_returns)),
            "mean_buy_hold_return_pct": float(np.mean(buy_hold_returns)),
            "median_buy_hold_return_pct": float(np.median(buy_hold_returns)),
            "mean_outperformance_pct": mean_outperformance,
            "fraction_of_folds_beating_buy_hold": fraction_beating,
        },
        "promotion_bar": {
            "min_outperformance_vs_buy_hold_pct": config.min_outperformance_vs_buy_hold_pct,
            "min_fraction_of_folds_beating_buy_hold": config.min_fraction_of_folds_beating_buy_hold,
        },
        "passed": passed,
        "verdict": (
            f"PASSED - beat buy-and-hold by {mean_outperformance:.1f} points on average, "
            f"in {fraction_beating:.0%} of folds."
            if passed else
            f"DID NOT PASS - {'underperformed' if mean_outperformance < 0 else 'did not sufficiently outperform'} "
            f"buy-and-hold (avg {mean_outperformance:+.1f} points, beat it in only {fraction_beating:.0%} of folds)."
        ),
    }

    _log_experiment(report)
    return report


def _log_experiment(report: dict) -> None:
    """Append this experiment to the running log, so the total count of
    attempts across all strategies/instruments stays visible."""
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if EXPERIMENT_LOG_PATH.exists():
        try:
            log = json.loads(EXPERIMENT_LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log = []

    log.append({
        "experiment_name": report["experiment_name"],
        "symbol": report["symbol"],
        "run_at_utc": report["run_at_utc"],
        "mean_outperformance_pct": report["summary"]["mean_outperformance_pct"],
        "passed": report["passed"],
    })
    EXPERIMENT_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    total = len(log)
    total_passed = sum(1 for e in log if e["passed"])
    print(f"\n[experiment log] This is experiment #{total} across all strategies/instruments tried so far.")
    print(f"[experiment log] {total_passed}/{total} have passed the promotion bar.")
    if total >= 20 and total_passed / total > 0.15:
        print("[experiment log] CAUTION: with this many attempts, some passing results are "
              "statistically expected by chance alone (multiple comparisons problem). "
              "Treat any single 'pass' with extra skepticism - look for it to keep passing "
              "on NEW data/instruments before trusting it.")
