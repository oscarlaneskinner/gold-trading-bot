"""
Create and test a new bot configuration for any symbol, using the same
validated harness (mandatory buy-and-hold comparison) as everything else.

This does NOT deploy a live trading bot. It only tests a configuration and
reports honest results. Deploying live is a deliberate separate step.

Reads parameters from environment variables (set by the GitHub Actions
workflow's input fields):
  BOT_SYMBOL          - ticker symbol, e.g. "MSFT"
  BOT_STOP_LOSS_PCT    - e.g. "10" for 10%
  BOT_TAKE_PROFIT_PCT  - e.g. "20" for 20%
  BOT_HOLD_DAYS        - e.g. "20"
  BOT_MODEL_TYPE       - "random_forest", "lightgbm", or "xgboost"
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import get_market_data
from features import MODEL_FEATURES, add_features
from harness import ExperimentConfig, run_experiment

DISCLAIMER = (
    "DISCLAIMER: This is a backtest of a rules-based strategy, not investment "
    "advice. Past performance does not guarantee future results. Every "
    "strategy tested so far in this project has either failed to beat, or "
    "only roughly matched, simply buying and holding the underlying asset. "
    "A 'PASSED' result below means this specific configuration beat "
    "buy-and-hold in this specific historical backtest - it does not mean "
    "it will do so in the future. Treat any bot built here as an experiment, "
    "not a proven source of income."
)


def get_predict_fn(model_type: str):
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        def predict(train_df, test_df, feature_cols, target_col):
            model = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            )
            model.fit(train_df[feature_cols], train_df[target_col])
            return model.predict_proba(test_df[feature_cols])[:, list(model.classes_).index(1)]
        return predict

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        def predict(train_df, test_df, feature_cols, target_col):
            model = LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.02, num_leaves=31,
                min_child_samples=40, subsample=0.80, colsample_bytree=0.80,
                reg_alpha=0.10, reg_lambda=2.0, class_weight="balanced",
                random_state=42, n_jobs=-1, verbosity=-1,
            )
            model.fit(train_df[feature_cols], train_df[target_col])
            return model.predict_proba(test_df[feature_cols])[:, list(model.classes_).index(1)]
        return predict

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        def predict(train_df, test_df, feature_cols, target_col):
            model = XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.025, subsample=0.80,
                colsample_bytree=0.80, min_child_weight=10, reg_alpha=0.10,
                reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
            model.fit(train_df[feature_cols], train_df[target_col])
            return model.predict_proba(test_df[feature_cols])[:, list(model.classes_).index(1)]
        return predict

    raise ValueError(f"Unknown model_type: {model_type}. Use random_forest, lightgbm, or xgboost.")


def run():
    symbol = os.environ.get("BOT_SYMBOL", "GLD").strip().upper()
    stop_loss_pct = float(os.environ.get("BOT_STOP_LOSS_PCT", "10")) / 100
    take_profit_pct = float(os.environ.get("BOT_TAKE_PROFIT_PCT", "20")) / 100
    hold_days = int(os.environ.get("BOT_HOLD_DAYS", "20"))
    model_type = os.environ.get("BOT_MODEL_TYPE", "random_forest").strip().lower()

    print(DISCLAIMER)
    print(f"\n=== Testing new bot config ===")
    print(f"Symbol: {symbol}")
    print(f"Stop-loss: {stop_loss_pct:.0%}  Take-profit: {take_profit_pct:.0%}  Hold days: {hold_days}")
    print(f"Model type: {model_type}\n")

    lookback_days = max(2500, hold_days * 150)  # enough history for meaningful folds
    raw = get_market_data(symbol, lookback_days)
    frame = add_features(raw)
    frame["future_close"] = frame["close"].shift(-hold_days)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    clean = frame.dropna(subset=MODEL_FEATURES + ["target"]).reset_index(drop=True)

    print(f"Data: {len(clean)} rows, {clean['timestamp'].min().date()} to {clean['timestamp'].max().date()}\n")

    config = ExperimentConfig(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_days=hold_days,
    )

    predict_fn = get_predict_fn(model_type)

    result = run_experiment(
        experiment_name=f"user_created_{symbol}_{model_type}",
        symbol=symbol,
        data=clean,
        feature_cols=MODEL_FEATURES,
        target_col="target",
        train_and_predict_fn=predict_fn,
        config=config,
        notes=f"User-created bot via Create New Bot workflow. Model={model_type}.",
    )

    print(f"\n{'='*60}")
    print(result["verdict"])
    print(f"{'='*60}\n")

    for f in result["folds"]:
        beat = "YES" if f["strategy_beat_buy_hold"] else "no"
        print(f"Fold {f['fold']}: {f['test_start'][:10]} to {f['test_end'][:10]} | "
              f"strategy {f['strategy']['total_return_pct']:>7.1f}% | "
              f"buy&hold {f['buy_hold_return_pct']:>7.1f}% | beat it? {beat}")

    print(f"\nAverage strategy return: {result['summary']['mean_strategy_return_pct']:.1f}%")
    print(f"Average buy&hold return: {result['summary']['mean_buy_hold_return_pct']:.1f}%")
    print(f"Average outperformance: {result['summary']['mean_outperformance_pct']:+.1f} points")

    if result["passed"]:
        print(f"\nThis configuration PASSED the promotion bar. If you want to deploy it "
              f"live (paper trading), that is a separate, deliberate step - this workflow "
              f"only tests and reports, it does not deploy anything.")
    else:
        print(f"\nThis configuration DID NOT beat buy-and-hold by the required margin. "
              f"Per the disclaimer above, this is consistent with most configurations "
              f"tested in this project. Consider this a learning exercise rather than "
              f"a bot ready to trade with.")

    report_path = Path(f"reports/user_created_{symbol}_{model_type}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["disclaimer"] = DISCLAIMER
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    run()
