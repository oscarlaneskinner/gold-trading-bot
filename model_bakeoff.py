"""
Honest model bake-off: tests candidate model types through the shared harness
(harness.py), so every candidate gets the SAME mandatory buy-and-hold
comparison. This replaces compare_ensemble.py's "research_score" ranking,
which only compared candidates against each other and never checked whether
ANY of them actually beat simply holding the asset - that gap is exactly
what let LightGBM get anointed "best" without ever being asked the question
that actually matters.

Run this in GitHub Actions (needs lightgbm, xgboost, config.py, data.py,
features.py installed/available - this sandbox doesn't have internet access
to install them).
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import HOLD_DAYS, SYMBOL, TRAINING_LOOKBACK_DAYS
from data import get_market_data
from features import MODEL_FEATURES, add_features
from harness import ExperimentConfig, run_experiment

REPORT_PATH = Path("reports/model_bakeoff.json")


def prepare_dataset():
    frame = add_features(get_market_data(SYMBOL, TRAINING_LOOKBACK_DAYS))
    frame["future_close"] = frame["close"].shift(-HOLD_DAYS)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=MODEL_FEATURES + ["target"]).reset_index(drop=True)


# ============ CANDIDATE 1: Random Forest (our original validated baseline) ============
def random_forest_predict(train_df, test_df, feature_cols, target_col):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model.predict_proba(test_df[feature_cols])[:, list(model.classes_).index(1)]


# ============ CANDIDATE 2: LightGBM (the "research_score" winner - untested vs buy&hold until now) ============
def lightgbm_predict(train_df, test_df, feature_cols, target_col):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.02, num_leaves=31,
        min_child_samples=40, subsample=0.80, colsample_bytree=0.80,
        reg_alpha=0.10, reg_lambda=2.0, class_weight="balanced",
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model.predict_proba(test_df[feature_cols])[:, list(model.classes_).index(1)]


# ============ CANDIDATE 3: XGBoost ============
def xgboost_predict(train_df, test_df, feature_cols, target_col):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=350, max_depth=3, learning_rate=0.025, subsample=0.80,
        colsample_bytree=0.80, min_child_weight=10, reg_alpha=0.10, reg_lambda=2.0,
        objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model.predict_proba(test_df[feature_cols])[:, list(model.classes_).index(1)]


CANDIDATES = {
    "random_forest": random_forest_predict,
    "lightgbm": lightgbm_predict,
    "xgboost": xgboost_predict,
}


def run():
    data = prepare_dataset()
    results = {}

    for name, predict_fn in CANDIDATES.items():
        print(f"\n{'='*60}\nCANDIDATE: {name}\n{'='*60}")
        try:
            result = run_experiment(
                experiment_name=f"bakeoff_{name}_{SYMBOL}",
                symbol=SYMBOL,
                data=data,
                feature_cols=MODEL_FEATURES,
                target_col="target",
                train_and_predict_fn=predict_fn,
                config=ExperimentConfig(n_folds=5),
                notes=f"Model bake-off candidate: {name}",
            )
            results[name] = result
            print(result["verdict"])
            print(f"Mean outperformance vs buy&hold: {result['summary']['mean_outperformance_pct']:+.1f} points")
        except Exception as error:
            print(f"FAILED: {error}")
            results[name] = {"error": str(error)}

    # Only recommend a winner if at least one candidate actually passed
    passing = {name: r for name, r in results.items() if r.get("passed")}

    print(f"\n{'='*60}\nBAKE-OFF SUMMARY\n{'='*60}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name}: ERROR - {r['error']}")
            continue
        print(f"{name}: mean outperformance {r['summary']['mean_outperformance_pct']:+.1f} pts, "
              f"passed={r['passed']}")

    if passing:
        winner = max(passing, key=lambda n: passing[n]["summary"]["mean_outperformance_pct"])
        print(f"\nRECOMMENDATION: promote '{winner}' - it's the only candidate (or best "
              f"among candidates) that actually beat buy-and-hold by the required margin.")
        recommendation = winner
    else:
        print(f"\nRECOMMENDATION: promote NONE of these candidates. Not one of them beat "
              f"buy-and-hold by the required margin ({ExperimentConfig().min_outperformance_vs_buy_hold_pct} "
              f"points, in {ExperimentConfig().min_fraction_of_folds_beating_buy_hold:.0%}+ of folds). "
              f"This matches the pattern found on GLD, silver, and AMZN today - deploying any of "
              f"these as 'the best bot' would just be picking the least-bad loser, not a winner.")
        recommendation = None

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "results": {k: v.get("summary", v) for k, v in results.items()},
        "recommendation": recommendation,
    }, indent=2, default=str), encoding="utf-8")

    return results


if __name__ == "__main__":
    run()
