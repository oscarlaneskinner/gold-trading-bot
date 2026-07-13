"""Research-only LightGBM tuning for GLD. This script cannot place trades."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from compare_models import (
    MODEL_FEATURES,
    REPORTS_DIR,
    RANDOM_STATE,
    build_folds,
    prepare_dataset,
    simulate,
)

JSON_PATH = REPORTS_DIR / "lightgbm_tuning.json"
CSV_PATH = REPORTS_DIR / "lightgbm_tuning_folds.csv"

PARAMETER_SETS = [
    {
        "name": "lgbm_a",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 30,
    },
    {
        "name": "lgbm_b",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 30,
    },
    {
        "name": "lgbm_c",
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "min_child_samples": 40,
    },
    {
        "name": "lgbm_d",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_child_samples": 40,
    },
]


def build_model(params: dict[str, Any]) -> LGBMClassifier:
    clean = {key: value for key, value in params.items() if key != "name"}
    return LGBMClassifier(
        **clean,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([row["total_return"] for row in rows], dtype=float)
    drawdowns = np.asarray([row["maximum_drawdown"] for row in rows], dtype=float)

    return {
        "folds": len(rows),
        "positive_folds": int((returns > 0).sum()),
        "median_fold_return": float(np.median(returns)),
        "mean_fold_return": float(np.mean(returns)),
        "worst_fold_return": float(np.min(returns)),
        "best_fold_return": float(np.max(returns)),
        "total_trades": int(sum(row["trades"] for row in rows)),
        "mean_win_rate": float(np.mean([row["win_rate"] for row in rows])),
        "mean_profit_factor": float(np.mean([
            min(float(row["profit_factor"]), 10.0) for row in rows
        ])),
        "mean_maximum_drawdown": float(np.mean(drawdowns)),
        "worst_maximum_drawdown": float(np.min(drawdowns)),
    }


def score(summary: dict[str, Any]) -> float:
    trade_score = min(1.0, summary["total_trades"] / 50.0)
    return float(
        4.0 * summary["median_fold_return"]
        + 2.0 * summary["mean_fold_return"]
        + 0.015 * summary["positive_folds"]
        + 0.010 * min(3.0, summary["mean_profit_factor"])
        + 0.015 * trade_score
        + 1.5 * summary["worst_maximum_drawdown"]
    )


def run() -> None:
    frame = prepare_dataset()
    folds = build_folds(frame)
    rows: list[dict[str, Any]] = []

    for params in PARAMETER_SETS:
        print(f"\nTesting {params['name']}")

        for fold_number, (training, testing) in enumerate(folds, start=1):
            model = build_model(params)
            model.fit(training[MODEL_FEATURES], training["target"])
            result = simulate(model, testing)

            row = {
                "configuration": params["name"],
                "fold": fold_number,
                **{key: value for key, value in params.items() if key != "name"},
                **result,
            }
            rows.append(row)

            print(
                f"  fold={fold_number}, "
                f"return={result['total_return']:.2%}, "
                f"trades={result['trades']}, "
                f"max_dd={result['maximum_drawdown']:.2%}"
            )

    summaries = {}
    for params in PARAMETER_SETS:
        config_rows = [
            row for row in rows if row["configuration"] == params["name"]
        ]
        summary = summarize(config_rows)
        summary["research_score"] = score(summary)
        summary["parameters"] = {
            key: value for key, value in params.items() if key != "name"
        }
        summaries[params["name"]] = summary

    ranking = sorted(
        summaries,
        key=lambda name: summaries[name]["research_score"],
        reverse=True,
    )

    report = {
        "summaries": summaries,
        "recommendation": {
            "research_winner": ranking[0],
            "ranking": ranking,
            "winner_details": summaries[ranking[0]],
            "note": "Research only. Do not deploy until compared with the current baseline.",
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    print("\n=== LIGHTGBM TUNING RESULT ===")
    print(json.dumps(report["recommendation"], indent=2))
    print(f"\nJSON report: {JSON_PATH}")
    print(f"CSV report: {CSV_PATH}")


if __name__ == "__main__":
    run()
