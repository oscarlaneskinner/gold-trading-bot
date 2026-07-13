"""
Train the production-candidate LightGBM model for the GLD paper-trading bot.

This script:
- uses the current 13-feature list from features.py
- predicts whether GLD will be higher after HOLD_DAYS
- removes rows without a known future close
- trains the selected lgbm_d configuration
- saves a separate LightGBM model
- saves metadata used for compatibility checks

It does not place orders.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import HOLD_DAYS, SYMBOL
from data import get_market_data
from features import MODEL_FEATURES, add_features


TRAINING_LOOKBACK_DAYS = 5000
TEST_FRACTION = 0.20
RANDOM_STATE = 42

MODEL_PATH = Path("models/lightgbm_model.pkl")
METADATA_PATH = Path("models/lightgbm_model_metadata.json")
REPORT_PATH = Path("reports/lightgbm_production_training.json")


def prepare_dataset():
    frame = add_features(
        get_market_data(
            symbol=SYMBOL,
            lookback_days=TRAINING_LOOKBACK_DAYS,
        )
    )

    frame["future_close"] = frame["close"].shift(-HOLD_DAYS)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)

    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = (
        frame.dropna(subset=MODEL_FEATURES + ["target"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(frame) < 1000:
        raise RuntimeError(
            f"Only {len(frame)} usable rows are available; at least 1000 are required."
        )

    return frame


def build_model():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def evaluate(model, X, y):
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)
    probability_up = probabilities[:, positive_index]

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability_up)),
    }


def atomic_pickle_dump(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    temporary.replace(path)


def atomic_json_dump(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def train():
    frame = prepare_dataset()

    split_index = int(len(frame) * (1 - TEST_FRACTION))
    train_frame = frame.iloc[:split_index].copy()
    test_frame = frame.iloc[split_index:].copy()

    if train_frame["target"].nunique() < 2 or test_frame["target"].nunique() < 2:
        raise RuntimeError("Training and testing sets must each contain both classes.")

    X_train = train_frame[MODEL_FEATURES]
    y_train = train_frame["target"]
    X_test = test_frame[MODEL_FEATURES]
    y_test = test_frame["target"]

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    created_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_type": "LGBMClassifier",
        "model_version": "lgbm_d",
        "created_at_utc": created_at,
        "symbol": SYMBOL,
        "hold_days": HOLD_DAYS,
        "features": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "scikit_learn_version": sklearn_version,
        "parameters": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "validation_metrics": metrics,
        "training_rows": len(train_frame),
        "testing_rows": len(test_frame),
        "training_start": str(train_frame.iloc[0]["timestamp"]),
        "training_end": str(train_frame.iloc[-1]["timestamp"]),
        "testing_start": str(test_frame.iloc[0]["timestamp"]),
        "testing_end": str(test_frame.iloc[-1]["timestamp"]),
    }

    report = {
        "status": "trained",
        **metadata,
    }

    atomic_pickle_dump(model, MODEL_PATH)
    atomic_json_dump(metadata, METADATA_PATH)
    atomic_json_dump(report, REPORT_PATH)

    print(json.dumps(report, indent=2))
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    train()
