"""Offline verification for the saved GLD LightGBM model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from config import HOLD_DAYS, SYMBOL
from features import MODEL_FEATURES

MODEL_PATH = Path("models/lightgbm_model.pkl")
METADATA_PATH = Path("models/lightgbm_model_metadata.json")


def run() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    if metadata.get("model_type") != "LGBMClassifier":
        errors.append("model_type is not LGBMClassifier")

    if metadata.get("symbol") != SYMBOL:
        errors.append(
            f"symbol mismatch: {metadata.get('symbol')!r} != {SYMBOL!r}"
        )

    if int(metadata.get("hold_days", -1)) != int(HOLD_DAYS):
        errors.append(
            f"hold_days mismatch: {metadata.get('hold_days')!r} != {HOLD_DAYS!r}"
        )

    if metadata.get("features") != MODEL_FEATURES:
        errors.append("saved feature list does not match MODEL_FEATURES")

    if int(metadata.get("feature_count", -1)) != len(MODEL_FEATURES):
        errors.append("saved feature_count does not match MODEL_FEATURES")

    if errors:
        raise RuntimeError(
            "Offline compatibility check failed:\n- " + "\n- ".join(errors)
        )

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    test_row = pd.DataFrame(
        [[0.0] * len(MODEL_FEATURES)],
        columns=MODEL_FEATURES,
    )

    prediction = int(model.predict(test_row)[0])
    positive_index = list(model.classes_).index(1)
    probability_up = float(model.predict_proba(test_row)[0][positive_index])

    print("Offline LightGBM verification passed")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model version: {metadata.get('model_version')}")
    print(f"Symbol: {metadata.get('symbol')}")
    print(f"Hold days: {metadata.get('hold_days')}")
    print(f"Feature count: {len(MODEL_FEATURES)}")
    print("Synthetic-row test only — not a market signal")
    print(f"Prediction: {prediction}")
    print(f"Probability up: {probability_up:.6f}")
    print("No network request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
