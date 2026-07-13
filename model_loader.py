"""
Safe model loader for the GLD paper-trading project.

Prefers the tuned LightGBM candidate when it exists and its metadata
matches the current symbol, holding period, and feature list.

Falls back to the existing Random Forest model only when the LightGBM
candidate files are absent. It does not silently ignore incompatibility.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from config import HOLD_DAYS, SYMBOL
from features import MODEL_FEATURES


LIGHTGBM_MODEL_PATH = Path("models/lightgbm_model.pkl")
LIGHTGBM_METADATA_PATH = Path("models/lightgbm_model_metadata.json")
RANDOM_FOREST_MODEL_PATH = Path("models/model.pkl")


class ModelCompatibilityError(RuntimeError):
    """Raised when a saved model does not match the current code."""


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelCompatibilityError(
            f"Metadata file is not valid JSON: {path}"
        ) from error


def validate_lightgbm_metadata(metadata: dict[str, Any]) -> None:
    errors: list[str] = []

    if metadata.get("model_type") != "LGBMClassifier":
        errors.append(
            f"model_type is {metadata.get('model_type')!r}, "
            "expected 'LGBMClassifier'"
        )

    if metadata.get("symbol") != SYMBOL:
        errors.append(
            f"symbol is {metadata.get('symbol')!r}, expected {SYMBOL!r}"
        )

    if int(metadata.get("hold_days", -1)) != int(HOLD_DAYS):
        errors.append(
            f"hold_days is {metadata.get('hold_days')!r}, "
            f"expected {HOLD_DAYS!r}"
        )

    saved_features = metadata.get("features")

    if saved_features != MODEL_FEATURES:
        errors.append(
            "saved feature list does not exactly match MODEL_FEATURES"
        )

    if int(metadata.get("feature_count", -1)) != len(MODEL_FEATURES):
        errors.append(
            f"feature_count is {metadata.get('feature_count')!r}, "
            f"expected {len(MODEL_FEATURES)}"
        )

    if errors:
        raise ModelCompatibilityError(
            "LightGBM model compatibility check failed:\n- "
            + "\n- ".join(errors)
        )


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    with path.open("rb") as file:
        return pickle.load(file)


def load_active_model() -> tuple[Any, dict[str, Any]]:
    """
    Return the active model and a description dictionary.

    LightGBM is preferred only when both the model and metadata exist.
    A compatibility problem raises an error instead of falling back.
    """

    lightgbm_model_exists = LIGHTGBM_MODEL_PATH.exists()
    lightgbm_metadata_exists = LIGHTGBM_METADATA_PATH.exists()

    if lightgbm_model_exists or lightgbm_metadata_exists:
        if not (
            lightgbm_model_exists
            and lightgbm_metadata_exists
        ):
            raise ModelCompatibilityError(
                "Only one LightGBM candidate file exists. "
                "Both the model and metadata are required."
            )

        metadata = read_metadata(
            LIGHTGBM_METADATA_PATH
        )

        validate_lightgbm_metadata(
            metadata
        )

        model = load_pickle(
            LIGHTGBM_MODEL_PATH
        )

        return model, {
            "model_name": "lightgbm",
            "model_path": str(
                LIGHTGBM_MODEL_PATH
            ),
            "metadata_path": str(
                LIGHTGBM_METADATA_PATH
            ),
            "model_version": metadata.get(
                "model_version"
            ),
            "features": metadata.get(
                "features"
            ),
        }

    model = load_pickle(
        RANDOM_FOREST_MODEL_PATH
    )

    saved_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if (
        saved_features is not None
        and list(saved_features)
        != MODEL_FEATURES
    ):
        raise ModelCompatibilityError(
            "Fallback Random Forest feature list "
            "does not match MODEL_FEATURES."
        )

    return model, {
        "model_name": "random_forest_fallback",
        "model_path": str(
            RANDOM_FOREST_MODEL_PATH
        ),
        "metadata_path": None,
        "model_version": None,
        "features": MODEL_FEATURES,
    }
