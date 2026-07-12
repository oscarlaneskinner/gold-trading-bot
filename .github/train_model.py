"""
Training and model-promotion pipeline for the GLD AI trading bot.

This script:
1. Downloads historical GLD data.
2. Builds the same features used by the daily bot.
3. Creates a forward-looking target.
4. Uses a chronological train/test split.
5. Trains a candidate Random Forest model.
6. Evaluates the candidate and current production model.
7. Promotes the candidate only when validation rules pass.
8. Writes a JSON model report.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import HOLD_DAYS, MODEL_PATH, SYMBOL
from data import get_market_data
from features import MODEL_FEATURES, add_features


# ============================================================
# TRAINING SETTINGS
# ============================================================

LOOKBACK_DAYS = 2500
TEST_FRACTION = 0.20

RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_DEPTH = 7
MIN_SAMPLES_LEAF = 5

# Minimum standards for promoting a candidate model.
MIN_TEST_ROWS = 50
MIN_ACCURACY = 0.52
MIN_ROC_AUC = 0.52

# Candidate should not be materially worse than the current model.
MAX_ALLOWED_ACCURACY_DROP = 0.005
MAX_ALLOWED_AUC_DROP = 0.005


# ============================================================
# FILE LOCATIONS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PRODUCTION_MODEL_PATH = ROOT_DIR / MODEL_PATH
CANDIDATE_MODEL_PATH = ROOT_DIR / "models" / "model_candidate.pkl"
BACKUP_MODEL_PATH = ROOT_DIR / "models" / "model_previous.pkl"

REPORT_PATH = ROOT_DIR / "reports" / "model_metrics.json"


def create_target(df):
    """
    Create the prediction target.

    target = 1 when GLD's closing price after HOLD_DAYS is greater
    than today's closing price; otherwise target = 0.

    Rows without a known future closing price are explicitly removed.
    """
    result = df.copy()

    result["future_close"] = result["close"].shift(-HOLD_DAYS)

    # This prevents the last HOLD_DAYS rows from being incorrectly
    # labeled as zero when no future price exists.
    result = result[result["future_close"].notna()].copy()

    result["target"] = (
        result["future_close"] > result["close"]
    ).astype(int)

    return result


def prepare_dataset():
    print("Downloading GLD market data...")

    df = get_market_data(
        symbol=SYMBOL,
        lookback_days=LOOKBACK_DAYS,
    )

    if df is None or df.empty:
        raise RuntimeError("No market data was returned.")

    print(f"Downloaded {len(df)} raw rows.")

    df = add_features(df)
    df = create_target(df)

    required_columns = MODEL_FEATURES + [
        "target",
        "timestamp",
        "close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Replace infinities, then remove incomplete feature rows.
    df[MODEL_FEATURES] = df[MODEL_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = (
        df.dropna(subset=MODEL_FEATURES + ["target"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(df) < 250:
        raise RuntimeError(
            f"Only {len(df)} usable rows remain. "
            "At least 250 are required."
        )

    return df


def chronological_split(df):
    """
    Split data in time order.

    Earlier observations are used for training.
    The newest observations are held out for testing.
    """
    split_index = int(len(df) * (1 - TEST_FRACTION))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(
            f"Test set has only {len(test_df)} rows. "
            f"At least {MIN_TEST_ROWS} are required."
        )

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["target"]

    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"]

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training target contains only one class."
        )

    if y_test.nunique() < 2:
        raise RuntimeError(
            "Test target contains only one class; "
            "ROC-AUC cannot be evaluated reliably."
        )

    return (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
    )


def build_candidate_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def positive_class_probabilities(model, X):
    """
    Return probability estimates for target class 1.
    """
    probabilities = model.predict_proba(X)

    class_labels = list(model.classes_)

    if 1 not in class_labels:
        raise RuntimeError(
            "The model does not contain positive class 1."
        )

    positive_index = class_labels.index(1)

    return probabilities[:, positive_index]


def evaluate_model(model, X, y) -> dict[str, Any]:
    predictions = model.predict(X)
    probabilities = positive_class_probabilities(model, X)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "confusion_matrix": matrix.tolist(),
        "predicted_up_rate": float(
            np.mean(predictions == 1)
        ),
        "actual_up_rate": float(
            np.mean(np.asarray(y) == 1)
        ),
    }


def load_existing_model():
    if not PRODUCTION_MODEL_PATH.exists():
        print("No current production model was found.")
        return None

    try:
        with PRODUCTION_MODEL_PATH.open("rb") as file:
            model = pickle.load(file)

        expected_features = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if expected_features is not None:
            expected_features = list(expected_features)

            if expected_features != MODEL_FEATURES:
                print(
                    "Current model uses a different feature set. "
                    "It will not be used for direct comparison."
                )
                return None

        return model

    except Exception as exc:
        print(
            "Could not load the existing production model: "
            f"{exc}"
        )
        return None


def candidate_passes_absolute_rules(
    metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []

    if metrics["accuracy"] < MIN_ACCURACY:
        reasons.append(
            f"accuracy {metrics['accuracy']:.4f} "
            f"is below {MIN_ACCURACY:.4f}"
        )

    if metrics["roc_auc"] < MIN_ROC_AUC:
        reasons.append(
            f"ROC-AUC {metrics['roc_auc']:.4f} "
            f"is below {MIN_ROC_AUC:.4f}"
        )

    return len(reasons) == 0, reasons


def candidate_is_not_worse(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    if current_metrics is None:
        return True, [
            "No compatible current model was available "
            "for comparison."
        ]

    reasons = []

    minimum_accuracy = (
        current_metrics["accuracy"]
        - MAX_ALLOWED_ACCURACY_DROP
    )

    minimum_auc = (
        current_metrics["roc_auc"]
        - MAX_ALLOWED_AUC_DROP
    )

    if candidate_metrics["accuracy"] < minimum_accuracy:
        reasons.append(
            "Candidate accuracy is materially below "
            "the current model."
        )

    if candidate_metrics["roc_auc"] < minimum_auc:
        reasons.append(
            "Candidate ROC-AUC is materially below "
            "the current model."
        )

    return len(reasons) == 0, reasons


def save_pickle(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as file:
        pickle.dump(
            model,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(temporary_path, path)


def promote_candidate(candidate_model):
    """
    Back up the current model and atomically promote the candidate.
    """
    PRODUCTION_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PRODUCTION_MODEL_PATH.exists():
        shutil.copy2(
            PRODUCTION_MODEL_PATH,
            BACKUP_MODEL_PATH,
        )

    save_pickle(
        candidate_model,
        PRODUCTION_MODEL_PATH,
    )

    print(
        f"Candidate promoted to {PRODUCTION_MODEL_PATH}"
    )


def save_report(report: dict[str, Any]):
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = REPORT_PATH.with_suffix(
        ".json.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            sort_keys=True,
        )

    os.replace(
        temporary_path,
        REPORT_PATH,
    )

    print(f"Report saved to {REPORT_PATH}")


def print_metrics(
    title: str,
    metrics: dict[str, Any],
):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1']:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(
        "Confusion matrix "
        "[[TN, FP], [FN, TP]]:"
    )
    print(metrics["confusion_matrix"])


def train():
    started_at = datetime.now(timezone.utc)

    df = prepare_dataset()

    (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(df)

    print(
        f"Training rows: {len(train_df)}"
    )
    print(
        f"Testing rows: {len(test_df)}"
    )
    print(
        "Training period: "
        f"{train_df.iloc[0]['timestamp']} through "
        f"{train_df.iloc[-1]['timestamp']}"
    )
    print(
        "Testing period: "
        f"{test_df.iloc[0]['timestamp']} through "
        f"{test_df.iloc[-1]['timestamp']}"
    )

    candidate_model = build_candidate_model()

    print("\nTraining candidate Random Forest...")

    candidate_model.fit(
        X_train,
        y_train,
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test,
    )

    print_metrics(
        "Candidate model",
        candidate_metrics,
    )

    # Save the candidate separately for audit/debugging.
    save_pickle(
        candidate_model,
        CANDIDATE_MODEL_PATH,
    )

    current_model = load_existing_model()
    current_metrics = None

    if current_model is not None:
        try:
            current_metrics = evaluate_model(
                current_model,
                X_test,
                y_test,
            )

            print_metrics(
                "Current production model",
                current_metrics,
            )

        except Exception as exc:
            print(
                "The current model could not be evaluated "
                f"on the new feature set: {exc}"
            )
            current_metrics = None

    absolute_pass, absolute_reasons = (
        candidate_passes_absolute_rules(
            candidate_metrics
        )
    )

    comparison_pass, comparison_reasons = (
        candidate_is_not_worse(
            candidate_metrics,
            current_metrics,
        )
    )

    promoted = absolute_pass and comparison_pass

    decision_reasons = (
        absolute_reasons + comparison_reasons
    )

    if promoted:
        promote_candidate(candidate_model)

        if not decision_reasons:
            decision_reasons = [
                "Candidate passed all validation rules."
            ]

        status = "promoted"

    else:
        print(
            "\nCandidate was NOT promoted. "
            "The current production model remains unchanged."
        )

        for reason in decision_reasons:
            print(f"- {reason}")

        status = "rejected"

    completed_at = datetime.now(timezone.utc)

    report = {
        "status": status,
        "symbol": SYMBOL,
        "hold_days": HOLD_DAYS,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "scikit_learn_version": sklearn_version,
        "model_type": "RandomForestClassifier",
        "model_parameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "feature_count": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "dataset": {
            "usable_rows": len(df),
            "training_rows": len(train_df),
            "testing_rows": len(test_df),
            "training_start": str(
                train_df.iloc[0]["timestamp"]
            ),
            "training_end": str(
                train_df.iloc[-1]["timestamp"]
            ),
            "testing_start": str(
                test_df.iloc[0]["timestamp"]
            ),
            "testing_end": str(
                test_df.iloc[-1]["timestamp"]
            ),
        },
        "thresholds": {
            "minimum_accuracy": MIN_ACCURACY,
            "minimum_roc_auc": MIN_ROC_AUC,
            "maximum_allowed_accuracy_drop":
                MAX_ALLOWED_ACCURACY_DROP,
            "maximum_allowed_auc_drop":
                MAX_ALLOWED_AUC_DROP,
        },
        "candidate_metrics": candidate_metrics,
        "current_model_metrics": current_metrics,
        "decision_reasons": decision_reasons,
        "production_model_path": str(
            PRODUCTION_MODEL_PATH
        ),
        "candidate_model_path": str(
            CANDIDATE_MODEL_PATH
        ),
    }

    save_report(report)

    print(f"\nTraining result: {status.upper()}")

    # Rejecting a candidate is a valid result, not a workflow error.
    return report


if __name__ == "__main__":
    try:
        train()

    except Exception as error:
        print(
            f"\nTraining pipeline failed: {error}",
            file=sys.stderr,
        )
        raise
