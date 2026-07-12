"""Train, evaluate, and conditionally promote a GLD Random Forest model."""

from __future__ import annotations
import json
import pickle
import shutil
from datetime import datetime, timezone
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from config import (
    CANDIDATE_MODEL_PATH, HOLD_DAYS, MAX_ALLOWED_ACCURACY_DROP,
    MAX_ALLOWED_AUC_DROP, MIN_MODEL_ACCURACY, MIN_MODEL_ROC_AUC,
    MIN_TEST_ROWS, MODEL_METADATA_PATH, MODEL_PATH, MODEL_REPORT_PATH,
    PREVIOUS_MODEL_PATH, RANDOM_FOREST_ESTIMATORS,
    RANDOM_FOREST_MAX_DEPTH, RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_STATE, SYMBOL, TEST_FRACTION, TRAINING_LOOKBACK_DAYS,
    create_project_directories,
)
from data import get_market_data
from features import MODEL_FEATURES, add_features
from logger import write_json

def prepare_dataset():
    frame = add_features(get_market_data(SYMBOL, TRAINING_LOOKBACK_DAYS))
    frame["future_close"] = frame["close"].shift(-HOLD_DAYS)
    frame = frame[frame["future_close"].notna()].copy()
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=MODEL_FEATURES + ["target"]).reset_index(drop=True)

def evaluate(model, X, y):
    prediction = model.predict(X)
    probability_up = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability_up)),
    }

def save_pickle(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)

def load_current_model():
    if not MODEL_PATH.exists():
        return None
    try:
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
        expected = list(getattr(model, "feature_names_in_", []))
        return model if not expected or expected == MODEL_FEATURES else None
    except Exception as error:
        print(f"Current model could not be loaded: {error}")
        return None

def train():
    create_project_directories()
    data = prepare_dataset()
    split = int(len(data) * (1 - TEST_FRACTION))
    train_data, test_data = data.iloc[:split], data.iloc[split:]

    if len(test_data) < MIN_TEST_ROWS:
        raise RuntimeError(f"Only {len(test_data)} testing rows are available.")
    if train_data["target"].nunique() < 2 or test_data["target"].nunique() < 2:
        raise RuntimeError("Training and testing sets must each contain both classes.")

    X_train, y_train = train_data[MODEL_FEATURES], train_data["target"]
    X_test, y_test = test_data[MODEL_FEATURES], test_data["target"]

    candidate = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_ESTIMATORS,
        max_depth=RANDOM_FOREST_MAX_DEPTH,
        min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    candidate.fit(X_train, y_train)
    candidate_metrics = evaluate(candidate, X_test, y_test)
    save_pickle(candidate, CANDIDATE_MODEL_PATH)

    current = load_current_model()
    current_metrics = evaluate(current, X_test, y_test) if current is not None else None

    absolute_pass = (
        candidate_metrics["accuracy"] >= MIN_MODEL_ACCURACY
        and candidate_metrics["roc_auc"] >= MIN_MODEL_ROC_AUC
    )
    comparison_pass = True
    if current_metrics is not None:
        comparison_pass = (
            candidate_metrics["accuracy"] >= current_metrics["accuracy"] - MAX_ALLOWED_ACCURACY_DROP
            and candidate_metrics["roc_auc"] >= current_metrics["roc_auc"] - MAX_ALLOWED_AUC_DROP
        )

    promoted = absolute_pass and comparison_pass
    if promoted:
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, PREVIOUS_MODEL_PATH)
        save_pickle(candidate, MODEL_PATH)

    report = {
        "status": "promoted" if promoted else "rejected",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "hold_days": HOLD_DAYS,
        "scikit_learn_version": sklearn_version,
        "features": MODEL_FEATURES,
        "training_rows": len(train_data),
        "testing_rows": len(test_data),
        "candidate_metrics": candidate_metrics,
        "current_metrics": current_metrics,
        "thresholds": {
            "minimum_accuracy": MIN_MODEL_ACCURACY,
            "minimum_roc_auc": MIN_MODEL_ROC_AUC,
        },
    }
    write_json(MODEL_REPORT_PATH, report)

    if promoted:
        write_json(
            MODEL_METADATA_PATH,
            {
                "created_at_utc": report["created_at_utc"],
                "scikit_learn_version": sklearn_version,
                "features": MODEL_FEATURES,
                "metrics": candidate_metrics,
            },
        )

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    train()
