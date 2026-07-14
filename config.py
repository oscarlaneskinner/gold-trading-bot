"""Central configuration for the Gold AI Trading Bot."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
PREVIOUS_MODEL_PATH = MODELS_DIR / "model_previous.pkl"
CANDIDATE_MODEL_PATH = MODELS_DIR / "model_candidate.pkl"

TRADE_LOG_PATH = LOGS_DIR / os.getenv(
    "TRADE_LOG_FILENAME",
    "trades.csv",
)
DECISION_LOG_PATH = LOGS_DIR / os.getenv(
    "DECISION_LOG_FILENAME",
    "decisions.csv",
)

MODEL_REPORT_PATH = REPORTS_DIR / "model_metrics.json"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_metrics.json"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

PAPER_TRADING = True

SYMBOL = "GLD"
DATA_LOOKBACK_DAYS = 500
TRAINING_LOOKBACK_DAYS = 7500

HOLD_DAYS = 20
MAX_HOLD_DAYS = 20

MIN_BUY_CONFIDENCE = 0.50
BEARISH_EXIT_PROBABILITY = 0.40

USE_TREND_FILTER = False
USE_RSI_FILTER = False
USE_VOLATILITY_FILTER = False

RSI_MINIMUM = 45.0
RSI_MAXIMUM = 70.0
MAX_ATR_PERCENT = 0.035

# The stable workflow defaults to 10%.
# The 15% paper-test workflow overrides this with POSITION_PERCENT=0.15.
POSITION_PERCENT = float(
    os.getenv(
        "POSITION_PERCENT",
        "0.10",
    )
)

MAX_POSITION_PERCENT = POSITION_PERCENT

DEFAULT_TRADE_AMOUNT = 1000.0
MIN_ORDER_AMOUNT = 10.0
MINIMUM_ACCOUNT_EQUITY = 100.0

STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20

ENABLE_TRAILING_STOP = False
TRAILING_STOP_PERCENT = 0.04
TRAILING_ACTIVATION_PERCENT = 0.03

TEST_FRACTION = 0.20
RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = 5
RANDOM_FOREST_MIN_SAMPLES_LEAF = 20

MIN_TEST_ROWS = 50
MIN_MODEL_ACCURACY = 0.52
MIN_MODEL_ROC_AUC = 0.52
MAX_ALLOWED_ACCURACY_DROP = 0.005
MAX_ALLOWED_AUC_DROP = 0.005

BACKTEST_INITIAL_CAPITAL = 10_000.0
BACKTEST_SLIPPAGE_PERCENT = 0.0005
BACKTEST_POSITION_PERCENT = POSITION_PERCENT


def create_project_directories() -> None:
    for directory in (
        MODELS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_configuration(
    require_credentials: bool = True,
) -> None:
    errors: list[str] = []

    if require_credentials:
        if not ALPACA_API_KEY:
            errors.append(
                "ALPACA_API_KEY is missing."
            )

        if not ALPACA_SECRET_KEY:
            errors.append(
                "ALPACA_SECRET_KEY is missing."
            )

    if not 0 < MIN_BUY_CONFIDENCE < 1:
        errors.append(
            "MIN_BUY_CONFIDENCE must be between 0 and 1."
        )

    if not 0.01 <= POSITION_PERCENT <= 0.15:
        errors.append(
            "POSITION_PERCENT must be between 0.01 and 0.15 "
            "during the current paper-research phase."
        )

    if RSI_MINIMUM >= RSI_MAXIMUM:
        errors.append(
            "RSI_MINIMUM must be below RSI_MAXIMUM."
        )

    if MAX_HOLD_DAYS < HOLD_DAYS:
        errors.append(
            "MAX_HOLD_DAYS must be at least HOLD_DAYS."
        )

    if not PAPER_TRADING:
        errors.append(
            "PAPER_TRADING must remain True."
        )

    if errors:
        raise ValueError(
            "Invalid configuration:\n- "
            + "\n- ".join(errors)
        )
