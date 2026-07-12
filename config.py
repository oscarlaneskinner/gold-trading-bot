"""
Central configuration for the Gold AI Trading Bot.

Keep PAPER_TRADING set to True while developing and testing.
API credentials are read from environment variables or GitHub Secrets.
"""

from __future__ import annotations

import os
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "model.pkl"
CANDIDATE_MODEL_PATH = MODELS_DIR / "model_candidate.pkl"
PREVIOUS_MODEL_PATH = MODELS_DIR / "model_previous.pkl"

TRADE_LOG_PATH = LOGS_DIR / "trade_log.csv"
DECISION_LOG_PATH = LOGS_DIR / "decision_log.csv"
MODEL_REPORT_PATH = REPORTS_DIR / "model_metrics.json"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_metrics.json"


# ============================================================
# ALPACA CREDENTIALS
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Keep True until extensive backtesting and paper testing are complete.
PAPER_TRADING = True


# ============================================================
# MARKET SETTINGS
# ============================================================

SYMBOL = "GLD"

# Daily bars.
DATA_LOOKBACK_DAYS = 500
TRAINING_LOOKBACK_DAYS = 2500

# Number of future trading days the AI attempts to predict.
HOLD_DAYS = 5


# ============================================================
# AI SIGNAL SETTINGS
# ============================================================

# The bot will not enter a new position unless the model predicts
# an upward move with at least this probability.
MIN_BUY_CONFIDENCE = 0.70

# Close a position early when the model becomes sufficiently bearish.
ENABLE_BEARISH_MODEL_EXIT = True
BEARISH_EXIT_PROBABILITY = 0.40

# Technical filters.
USE_TREND_FILTER = True
USE_RSI_FILTER = True
USE_VOLATILITY_FILTER = True

RSI_MINIMUM = 45.0
RSI_MAXIMUM = 70.0

# Price must be above the 200-day EMA when enabled.
REQUIRE_PRICE_ABOVE_EMA_200 = True

# Require short-term moving-average alignment:
# EMA 9 > EMA 21 > EMA 50.
REQUIRE_BULLISH_EMA_ALIGNMENT = True

# Skip unusually volatile entries when ATR exceeds this percentage
# of the current price.
MAX_ATR_PERCENT = 0.035


# ============================================================
# POSITION SIZING
# ============================================================

# Maximum percentage of account equity allocated to one GLD position.
MAX_POSITION_PERCENT = 0.10

# Fallback trade size if equity-based sizing cannot be calculated.
DEFAULT_TRADE_AMOUNT = 1000.00

# Minimum order value.
MIN_ORDER_AMOUNT = 10.00

# Alpaca supports fractional stock orders through notional values.
USE_NOTIONAL_ORDERS = True


# ============================================================
# EXIT AND RISK SETTINGS
# ============================================================

# ATR-based estimated risk levels used by the strategy and logs.
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

# Broker-independent percentage exits checked during the daily run.
STOP_LOSS_PERCENT = 0.05
TAKE_PROFIT_PERCENT = 0.10

ENABLE_TRAILING_STOP = True
TRAILING_STOP_PERCENT = 0.04
TRAILING_ACTIVATION_PERCENT = 0.03

# Maximum time a trade may remain open.
MAX_HOLD_DAYS = 20

# Prevent the bot from adding to an existing position.
ALLOW_POSITION_PYRAMIDING = False


# ============================================================
# ACCOUNT SAFETY LIMITS
# ============================================================

# Do not submit new orders if the account is below this equity.
MINIMUM_ACCOUNT_EQUITY = 100.00

# Optional loss limits. They will be enforced once account-performance
# tracking is added. None disables a limit.
MAX_DAILY_LOSS_PERCENT = None
MAX_WEEKLY_LOSS_PERCENT = None


# ============================================================
# MODEL TRAINING SETTINGS
# ============================================================

TEST_FRACTION = 0.20

RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 500
RANDOM_FOREST_MAX_DEPTH = 7
RANDOM_FOREST_MIN_SAMPLES_LEAF = 5

MIN_TEST_ROWS = 50
MIN_MODEL_ACCURACY = 0.52
MIN_MODEL_ROC_AUC = 0.52

MAX_ALLOWED_ACCURACY_DROP = 0.005
MAX_ALLOWED_AUC_DROP = 0.005


# ============================================================
# BACKTEST SETTINGS
# ============================================================

BACKTEST_INITIAL_CAPITAL = 10_000.00
BACKTEST_COMMISSION_PER_TRADE = 0.00
BACKTEST_SLIPPAGE_PERCENT = 0.0005

# Capital allocation used in the historical simulation.
BACKTEST_POSITION_PERCENT = MAX_POSITION_PERCENT


# ============================================================
# LOGGING AND NOTIFICATIONS
# ============================================================

LOG_LEVEL = "INFO"

# Optional Discord webhook stored in GitHub Secrets.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

ENABLE_DISCORD_NOTIFICATIONS = bool(DISCORD_WEBHOOK_URL)


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration(require_credentials: bool = True) -> None:
    """
    Validate configuration before connecting to Alpaca.

    Training, trading, and data-download scripts should call this
    before creating API clients.
    """

    errors: list[str] = []

    if require_credentials:
        if not ALPACA_API_KEY:
            errors.append(
                "ALPACA_API_KEY environment variable is missing."
            )

        if not ALPACA_SECRET_KEY:
            errors.append(
                "ALPACA_SECRET_KEY environment variable is missing."
            )

    if not 0.0 < MIN_BUY_CONFIDENCE < 1.0:
        errors.append(
            "MIN_BUY_CONFIDENCE must be between 0 and 1."
        )

    if not 0.0 < MAX_POSITION_PERCENT <= 1.0:
        errors.append(
            "MAX_POSITION_PERCENT must be greater than 0 "
            "and no more than 1."
        )

    if not 0.0 < STOP_LOSS_PERCENT < 1.0:
        errors.append(
            "STOP_LOSS_PERCENT must be between 0 and 1."
        )

    if not 0.0 < TAKE_PROFIT_PERCENT < 1.0:
        errors.append(
            "TAKE_PROFIT_PERCENT must be between 0 and 1."
        )

    if RSI_MINIMUM >= RSI_MAXIMUM:
        errors.append(
            "RSI_MINIMUM must be below RSI_MAXIMUM."
        )

    if HOLD_DAYS < 1:
        errors.append(
            "HOLD_DAYS must be at least 1."
        )

    if MAX_HOLD_DAYS < HOLD_DAYS:
        errors.append(
            "MAX_HOLD_DAYS must be greater than or equal "
            "to HOLD_DAYS."
        )

    if errors:
        formatted_errors = "\n- ".join(errors)

        raise ValueError(
            "Invalid bot configuration:\n- "
            + formatted_errors
        )


def create_project_directories() -> None:
    """
    Create folders used by models, reports, and local logs.
    """

    for directory in (
        MODELS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
