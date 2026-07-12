"""Central configuration for the Gold AI Trading Bot v3."""

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
TRADE_LOG_PATH = LOGS_DIR / "trades.csv"
DECISION_LOG_PATH = LOGS_DIR / "decisions.csv"
MODEL_REPORT_PATH = REPORTS_DIR / "model_metrics.json"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_metrics.json"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING = True

SYMBOL = "GLD"
DATA_LOOKBACK_DAYS = 500
TRAINING_LOOKBACK_DAYS = 7500

# --- VALIDATED VALUES (from today's walk-forward testing) ---
# The model predicts direction 20 trading days out, and is held for up to 20
# trading days. Testing hold periods of 3/5/10/20 days showed 20 performed
# best and most consistently across 5 independent time periods.
HOLD_DAYS = 20
MAX_HOLD_DAYS = 20

# Confidence threshold: testing showed raising this (e.g. to 0.70) consistently
# HURT performance across all 5 periods tested (dropped avg return from ~119%
# to ~9%) because it left too few, not-obviously-better trades. Keep disabled
# (0.50 = take any "up" prediction, matching what was actually tested).
MIN_BUY_CONFIDENCE = 0.50
BEARISH_EXIT_PROBABILITY = 0.40  # untested feature (model-triggered early exit) - see note below

# Trend/RSI/volatility filters: each tested individually today and each one
# REDUCED performance vs. the unfiltered model (trend filter alone: 119.1% ->
# 91.4%). Stacked together (as in v3), they produced ZERO trades across all 5
# periods. Disabled here to match the validated, tested configuration.
USE_TREND_FILTER = False
USE_RSI_FILTER = False
USE_VOLATILITY_FILTER = False
RSI_MINIMUM = 45.0
RSI_MAXIMUM = 70.0
MAX_ATR_PERCENT = 0.035

MAX_POSITION_PERCENT = 0.10
DEFAULT_TRADE_AMOUNT = 1000.0
MIN_ORDER_AMOUNT = 10.0
MINIMUM_ACCOUNT_EQUITY = 100.0

# Stop-loss / take-profit: 10% / 20% tested best among several combinations,
# and was consistent across all 5 walk-forward periods (avg 185.0% vs 154.7%
# baseline with no stops). Narrower stops (3-9%) cut off good trades early;
# this was confirmed by comparing exit-reason counts (narrow stops triggered
# on ~30-50% of trades; the 10% stop triggered on <5%).
STOP_LOSS_PERCENT = 0.10
TAKE_PROFIT_PERCENT = 0.20

# Trailing stop: tested with mixed, inconsistent results (helped in 2 of 5
# periods, hurt in 2, flat in 1) - meaningfully weaker evidence than the
# fixed stop/target combo above. Disabled to match the version actually
# validated and deployed.
ENABLE_TRAILING_STOP = False
TRAILING_STOP_PERCENT = 0.04
TRAILING_ACTIVATION_PERCENT = 0.03

TEST_FRACTION = 0.20
RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 200      # matches tested model (v3's 500 estimators is untested - fine to
                                    # experiment with, but re-validate via walk_forward_backtest.py first)
RANDOM_FOREST_MAX_DEPTH = 5         # matches tested model (v3's max_depth=7 is untested)
RANDOM_FOREST_MIN_SAMPLES_LEAF = 20 # matches tested model (v3's min_samples_leaf=5 is untested,
                                    # and a smaller leaf size generally increases overfitting risk)

MIN_TEST_ROWS = 50
MIN_MODEL_ACCURACY = 0.52
MIN_MODEL_ROC_AUC = 0.52
MAX_ALLOWED_ACCURACY_DROP = 0.005
MAX_ALLOWED_AUC_DROP = 0.005

BACKTEST_INITIAL_CAPITAL = 10_000.0
BACKTEST_SLIPPAGE_PERCENT = 0.0005
BACKTEST_POSITION_PERCENT = MAX_POSITION_PERCENT

def create_project_directories() -> None:
    for directory in (MODELS_DIR, LOGS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

def validate_configuration(require_credentials: bool = True) -> None:
    errors: list[str] = []
    if require_credentials:
        if not ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY is missing.")
        if not ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY is missing.")
    if not 0 < MIN_BUY_CONFIDENCE < 1:
        errors.append("MIN_BUY_CONFIDENCE must be between 0 and 1.")
    if not 0 < MAX_POSITION_PERCENT <= 1:
        errors.append("MAX_POSITION_PERCENT must be between 0 and 1.")
    if RSI_MINIMUM >= RSI_MAXIMUM:
        errors.append("RSI_MINIMUM must be below RSI_MAXIMUM.")
    if MAX_HOLD_DAYS < HOLD_DAYS:
        errors.append("MAX_HOLD_DAYS must be at least HOLD_DAYS.")
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
