"""
Central configuration for Gold AI Trading Bot
"""

import os


# =====================
# ALPACA SETTINGS
# =====================

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

PAPER_TRADING = True


# =====================
# TRADING SETTINGS
# =====================

SYMBOL = "GLD"

DEFAULT_TRADE_AMOUNT = 1000

RISK_PER_TRADE = 0.02

MAX_POSITION_PERCENT = 0.10


# =====================
# AI SETTINGS
# =====================

MODEL_PATH = "models/model.pkl"

CONFIDENCE_THRESHOLD = 0.70


# =====================
# STRATEGY SETTINGS
# =====================

HOLD_DAYS = 5

USE_TREND_FILTER = True
USE_RSI_FILTER = True


# =====================
# RISK SETTINGS
# =====================

ATR_STOP_MULTIPLIER = 2

ATR_TARGET_MULTIPLIER = 3


# =====================
# LOGGING
# =====================

LOG_FILE = "logs/trades.csv"
