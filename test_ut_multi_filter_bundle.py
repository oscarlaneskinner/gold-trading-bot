"""Offline validation for the UT Multi-Filter Championship bundle."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED = [
    "tradingview/ut_multi_filter_championship.pine",
    "multi_filter_research_wizard.py",
    "ut_multi_filter_manager.py",
    "docs/UT-MULTI-FILTER-CHAMPIONSHIP-GUIDE.md",
]

missing = [name for name in REQUIRED if not Path(name).exists()]
pine = Path(REQUIRED[0]).read_text(encoding="utf-8")

expected_competitors = [
    "UT Original",
    "UT + EMA200 + RSI",
    "UT + EMA200 + MACD",
    "UT + EMA200 + Supertrend",
    "UT + RSI + MACD",
    "UT + EMA200 + RSI + MACD",
    "UT + EMA200 + RSI + MACD + Relative Volume",
]

checks = {
    "pine_v6": "//@version=6" in pine,
    "twenty_competitors": all(name in pine for name in expected_competitors),
    "wizard_present": Path("multi_filter_research_wizard.py").exists(),
    "manager_present": Path("ut_multi_filter_manager.py").exists(),
    "no_alpaca_credentials": "ALPACA_API_KEY" not in pine,
}

output = {
    "status": "passed" if not missing and all(checks.values()) else "failed",
    "missing_files": missing,
    "checks": checks,
    "market_request_made": False,
    "order_submitted": False,
}

print("UT Multi-Filter Championship bundle test")
print(json.dumps(output, indent=2))
print("No market request was made.")
print("No order was submitted.")

if output["status"] != "passed":
    raise SystemExit("Multi-filter championship bundle test failed.")
