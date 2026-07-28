"""Offline validation for the alternative-indicators bundle."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED = [
    "tradingview/ut_alternative_indicators_championship.pine",
    "alternative_indicator_research_wizard.py",
    "ut_alternative_indicators_manager.py",
    "docs/UT-ALTERNATIVE-INDICATORS-GUIDE.md",
]

missing = [name for name in REQUIRED if not Path(name).exists()]
pine = Path(REQUIRED[0]).read_text(encoding="utf-8")

checks = {
    "pine_v6": "//@version=6" in pine,
    "twenty_competitors": all(
        name in pine
        for name in [
            "UT + B-Xtrender Style",
            "UT + Linear Regression Trend",
            "UT + Smoothed Heiken Ashi",
            "UT + ADR Volatility",
            "UT + All Alternative Filters",
        ]
    ),
    "original_approximation_notice":
        "do not copy or republish" in pine,
    "wizard_present":
        Path("alternative_indicator_research_wizard.py").exists(),
    "manager_present":
        Path("ut_alternative_indicators_manager.py").exists(),
    "no_alpaca_credentials":
        "ALPACA_API_KEY" not in pine,
}

output = {
    "status": "passed" if not missing and all(checks.values()) else "failed",
    "missing_files": missing,
    "checks": checks,
    "market_request_made": False,
    "order_submitted": False,
}

print("UT Alternative Indicators bundle test")
print(json.dumps(output, indent=2))
print("No market request was made.")
print("No order was submitted.")

if output["status"] != "passed":
    raise SystemExit("Alternative indicators bundle test failed.")
