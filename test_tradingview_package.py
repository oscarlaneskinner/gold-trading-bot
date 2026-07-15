from pathlib import Path
import json

REQUIRED = [
    "tradingview/gld_research_strategy.pine",
    "tradingview/ALERT-MESSAGE.json",
    "tradingview/SETUP-GUIDE.md",
    "tradingview/TRIAL-CHECKLIST.md",
    "tradingview_signal_log.py",
]

missing = [f for f in REQUIRED if not Path(f).exists()]
pine = Path("tradingview/gld_research_strategy.pine").read_text(encoding="utf-8")
checks = {
    "pine_version_6": "//@version=6" in pine,
    "strategy_declared": "strategy(" in pine,
    "alert_json_present": '"source":"tradingview"' in pine,
    "no_alpaca_credentials": "ALPACA_API_KEY" not in pine,
}
output = {
    "status": "passed" if not missing and all(checks.values()) else "failed",
    "missing_files": missing,
    "checks": checks,
    "market_request_made": False,
    "order_submitted": False,
}
print("GLD TradingView Research Package 1 test")
print(json.dumps(output, indent=2))
print("No market request was made.")
print("No order was submitted.")
if output["status"] != "passed":
    raise SystemExit("TradingView package validation failed.")
