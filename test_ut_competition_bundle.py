from pathlib import Path
import json
from ut_competition_results import build_leaderboard

required=["tradingview/gld_ut_competition_lab.pine","ut_competition_results.py","tradingview/UT-COMPETITION-GUIDE.md"]
missing=[f for f in required if not Path(f).exists()]
pine=Path("tradingview/gld_ut_competition_lab.pine").read_text(encoding="utf-8")
sample=[
{"variant":"UT Original","net_profit_percent":8,"max_drawdown_percent":5,"percent_profitable":48,"profit_factor":1.25,"closed_trades":40},
{"variant":"UT + EMA200","net_profit_percent":10,"max_drawdown_percent":4,"percent_profitable":55,"profit_factor":1.5,"closed_trades":35},
]
leaderboard=build_leaderboard(sample)
checks={
"pine_version_6":"//@version=6" in pine,
"ut_logic_present":"atrTrailingStop" in pine,
"variant_selector_present":"UT + EMA200" in pine,
"leaderboard_ranked":leaderboard[0]["variant"]=="UT + EMA200",
"no_alpaca_credentials":"ALPACA_API_KEY" not in pine,
}
out={"status":"passed" if not missing and all(checks.values()) else "failed","missing_files":missing,"checks":checks,"top_sample_variant":leaderboard[0]["variant"],"market_request_made":False,"order_submitted":False}
print("GLD UT competition bundle test")
print(json.dumps(out,indent=2))
print("No market request was made.")
print("No order was submitted.")
if out["status"]!="passed":
    raise SystemExit("UT competition validation failed.")
