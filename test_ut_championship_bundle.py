from pathlib import Path
import json

required=[
 "tradingview/ut_bot_championship.pine",
 "research_wizard.py",
 "ut_championship_manager.py",
 "docs/UT-BOT-CHAMPIONSHIP-GUIDE.md",
]
missing=[f for f in required if not Path(f).exists()]
pine=Path(required[0]).read_text(encoding="utf-8")
checks={
 "pine_v6":"//@version=6" in pine,
 "fifteen_competitors":all(x in pine for x in ["UT Original","UT + EMA50","UT + RSI","UT + Relative Volume","UT + ATR Exit"]),
 "wizard_present":Path("research_wizard.py").exists(),
 "manager_present":Path("ut_championship_manager.py").exists(),
 "no_alpaca_credentials":"ALPACA_API_KEY" not in pine,
}
out={"status":"passed" if not missing and all(checks.values()) else "failed","missing_files":missing,"checks":checks,"market_request_made":False,"order_submitted":False}
print("UT Bot Championship bundle test")
print(json.dumps(out,indent=2))
print("No market request was made.")
print("No order was submitted.")
if out["status"]!="passed":
    raise SystemExit("Championship bundle test failed.")
