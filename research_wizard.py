from __future__ import annotations
import json
from research_tournament_db import register_experiment, build_leaderboard
from research_command_center import run as refresh_reports

def ask(prompt, default=""):
    value=input(f"{prompt} [{default}]: ").strip()
    return value or str(default)

def ask_float(prompt, default=""):
    while True:
        try: return float(ask(prompt, default))
        except ValueError: print("Enter a valid number.")

def ask_int(prompt, default=""):
    while True:
        try: return int(ask(prompt, default))
        except ValueError: print("Enter a whole number.")

def run():
    print("UT BOT CHAMPIONSHIP — EXPERIMENT WIZARD")
    payload={
      "asset":ask("Asset","GLD"),
      "venue":ask("Venue","NYSE Arca"),
      "strategy_name":ask("Strategy name","UT Bot Championship"),
      "strategy_variant":ask("Competitor","UT Original"),
      "timeframe":ask("Timeframe","1D"),
      "date_start":ask("Start date","2004-11-18"),
      "date_end":ask("End date","2026-07-15"),
      "starting_capital":ask_float("Starting capital",100000),
      "commission_percent":ask_float("Commission percent",0.01),
      "slippage_units":ask_float("Slippage units",1),
      "position_percent":ask_float("Position size percent",15),
      "net_profit_amount":ask_float("Net profit amount"),
      "net_profit_percent":ask_float("Net profit percent"),
      "max_drawdown_amount":ask_float("Max drawdown amount"),
      "max_drawdown_percent":ask_float("Max drawdown percent"),
      "profitable_trades_percent":ask_float("Profitable trades percent"),
      "profit_factor":ask_float("Profit factor"),
      "closed_trades":ask_int("Closed trades"),
      "market_regime":ask("Market regime",""),
      "source_platform":ask("Source platform","TradingView"),
      "strategy_version":ask("Strategy version","3.0"),
      "notes":ask("Notes",""),
    }
    print(json.dumps(payload,indent=2))
    if ask("Save? yes/no","yes").lower() not in {"yes","y"}:
        print("Not saved."); return
    record=register_experiment(payload)
    board=build_leaderboard(asset=record["asset"],timeframe=record["timeframe"],minimum_trades=30)
    refresh_reports()
    rank=next((x["rank"] for x in board["leaderboard"] if x["experiment_code"]==record["experiment_code"]),None)
    print(json.dumps({"experiment_code":record["experiment_code"],"score":record["score"],"rank":rank,"reports_refreshed":True,"order_submitted":False},indent=2))

if __name__=="__main__":
    run()
