from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from research_tournament_db import load_experiments

REPORT=Path("reports/ut_championship_summary.json")
DIVISIONS={
 "Trend":{"UT Original","UT + EMA50","UT + EMA100","UT + EMA200","UT + Supertrend"},
 "Momentum":{"UT + RSI","UT + MACD","UT + ADX","UT + Stochastic"},
 "Volume":{"UT + Relative Volume","UT + OBV","UT + Volume Spike"},
 "Risk":{"UT + ATR Exit","UT + Chandelier Exit","UT + Dynamic Trail"},
}

def division(name):
    return next((d for d,names in DIVISIONS.items() if name in names),"Unassigned")

def build_summary():
    grouped=defaultdict(list)
    for r in load_experiments():
        if "UT" in r["strategy_variant"]:
            grouped[division(r["strategy_variant"])].append(r)
    reports={}
    overall=[]
    for d,items in grouped.items():
        items.sort(key=lambda x:(x["score"],x["net_profit_percent"]),reverse=True)
        for i,x in enumerate(items,1): x["division_rank"]=i
        reports[d]={"competitor_count":len(items),"leader":items[0] if items else None,"competitors":items}
        overall.extend(items)
    overall.sort(key=lambda x:(x["score"],x["net_profit_percent"]),reverse=True)
    for i,x in enumerate(overall,1): x["overall_rank"]=i
    return {"division_reports":reports,"overall_champion":overall[0] if overall else None,"overall_competitors":overall,"production_strategy_changed":False,"market_request_made":False,"order_submitted":False}

def run():
    report=build_summary()
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("UT Bot Championship summary")
    print(json.dumps(report,indent=2))
    print(f"JSON report: {REPORT}")
    print("No market request was made.")
    print("No order was submitted.")

if __name__=="__main__":
    run()
