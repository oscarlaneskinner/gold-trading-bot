from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("reports/ut_competition_results.json")
LEADERBOARD = Path("reports/ut_competition_leaderboard.json")

def load():
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"results":[]}

def score(r):
    return round(
        float(r["net_profit_percent"]) * 2
        + min(float(r["profit_factor"]), 3) * 10
        + float(r["percent_profitable"]) * 0.2
        + min(int(r["closed_trades"]), 200) * 0.03
        - abs(float(r["max_drawdown_percent"])) * 1.5,
        3,
    )

def build_leaderboard(results):
    rows=[]
    for item in results:
        row=dict(item)
        row["score"]=score(item)
        rows.append(row)
    rows.sort(key=lambda x:(x["score"],x["net_profit_percent"]), reverse=True)
    for i,row in enumerate(rows,1):
        row["rank"]=i
    return rows

def run():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",required=True)
    p.add_argument("--timeframe",required=True)
    p.add_argument("--date-range",required=True)
    p.add_argument("--net-profit-percent",type=float,required=True)
    p.add_argument("--max-drawdown-percent",type=float,required=True)
    p.add_argument("--percent-profitable",type=float,required=True)
    p.add_argument("--profit-factor",type=float,required=True)
    p.add_argument("--closed-trades",type=int,required=True)
    p.add_argument("--symbol",default="GLD")
    p.add_argument("--notes",default="")
    a=p.parse_args()

    payload=load()
    record={
        "variant":a.variant,
        "symbol":a.symbol,
        "timeframe":a.timeframe,
        "date_range":a.date_range,
        "net_profit_percent":a.net_profit_percent,
        "max_drawdown_percent":a.max_drawdown_percent,
        "percent_profitable":a.percent_profitable,
        "profit_factor":a.profit_factor,
        "closed_trades":a.closed_trades,
        "notes":a.notes,
        "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    payload["results"].append(record)
    RESULTS.parent.mkdir(parents=True,exist_ok=True)
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    LEADERBOARD.write_text(json.dumps({"leaderboard":build_leaderboard(payload["results"]),"order_submitted":False},indent=2),encoding="utf-8")
    print(json.dumps({"status":"saved","result":record,"order_submitted":False},indent=2))

if __name__=="__main__":
    run()
