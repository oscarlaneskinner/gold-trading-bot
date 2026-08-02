from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
with tempfile.TemporaryDirectory() as td:
    d=Path(td)/"data"; d.mkdir()
    for off,symbol in enumerate(["GLD","SPY","QQQ","IWM"]):
        n=1200; x=np.arange(n); close=100+.02*x+5*np.sin(x/(16+off))+1.5*np.sin(x/5)
        pd.DataFrame({"date":pd.date_range("2020-01-01",periods=n,freq="D"),"open":close-.25,"high":close+1.1,"low":close-1.1,"close":close,"volume":1_000_000+100_000*np.sin(x/8)}).to_csv(d/f"{symbol}_1D.csv",index=False)
    p=subprocess.run([sys.executable,"multi_market_strategy_arena_v2.py","--data-dir",str(d)],capture_output=True,text=True)
    rp=Path("reports/multi_market_strategy_arena_v2.json"); report=json.loads(rp.read_text()) if rp.exists() else {}
    out={"status":"passed" if p.returncode==0 else "failed","symbols_loaded":report.get("symbols_loaded",[]),"candidate_count":report.get("candidate_count",0),"json_report_created":rp.exists(),"csv_report_created":Path("reports/multi_market_strategy_arena_v2.csv").exists(),"summary_report_created":Path("reports/multi_market_strategy_arena_v2_summary.txt").exists(),"market_request_made":False,"order_submitted":False}
    print("UT Multi-Market Strategy Arena v2 test"); print(json.dumps(out,indent=2)); print("No market request was made."); print("No order was submitted.")
    if out["status"]!="passed" or out["candidate_count"]<100:
        print(p.stdout); print(p.stderr); raise SystemExit("Multi-market arena test failed.")
