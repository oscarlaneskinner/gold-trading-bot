from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/multi_market_arena_v2.json"); ap.add_argument("--output-dir",default="data")
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    key=os.getenv("ALPACA_API_KEY","").strip(); secret=os.getenv("ALPACA_SECRET_KEY","").strip()
    if not key or not secret: raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY are required.")
    client=StockHistoricalDataClient(key,secret); outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True); saved={}
    for symbol in config["symbols"]:
        req=StockBarsRequest(symbol_or_symbols=[symbol],timeframe=TimeFrame.Day,start=datetime.fromisoformat(config["start_date"]),end=datetime.now(timezone.utc))
        bars=client.get_stock_bars(req).df
        if bars.empty: saved[symbol]=0; continue
        bars=bars.reset_index().rename(columns={"timestamp":"date"})
        cols=[c for c in ["date","open","high","low","close","volume"] if c in bars.columns]
        dest=outdir/f"{symbol}_1D.csv"; bars[cols].to_csv(dest,index=False); saved[symbol]=len(bars)
        print(f"Saved {len(bars)} bars for {symbol} to {dest}.")
    print(json.dumps({"symbols_requested":config["symbols"],"bars_saved":saved,"market_data_request_made":True,"order_submitted":False},indent=2))
    print("Historical market-data requests were made."); print("No order was submitted.")
if __name__=="__main__": main()
