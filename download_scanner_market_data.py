from __future__ import annotations
import argparse,json,os
from datetime import datetime,timedelta,timezone
from pathlib import Path
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/championship_scanner_v1.json'); ap.add_argument('--output-dir',default='data/scanner'); a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text()); key=os.getenv('ALPACA_API_KEY','').strip(); secret=os.getenv('ALPACA_SECRET_KEY','').strip()
    if not key or not secret: raise SystemExit('ALPACA_API_KEY and ALPACA_SECRET_KEY are required.')
    client=StockHistoricalDataClient(key,secret); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); end=datetime.now(timezone.utc); start=end-timedelta(days=int(cfg['lookback_days'])*2); saved={}; failed={}
    for s in cfg['symbols']:
        print(f'Downloading {s}...')
        try:
            req=StockBarsRequest(symbol_or_symbols=[s],timeframe=TimeFrame.Day,start=start,end=end,feed=DataFeed.IEX); bars=client.get_stock_bars(req).df
            if bars.empty: failed[s]='No data returned'; continue
            bars=bars.reset_index().rename(columns={'timestamp':'date'}); cols=[c for c in ['date','open','high','low','close','volume'] if c in bars.columns]; dest=out/f'{s}_1D.csv'; bars[cols].tail(int(cfg['lookback_days'])).to_csv(dest,index=False); saved[s]=min(len(bars),int(cfg['lookback_days'])); print(f'Saved {saved[s]} bars for {s}.')
        except Exception as e: failed[s]=f'{type(e).__name__}: {e}'; print(f'Failed {s}: {e}')
    print(json.dumps({'status':'completed' if saved else 'failed','feed':'IEX','symbols_saved':saved,'symbols_failed':failed,'market_data_request_made':True,'trading_client_created':False,'order_submitted':False},indent=2)); print('Historical market-data requests were made.'); print('No trading client was created.'); print('No order was submitted.')
    if not saved: raise SystemExit('No scanner data was downloaded.')
if __name__=='__main__': main()
