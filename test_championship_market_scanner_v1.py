from __future__ import annotations
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
with tempfile.TemporaryDirectory() as td:
    d=Path(td)/'scanner'; d.mkdir(); symbols=['AAPL','MSFT','NVDA','AMZN','TSLA','SPY','QQQ','GLD']
    for off,s in enumerate(symbols):
        n=320; x=np.arange(n); direction=1 if off%2==0 else -1; close=100+direction*.06*x+4*np.sin(x/(15+off)); pd.DataFrame({'date':pd.date_range('2024-01-01',periods=n,freq='D'),'open':close-.3,'high':close+1.2,'low':close-1.2,'close':close,'volume':1_500_000+300_000*np.sin(x/8)}).to_csv(d/f'{s}_1D.csv',index=False)
    p=subprocess.run([sys.executable,'championship_market_scanner_v1.py','--data-dir',str(d)],capture_output=True,text=True); rp=Path('reports/scanner/championship_scanner_v1.json'); r=json.loads(rp.read_text()) if rp.exists() else {}; out={'status':'passed' if p.returncode==0 else 'failed','symbols_loaded':r.get('symbols_loaded',[]),'long_count':len(r.get('top_longs',[])),'short_count':len(r.get('top_shorts',[])),'json_report_created':rp.exists(),'long_csv_created':Path('reports/scanner/top_longs.csv').exists(),'short_csv_created':Path('reports/scanner/top_shorts.csv').exists(),'summary_created':Path('reports/scanner/championship_scanner_v1_summary.txt').exists(),'market_request_made':False,'order_submitted':False}; print('Championship Market Scanner v1 test'); print(json.dumps(out,indent=2)); print('No market request was made.'); print('No order was submitted.')
    if out['status']!='passed' or out['long_count']==0 or out['short_count']==0: print(p.stdout); print(p.stderr); raise SystemExit('Scanner test failed.')
