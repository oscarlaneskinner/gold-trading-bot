from __future__ import annotations
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
with tempfile.TemporaryDirectory() as td:
 d=Path(td)/'data'; d.mkdir()
 for off,symbol in enumerate(['GLD','SPY','QQQ','IWM','SLV']):
  n=900; x=np.arange(n); close=100+.025*x+4*np.sin(x/(14+off))+1.2*np.sin(x/5)
  pd.DataFrame({'date':pd.date_range('2020-01-01',periods=n,freq='D'),'open':close-.2,'high':close+1,'low':close-1,'close':close,'volume':1_000_000+100_000*np.sin(x/7)}).to_csv(d/f'{symbol}_1D.csv',index=False)
 p=subprocess.run([sys.executable,'arena_v3.py','--data-dir',str(d),'--limit','120'],capture_output=True,text=True)
 rp=Path('reports/arena_v3_results.json'); result=json.loads(rp.read_text()) if rp.exists() else {}
 out={'status':'passed' if p.returncode==0 else 'failed','candidate_count':result.get('candidate_count',0),'json_created':rp.exists(),'leaderboard_created':Path('reports/arena_v3_leaderboard.csv').exists(),'tradingview_export_created':Path('reports/arena_v3_tradingview_finalists.csv').exists(),'heatmap_directory_created':Path('reports/heatmaps').exists(),'market_request_made':False,'order_submitted':False}
 print('UT Bot Championship Arena v3 test'); print(json.dumps(out,indent=2)); print('No market request was made.'); print('No order was submitted.')
 if out['status']!='passed' or out['candidate_count']!=120:
  print(p.stdout); print(p.stderr); raise SystemExit('Arena v3 test failed.')
