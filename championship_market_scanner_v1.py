from __future__ import annotations
import argparse, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class ScanResult:
    symbol:str; side:str; score:float; close:float; average_volume_20d:float; atr_14:float; atr_percent:float; rsi_14:float; distance_from_ema_200_percent:float; suggested_stop:float; suggested_target:float; maximum_position_percent:float; signal_reasons:list[str]

def load_config(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding='utf-8'))
def rma(s:pd.Series,n:int)->pd.Series: return s.ewm(alpha=1/n,adjust=False).mean()
def normalize(df:pd.DataFrame)->pd.DataFrame:
    ren={}
    for c in df.columns:
        k=c.strip().lower()
        if k in {'time','date','datetime','timestamp'}: ren[c]='date'
        elif k in {'open','high','low','close','volume'}: ren[c]=k
    df=df.rename(columns=ren)
    req={'open','high','low','close','volume'}
    miss=req-set(df.columns)
    if miss: raise ValueError(f'CSV missing columns: {sorted(miss)}')
    for c in req: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=list(req)).reset_index(drop=True)
def indicators(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy(); c=x.close; h=x.high; l=x.low; v=x.volume
    x['ema20']=c.ewm(span=20,adjust=False).mean(); x['ema50']=c.ewm(span=50,adjust=False).mean(); x['ema200']=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0); rs=rma(g,14)/rma(loss,14).replace(0,np.nan); x['rsi14']=100-100/(1+rs)
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); x['atr14']=rma(tr,14); x['atrp']=100*x.atr14/c.replace(0,np.nan)
    x['vma20']=v.rolling(20).mean(); x['relvol']=v/x.vma20.replace(0,np.nan); x['high20']=h.rolling(20).max().shift(1); x['low20']=l.rolling(20).min().shift(1)
    x['ret5']=c.pct_change(5)*100; x['ret20']=c.pct_change(20)*100; x['dist200']=100*(c-x.ema200)/x.ema200.replace(0,np.nan)
    return x
def clip(v:float)->float: return float(max(0,min(100,v)))
def score(symbol:str,df:pd.DataFrame,cfg:dict[str,Any],side:str)->ScanResult|None:
    if len(df)<220: return None
    x=indicators(df); r=x.iloc[-1]; close=float(r.close); av=float(r.vma20)
    if close<float(cfg['minimum_price']) or not np.isfinite(av) or av<float(cfg['minimum_average_volume']): return None
    e20,e50,e200=float(r.ema20),float(r.ema50),float(r.ema200); rsi=float(r.rsi14); atr=float(r.atr14); atrp=float(r.atrp); rv=float(r.relvol); ret5=float(r.ret5); ret20=float(r.ret20); h20=float(r.high20); l20=float(r.low20); dist=float(r.dist200)
    reasons=[]
    if side=='long':
        trend=clip(40+(20 if close>e200 else -20)+(20 if e20>e50 else -10)+(20 if e50>e200 else -10)); momentum=clip(50+ret5*5+ret20*1.5); breakout=clip(50+((close/h20)-1)*1000) if h20 else 0
        if close>e200: reasons.append('Price above EMA200')
        if e20>e50>e200: reasons.append('Bullish EMA alignment')
        if rsi>=55: reasons.append('Positive RSI momentum')
        if close>=h20: reasons.append('20-day breakout')
        stop=close-atr*float(cfg['risk_defaults']['stop_atr_multiple']); target=close+atr*float(cfg['risk_defaults']['target_atr_multiple'])
    else:
        trend=clip(40+(20 if close<e200 else -20)+(20 if e20<e50 else -10)+(20 if e50<e200 else -10)); momentum=clip(50-ret5*5-ret20*1.5); breakout=clip(50+((l20/close)-1)*1000) if close else 0
        if close<e200: reasons.append('Price below EMA200')
        if e20<e50<e200: reasons.append('Bearish EMA alignment')
        if rsi<=45: reasons.append('Negative RSI momentum')
        if close<=l20: reasons.append('20-day breakdown')
        stop=close+atr*float(cfg['risk_defaults']['stop_atr_multiple']); target=close-atr*float(cfg['risk_defaults']['target_atr_multiple'])
    if rv>=1.2: reasons.append('Above-average volume')
    relvol=clip(rv*50); volq=clip(100-abs(atrp-3)*20); total=.30*trend+.30*momentum+.20*breakout+.10*relvol+.10*volq
    return ScanResult(symbol,side,round(total,4),round(close,4),round(av,2),round(atr,4),round(atrp,4),round(rsi,4),round(dist,4),round(stop,4),round(max(.01,target),4),float(cfg['risk_defaults']['maximum_position_percent']),reasons)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data/scanner'); ap.add_argument('--config',default='config/championship_scanner_v1.json'); a=ap.parse_args(); cfg=load_config(Path(a.config)); d=Path(a.data_dir); longs=[]; shorts=[]; loaded=[]
    for s in cfg['symbols']:
        p=d/f'{s}_1D.csv'
        if not p.exists(): continue
        loaded.append(s); df=normalize(pd.read_csv(p)); l=score(s,df,cfg,'long'); sh=score(s,df,cfg,'short')
        if l: longs.append(l)
        if sh: shorts.append(sh)
    longs=sorted(longs,key=lambda z:z.score,reverse=True)[:int(cfg['top_long_count'])]; shorts=sorted(shorts,key=lambda z:z.score,reverse=True)[:int(cfg['top_short_count'])]
    outdir=Path('reports/scanner'); outdir.mkdir(parents=True,exist_ok=True); payload={'symbols_loaded':loaded,'top_longs':[asdict(z) for z in longs],'top_shorts':[asdict(z) for z in shorts],'production_strategy_changed':False,'market_request_made':False,'order_submitted':False}
    (outdir/'championship_scanner_v1.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); pd.DataFrame(payload['top_longs']).to_csv(outdir/'top_longs.csv',index=False); pd.DataFrame(payload['top_shorts']).to_csv(outdir/'top_shorts.csv',index=False)
    lines=['CHAMPIONSHIP MARKET SCANNER V1','='*34,'','TOP LONGS']+[f"{i}. {z.symbol} score={z.score} stop={z.suggested_stop} target={z.suggested_target}" for i,z in enumerate(longs,1)]+['','TOP SHORTS']+[f"{i}. {z.symbol} score={z.score} stop={z.suggested_stop} target={z.suggested_target}" for i,z in enumerate(shorts,1)]; (outdir/'championship_scanner_v1_summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('Championship Market Scanner v1'); print(json.dumps({'symbols_loaded':loaded,'long_candidate_count':len(longs),'short_candidate_count':len(shorts),'top_longs':[asdict(z) for z in longs[:10]],'top_shorts':[asdict(z) for z in shorts[:10]],'production_strategy_changed':False,'market_request_made':False,'order_submitted':False},indent=2)); print('No market request was made.'); print('No order was submitted.')
if __name__=='__main__': main()
