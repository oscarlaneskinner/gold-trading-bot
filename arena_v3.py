from __future__ import annotations
import argparse,itertools,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,dataclass
from pathlib import Path
from statistics import mean,median
from typing import Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Candidate:
    candidate_id:str; filter_set:str; sensitivity:float; atr_period:int; max_bars_held:int; stop_loss_percent:float; take_profit_percent:float

def load_config(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def normalize(df):
    rename={}
    for c in df.columns:
        k=c.strip().lower()
        if k in {'time','date','datetime','timestamp'}: rename[c]='date'
        elif k in {'open','high','low','close','volume'}: rename[c]=k
    df=df.rename(columns=rename)
    missing={'open','high','low','close'}-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if 'volume' not in df: df['volume']=0.0
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=['open','high','low','close']).reset_index(drop=True)
def prepare(df):
    x=df.copy(); c=x.close; h=x.high; l=x.low; v=x.volume; pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); x['tr']=tr; x['ema200']=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); lo=-d.clip(upper=0); rs=rma(g,14)/rma(lo,14).replace(0,np.nan); x['rsi14']=100-100/(1+rs)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x['macd']=e12-e26; x['macd_signal']=x.macd.ewm(span=9,adjust=False).mean()
    up=h.diff(); down=-l.diff(); pdm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=x.index); mdm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=x.index)
    atr14=rma(tr,14).replace(0,np.nan); pdi=100*rma(pdm,14)/atr14; mdi=100*rma(mdm,14)/atr14; dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x['adx']=rma(dx,14); x['plus_di']=pdi; x['minus_di']=mdi; x['volume_ma20']=v.rolling(20).mean(); x['relative_volume']=v/x.volume_ma20.replace(0,np.nan)
    x['linreg50']=c.rolling(50).apply(lambda z: np.polyval(np.polyfit(np.arange(len(z)),z,1),len(z)-1),raw=True); x['linreg50_prev']=x.linreg50.shift(1)
    so=x.open.ewm(span=10,adjust=False).mean(); sh=h.ewm(span=10,adjust=False).mean(); sl=l.ewm(span=10,adjust=False).mean(); sc=c.ewm(span=10,adjust=False).mean(); x['sha_close']=(so+sh+sl+sc)/4
    sha_open=pd.Series(index=x.index,dtype=float)
    for i in x.index: sha_open.iloc[i]=(so.iloc[i]+sc.iloc[i])/2 if i==0 else (sha_open.iloc[i-1]+x.sha_close.iloc[i-1])/2
    x['sha_open']=sha_open; frs=rma(g,5)/rma(lo,5).replace(0,np.nan); frsi=100-100/(1+frs); em=100*(c.ewm(span=5,adjust=False).mean()-c.ewm(span=20,adjust=False).mean())/c.replace(0,np.nan); x['bx_style']=(frsi-50).ewm(span=5,adjust=False).mean()+em
    return x
def mask(df,name):
    m={'Original':pd.Series(True,index=df.index),'EMA200':df.close>df.ema200,'RSI':(df.rsi14>50)&(df.rsi14<75),'MACD':df.macd>df.macd_signal,'ADX':(df.adx>20)&(df.plus_di>df.minus_di),'RelativeVolume':df.relative_volume>1,'LinearRegression':(df.close>df.linreg50)&(df.linreg50>df.linreg50_prev),'SmoothedHeikenAshi':(df.sha_close>df.sha_open)&(df.sha_close>df.sha_close.shift(1)),'BXStyle':(df.bx_style>0)&(df.bx_style>df.bx_style.shift(1))}
    out=pd.Series(True,index=df.index)
    for part in name.split('+'): out &= m[part]
    return out.fillna(False)
def ut(df,sens,atrp):
    atr=rma(df.tr,atrp); stop=np.full(len(df),np.nan); c=df.close
    for i in range(len(df)):
        src=float(c.iloc[i]); dist=float(atr.iloc[i])*sens if np.isfinite(atr.iloc[i]) else 0; ps=src if i==0 or not np.isfinite(stop[i-1]) else stop[i-1]; prev=src if i==0 else float(c.iloc[i-1])
        if src>ps and prev>ps: stop[i]=max(ps,src-dist)
        elif src<ps and prev<ps: stop[i]=min(ps,src+dist)
        elif src>ps: stop[i]=src-dist
        else: stop[i]=src+dist
    s=pd.Series(stop,index=df.index); return ((c>s)&(c.shift(1)<=s.shift(1))).fillna(False),((c<s)&(c.shift(1)>=s.shift(1))).fillna(False)
def simulate(df,entries,exits,filt,cfg,cand):
    initial=float(cfg['starting_capital']); cash=initial; qty=0.0; ep=0.0; ei=-1; trades=[]; curve=[]; frac=float(cfg['position_percent'])/100; fee=float(cfg['commission_percent'])/100; slip=float(cfg['slippage_dollars'])
    for i,row in df.iterrows():
        close=float(row.close)
        if qty==0 and bool(entries.iloc[i]) and bool(filt.iloc[i]):
            fill=close+slip; notional=cash*frac; qty=notional/fill; cash-=notional+notional*fee; ep=fill; ei=i
        elif qty>0:
            sp=ep*(1-cand.stop_loss_percent/100); tp=ep*(1+cand.take_profit_percent/100); xp=None
            if float(row.low)<=sp: xp=sp-slip
            elif float(row.high)>=tp: xp=tp-slip
            elif bool(exits.iloc[i]) or i-ei>=cand.max_bars_held: xp=close-slip
            if xp is not None:
                proceeds=qty*xp; f=proceeds*fee; trades.append(float(proceeds-f-qty*ep)); cash+=proceeds-f; qty=0; ep=0; ei=-1
        curve.append(cash+qty*close)
    if qty>0:
        xp=float(df.iloc[-1].close)-slip; proceeds=qty*xp; f=proceeds*fee; trades.append(float(proceeds-f-qty*ep)); cash+=proceeds-f; curve[-1]=cash
    eq=np.asarray(curve,dtype=float); peaks=np.maximum.accumulate(eq) if len(eq) else np.asarray([initial]); dd=peaks-eq if len(eq) else np.asarray([0.0]); idx=int(dd.argmax()); ddp=100*float(dd.max())/peaks[idx] if peaks[idx] else 0
    wins=[t for t in trades if t>0]; losses=[t for t in trades if t<0]; gp=sum(wins); gl=abs(sum(losses)); pf=gp/gl if gl>0 else (99.0 if gp>0 else 0.0)
    return {'return_percent':100*(cash-initial)/initial,'drawdown_percent':ddp,'profit_factor':pf,'win_rate':100*len(wins)/len(trades) if trades else 0.0,'trade_count':len(trades),'trades':trades}
def slices(length,folds):
    warm=max(250,length//3); rem=max(length-warm,1); size=max(rem//folds,1); out=[]; start=warm
    for f in range(folds):
        end=length if f==folds-1 else min(length,start+size)
        if end>start: out.append((start,end))
        start=end
    return out
def evaluate(cand,datasets,cfg):
    rows=[]; all_trades=[]
    for symbol,full in datasets.items():
        entries,exits=ut(full,cand.sensitivity,cand.atr_period); filt=mask(full,cand.filter_set)
        for fold,(start,end) in enumerate(slices(len(full),int(cfg['walk_forward_folds'])),1):
            data=full.iloc[start:end].reset_index(drop=True); r=simulate(data,entries.iloc[start:end].reset_index(drop=True),exits.iloc[start:end].reset_index(drop=True),filt.iloc[start:end].reset_index(drop=True),cfg,cand); all_trades.extend(r['trades'])
            passed=r['trade_count']>=int(cfg['minimum_trades_per_fold']) and r['return_percent']>0 and r['profit_factor']>1.05
            rows.append({'symbol':symbol,'fold':fold,'return_percent':round(r['return_percent'],4),'drawdown_percent':round(r['drawdown_percent'],4),'profit_factor':round(min(r['profit_factor'],10),4),'win_rate':round(r['win_rate'],4),'trade_count':r['trade_count'],'passed':passed})
    returns=[r['return_percent'] for r in rows]; dds=[r['drawdown_percent'] for r in rows]; pfs=[r['profit_factor'] for r in rows]; counts=[r['trade_count'] for r in rows]
    symbols_passed=len({r['symbol'] for r in rows if r['passed']}); consistency=100*sum(r['passed'] for r in rows)/len(rows) if rows else 0; medr=median(returns) if returns else 0; meanr=mean(returns) if returns else 0; worstr=min(returns) if returns else 0; medd=median(dds) if dds else 0; worstd=max(dds) if dds else 0; medpf=median(pfs) if pfs else 0; medtc=median(counts) if counts else 0
    score=medr*3+meanr*1.5+medpf*8+consistency*.2+max(worstr,-10)*.75-medd*1.5-worstd*.5
    status='WALK_FORWARD_FINALIST' if symbols_passed>=int(cfg['minimum_symbols_passed']) and medr>0 and medpf>1.05 else 'REJECT'
    return {**asdict(cand),'symbols_tested':len(datasets),'symbols_passed':symbols_passed,'fold_count':len(rows),'median_test_return_percent':round(medr,4),'mean_test_return_percent':round(meanr,4),'worst_test_return_percent':round(worstr,4),'median_drawdown_percent':round(medd,4),'worst_drawdown_percent':round(worstd,4),'median_profit_factor':round(medpf,4),'median_trade_count':round(medtc,2),'consistency_percent':round(consistency,2),'score':round(score,4),'status':status,'trade_returns':all_trades,'fold_results':rows}
def monte_carlo(trades,initial,runs,seed):
    if not trades: return {'mc_runs':runs,'median_final_equity':initial,'fifth_percentile_final_equity':initial,'median_max_drawdown_percent':0.0,'ninety_fifth_percentile_drawdown_percent':0.0,'probability_of_loss_percent':100.0}
    rng=np.random.default_rng(seed); finals=[]; dds=[]; arr=np.asarray(trades,float)
    for _ in range(runs):
        shuffled=rng.permutation(arr); eq=np.concatenate([[initial],initial+np.cumsum(shuffled)]); peaks=np.maximum.accumulate(eq); draw=100*(peaks-eq)/np.where(peaks==0,1,peaks); finals.append(float(eq[-1])); dds.append(float(draw.max()))
    return {'mc_runs':runs,'median_final_equity':round(float(np.median(finals)),2),'fifth_percentile_final_equity':round(float(np.percentile(finals,5)),2),'median_max_drawdown_percent':round(float(np.median(dds)),4),'ninety_fifth_percentile_drawdown_percent':round(float(np.percentile(dds,95)),4),'probability_of_loss_percent':round(100*float(np.mean(np.asarray(finals)<initial)),4)}
def build_candidates(cfg):
    g=cfg['parameter_grid']; out=[]
    for i,v in enumerate(itertools.product(g['filter_sets'],g['sensitivities'],g['atr_periods'],g['max_bars_held'],g['stop_loss_percent'],g['take_profit_percent']),1):
        f,s,a,h,sl,tp=v; out.append(Candidate(f'V3-{i:06d}',str(f),float(s),int(a),int(h),float(sl),float(tp)))
    return out
def load_data(path,symbols):
    out={}
    for s in symbols:
        p=Path(path)/f'{s}_1D.csv'
        if p.exists(): out[s]=prepare(normalize(pd.read_csv(p)))
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--config',default='config/arena_v3.json'); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args(); cfg=load_config(args.config); datasets=load_data(args.data_dir,cfg['symbols'])
    if not datasets: raise SystemExit('No symbol CSV files found.')
    candidates=build_candidates(cfg); candidates=candidates[:args.limit] if args.limit>0 else candidates; workers=int(cfg.get('max_workers',0)) or max(1,(os.cpu_count() or 2)-1); results=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(evaluate,c,datasets,cfg):c.candidate_id for c in candidates}
        done=0
        for fut in as_completed(futures):
            results.append(fut.result()); done+=1
            if done%100==0 or done==len(candidates): print(f'Completed {done}/{len(candidates)} candidates...')
    results.sort(key=lambda r:(r['status']=='WALK_FORWARD_FINALIST',r['score']),reverse=True); mc_count=min(int(cfg['top_candidates_for_monte_carlo']),len(results))
    for i,row in enumerate(results[:mc_count]): row['monte_carlo']=monte_carlo(row.pop('trade_returns'),float(cfg['starting_capital']),int(cfg['monte_carlo_runs']),1000+i)
    for row in results[mc_count:]: row.pop('trade_returns',None); row['monte_carlo']=None
    for rank,row in enumerate(results,1): row['rank']=rank
    finalists=[r for r in results if r['status']=='WALK_FORWARD_FINALIST']; top=finalists[:int(cfg['top_candidates_for_reports'])]; reports=Path('reports'); reports.mkdir(exist_ok=True)
    output={'symbols_loaded':sorted(datasets),'candidate_count':len(results),'finalist_count':len(finalists),'worker_count':workers,'walk_forward_folds':int(cfg['walk_forward_folds']),'monte_carlo_runs':int(cfg['monte_carlo_runs']),'top_finalists':top,'production_strategy_changed':False,'market_request_made':False,'order_submitted':False}
    (reports/'arena_v3_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    cols=['rank','candidate_id','filter_set','sensitivity','atr_period','max_bars_held','stop_loss_percent','take_profit_percent','symbols_passed','median_test_return_percent','mean_test_return_percent','worst_test_return_percent','median_drawdown_percent','worst_drawdown_percent','median_profit_factor','consistency_percent','score','status']
    frame=pd.DataFrame(results); frame[cols].to_csv(reports/'arena_v3_leaderboard.csv',index=False); pd.DataFrame(top)[cols].to_csv(reports/'arena_v3_top_100.csv',index=False)
    heat=reports/'heatmaps'; heat.mkdir(exist_ok=True)
    for metric in ['score','median_test_return_percent','median_profit_factor']: frame.pivot_table(index='sensitivity',columns='atr_period',values=metric,aggfunc='mean').to_csv(heat/f'heatmap_{metric}.csv')
    pd.DataFrame([{'rank':r['rank'],'candidate_id':r['candidate_id'],'competitor':r['filter_set'],'ut_sensitivity':r['sensitivity'],'ut_atr_period':r['atr_period'],'maximum_bars_held':r['max_bars_held'],'hard_stop_percent':r['stop_loss_percent'],'take_profit_percent':r['take_profit_percent'],'score':r['score']} for r in top]).to_csv(reports/'arena_v3_tradingview_finalists.csv',index=False)
    print('UT Bot Championship Arena v3'); print(json.dumps({'symbols_loaded':output['symbols_loaded'],'candidate_count':output['candidate_count'],'finalist_count':output['finalist_count'],'worker_count':workers,'top_finalists':top[:10],'production_strategy_changed':False,'market_request_made':False,'order_submitted':False},indent=2)); print('No market request was made.'); print('No order was submitted.')
if __name__=='__main__': main()
