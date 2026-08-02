from __future__ import annotations
import argparse, itertools, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class SymbolResult:
    symbol: str
    train_return_percent: float
    test_return_percent: float
    test_drawdown_percent: float
    test_profit_factor: float
    test_win_rate: float
    test_trade_count: int
    passed: bool

@dataclass
class CandidateResult:
    candidate_id: str
    filter_set: str
    sensitivity: float
    atr_period: int
    max_bars_held: int
    risk_profile: str
    stop_loss_percent: float
    take_profit_percent: float
    symbols_tested: int
    symbols_passed: int
    median_test_return_percent: float
    mean_test_return_percent: float
    worst_test_return_percent: float
    median_drawdown_percent: float
    worst_drawdown_percent: float
    median_profit_factor: float
    median_trade_count: float
    consistency_percent: float
    score: float
    status: str
    per_symbol: list[dict[str, Any]]

def cfg(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    rename={}
    for c in df.columns:
        k=c.strip().lower()
        if k in {"time","date","datetime","timestamp"}: rename[c]="date"
        elif k in {"open","high","low","close","volume"}: rename[c]=k
    df=df.rename(columns=rename)
    missing={"open","high","low","close"}-set(df.columns)
    if missing: raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if "volume" not in df: df["volume"]=0.0
    for c in ["open","high","low","close","volume"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)

def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1/n, adjust=False).mean()

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy(); c=x.close; h=x.high; l=x.low; v=x.volume
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x["tr"]=tr; x["ema200"]=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0)
    rs=rma(g,14)/rma(loss,14).replace(0,np.nan)
    x["rsi14"]=100-100/(1+rs)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    x["macd"]=e12-e26; x["macd_signal"]=x.macd.ewm(span=9,adjust=False).mean()
    up=h.diff(); down=-l.diff()
    pdm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=x.index)
    mdm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=x.index)
    atr14=rma(tr,14).replace(0,np.nan)
    pdi=100*rma(pdm,14)/atr14; mdi=100*rma(mdm,14)/atr14
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["adx"]=rma(dx,14); x["pdi"]=pdi; x["mdi"]=mdi
    x["vma20"]=v.rolling(20).mean(); x["relvol"]=v/x.vma20.replace(0,np.nan)
    x["linreg50"]=c.rolling(50).apply(lambda z: np.polyval(np.polyfit(np.arange(len(z)),z,1),len(z)-1),raw=True)
    x["linreg50_prev"]=x.linreg50.shift(1)
    so=x.open.ewm(span=10,adjust=False).mean(); sh=h.ewm(span=10,adjust=False).mean()
    sl=l.ewm(span=10,adjust=False).mean(); sc=c.ewm(span=10,adjust=False).mean()
    x["sha_close"]=(so+sh+sl+sc)/4
    sha_open=pd.Series(index=x.index,dtype=float)
    for i in x.index:
        sha_open.iloc[i]=(so.iloc[i]+sc.iloc[i])/2 if i==0 else (sha_open.iloc[i-1]+x.sha_close.iloc[i-1])/2
    x["sha_open"]=sha_open
    frs=rma(g,5)/rma(loss,5).replace(0,np.nan); frsi=100-100/(1+frs)
    em=100*(c.ewm(span=5,adjust=False).mean()-c.ewm(span=20,adjust=False).mean())/c.replace(0,np.nan)
    x["bx"]=(frsi-50).ewm(span=5,adjust=False).mean()+em
    return x

def ut(df: pd.DataFrame, sensitivity: float, atr_period: int):
    atr=rma(df.tr,atr_period); stop=np.full(len(df),np.nan); c=df.close
    for i in range(len(df)):
        src=float(c.iloc[i]); dist=float(atr.iloc[i])*sensitivity if np.isfinite(atr.iloc[i]) else 0
        ps=src if i==0 or not np.isfinite(stop[i-1]) else stop[i-1]
        prev=src if i==0 else float(c.iloc[i-1])
        if src>ps and prev>ps: stop[i]=max(ps,src-dist)
        elif src<ps and prev<ps: stop[i]=min(ps,src+dist)
        elif src>ps: stop[i]=src-dist
        else: stop[i]=src+dist
    s=pd.Series(stop,index=df.index)
    return ((c>s)&(c.shift(1)<=s.shift(1))).fillna(False), ((c<s)&(c.shift(1)>=s.shift(1))).fillna(False)

def mask(df: pd.DataFrame, name: str) -> pd.Series:
    m={
        "Original":pd.Series(True,index=df.index),
        "EMA200":df.close>df.ema200,
        "RSI":(df.rsi14>50)&(df.rsi14<75),
        "MACD":df.macd>df.macd_signal,
        "ADX":(df.adx>20)&(df.pdi>df.mdi),
        "RelativeVolume":df.relvol>1,
        "LinearRegression":(df.close>df.linreg50)&(df.linreg50>df.linreg50_prev),
        "SmoothedHeikenAshi":(df.sha_close>df.sha_open)&(df.sha_close>df.sha_close.shift(1)),
        "BXStyle":(df.bx>0)&(df.bx>df.bx.shift(1)),
    }
    out=pd.Series(True,index=df.index)
    for part in name.split("+"): out &= m[part]
    return out.fillna(False)

def simulate(df, entries, exits, filt, config, hold, stop_pct, target_pct):
    initial=float(config["starting_capital"]); cash=initial; qty=0.0; ep=0.0; ei=-1
    trades=[]; curve=[]; pf=float(config["position_percent"])/100
    fee_rate=float(config["commission_percent"])/100; slip=float(config["slippage_dollars"])
    for i,row in df.iterrows():
        close=float(row.close)
        if qty==0 and bool(entries.iloc[i]) and bool(filt.iloc[i]):
            fill=close+slip; notional=cash*pf; qty=notional/fill
            cash-=notional+notional*fee_rate; ep=fill; ei=i
        elif qty>0:
            sp=ep*(1-stop_pct/100); tp=ep*(1+target_pct/100); xp=None
            if float(row.low)<=sp: xp=sp-slip
            elif float(row.high)>=tp: xp=tp-slip
            elif bool(exits.iloc[i]) or i-ei>=hold: xp=close-slip
            if xp is not None:
                proceeds=qty*xp; fee=proceeds*fee_rate; trades.append(proceeds-fee-qty*ep)
                cash+=proceeds-fee; qty=0; ep=0; ei=-1
        curve.append(cash+qty*close)
    if qty>0:
        proceeds=qty*(float(df.iloc[-1].close)-slip); fee=proceeds*fee_rate
        trades.append(proceeds-fee-qty*ep); cash+=proceeds-fee; curve[-1]=cash
    ret=100*(cash-initial)/initial
    equity=np.asarray(curve,dtype=float); peaks=np.maximum.accumulate(equity) if len(equity) else np.array([initial])
    dd=peaks-equity if len(equity) else np.array([0.0]); idx=int(dd.argmax())
    ddp=100*float(dd.max())/peaks[idx] if peaks[idx] else 0
    wins=[z for z in trades if z>0]; losses=[z for z in trades if z<0]
    gp=sum(wins); gl=abs(sum(losses)); prof=gp/gl if gl>0 else (99.0 if gp>0 else 0.0)
    wr=100*len(wins)/len(trades) if trades else 0
    return {"return_percent":ret,"drawdown_percent":ddp,"profit_factor":prof,"win_rate":wr,"trade_count":len(trades)}

def load_data(data_dir: Path, symbols: list[str]):
    out={}
    for s in symbols:
        p=data_dir/f"{s}_1D.csv"
        if p.exists(): out[s]=prepare(normalize(pd.read_csv(p)))
    return out

def aggregate(cid, fset, sens, atr, hold, risk, datasets, config):
    rows=[]
    for symbol,full in datasets.items():
        split=int(len(full)*float(config["train_fraction"]))
        train=full.iloc[:split].reset_index(drop=True); test=full.iloc[split:].reset_index(drop=True)
        tb,ts=ut(train,sens,atr); vb,vs=ut(test,sens,atr)
        tr=simulate(train,tb,ts,mask(train,fset),config,hold,float(risk["stop_loss_percent"]),float(risk["take_profit_percent"]))
        te=simulate(test,vb,vs,mask(test,fset),config,hold,float(risk["stop_loss_percent"]),float(risk["take_profit_percent"]))
        passed=te["trade_count"]>=int(config["minimum_trades_per_symbol"]) and te["return_percent"]>0 and te["profit_factor"]>1.05
        rows.append(SymbolResult(symbol,round(tr["return_percent"],4),round(te["return_percent"],4),round(te["drawdown_percent"],4),round(te["profit_factor"],4),round(te["win_rate"],4),te["trade_count"],passed))
    rets=np.array([r.test_return_percent for r in rows]); dds=np.array([r.test_drawdown_percent for r in rows])
    pfs=np.array([min(r.test_profit_factor,5.0) for r in rows]); tcs=np.array([r.test_trade_count for r in rows])
    passed=sum(r.passed for r in rows); consistency=100*passed/len(rows) if rows else 0
    medr=float(np.median(rets)); meanr=float(np.mean(rets)); worstr=float(np.min(rets))
    medd=float(np.median(dds)); worstd=float(np.max(dds)); medpf=float(np.median(pfs)); medtc=float(np.median(tcs))
    score=medr*3+meanr*1.5+medpf*8+consistency*.2+max(worstr,-10)*.75-medd*1.5-worstd*.5
    status="MULTI_MARKET_FINALIST" if passed>=int(config["minimum_symbols_passed"]) and medr>0 and medpf>1.05 else "REJECT"
    return CandidateResult(cid,fset,sens,atr,hold,str(risk["name"]),float(risk["stop_loss_percent"]),float(risk["take_profit_percent"]),len(rows),passed,round(medr,4),round(meanr,4),round(worstr,4),round(medd,4),round(worstd,4),round(medpf,4),round(medtc,2),round(consistency,2),round(score,4),status,[asdict(r) for r in rows])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data"); ap.add_argument("--config",default="config/multi_market_arena_v2.json")
    args=ap.parse_args(); config=cfg(Path(args.config)); datasets=load_data(Path(args.data_dir),config["symbols"])
    if not datasets: raise SystemExit("No symbol CSV files were found.")
    results=[]; counter=1
    combos=itertools.product(config["filter_sets"],config["ut_sensitivities"],config["atr_periods"],config["max_bars_held"],config["risk_profiles"])
    for fset,sens,atr,hold,risk in combos:
        results.append(aggregate(f"MM-{counter:04d}",fset,float(sens),int(atr),int(hold),risk,datasets,config)); counter+=1
    results.sort(key=lambda z:(z.status=="MULTI_MARKET_FINALIST",z.score),reverse=True)
    board=[{"rank":i,**asdict(r)} for i,r in enumerate(results,1)]
    top=[r for r in board if r["status"]=="MULTI_MARKET_FINALIST"][:int(config["top_n_candidates"])]
    out={"symbols_requested":config["symbols"],"symbols_loaded":sorted(datasets),"candidate_count":len(board),"finalist_count":sum(r["status"]=="MULTI_MARKET_FINALIST" for r in board),"top_finalists":top,"leaderboard":board,"tradingview_validation_required":True,"production_strategy_changed":False,"market_request_made":False,"order_submitted":False}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/multi_market_strategy_arena_v2.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    pd.DataFrame(board).drop(columns=["per_symbol"]).to_csv("reports/multi_market_strategy_arena_v2.csv",index=False)
    lines=["UT MULTI-MARKET STRATEGY ARENA V2","="*38,f"Symbols loaded: {', '.join(sorted(datasets))}",f"Candidates tested: {len(board)}",f"Finalists: {out['finalist_count']}","","TOP FINALISTS"]
    for r in top[:10]:
        lines.append(f"{r['rank']}. {r['candidate_id']} | {r['filter_set']} | sens={r['sensitivity']} atr={r['atr_period']} hold={r['max_bars_held']} | {r['risk_profile']} | score={r['score']} | median return={r['median_test_return_percent']}% | median PF={r['median_profit_factor']} | consistency={r['consistency_percent']}%")
    Path("reports/multi_market_strategy_arena_v2_summary.txt").write_text("\n".join(lines),encoding="utf-8")
    print("UT Multi-Market Strategy Arena v2")
    print(json.dumps({"symbols_loaded":out["symbols_loaded"],"candidate_count":out["candidate_count"],"finalist_count":out["finalist_count"],"top_finalists":top[:10],"production_strategy_changed":False,"market_request_made":False,"order_submitted":False},indent=2))
    print("No market request was made."); print("No order was submitted.")

if __name__=="__main__": main()
