"""Independent historical test of production volume screen. No emails."""
import os,json,time,traceback
from pathlib import Path
import pandas as pd
from pykrx import stock

OUT=Path("outputs_volume");OUT.mkdir(exist_ok=True)
DATES=["20250331","20250930","20260331","20260930"]
def call(fn,*args,**kwargs):
    for i in range(3):
        try:
            x=fn(*args,**kwargs)
            if x is None or x.empty: raise ValueError("empty response")
            time.sleep(.25)
            return x
        except Exception:
            if i==2: raise
            time.sleep(2*(i+1))
def market_day(date):
    parts=[]
    for m in ("KOSPI","KOSDAQ"):
        x=call(stock.get_market_ohlcv_by_ticker,date,market=m)
        x=x[["종가","거래량"]].copy()
        x["ticker"]=x.index.astype(str).str.zfill(6);x["market"]=m
        parts.append(x.reset_index(drop=True))
    return pd.concat(parts,ignore_index=True)
def market_cap(date):
    parts=[]
    for m in ("KOSPI","KOSDAQ"):
        x=call(stock.get_market_cap_by_ticker,date,market=m)
        x=x[["시가총액"]].copy()
        x["ticker"]=x.index.astype(str).str.zfill(6);x["market"]=m
        parts.append(x.reset_index(drop=True))
    return pd.concat(parts,ignore_index=True)
def run_date(anchor):
    # Obtain actual sessions from KRX index data, including holidays.
    from datetime import timedelta
    end=pd.Timestamp(anchor)
    start=end-timedelta(days=35)
    ix=call(stock.get_index_ohlcv_by_date,start.strftime("%Y%m%d"),anchor,"1001")
    days=[d.strftime("%Y%m%d") for d in ix.index if d<=end][-10:]
    if len(days)!=10: raise RuntimeError("fewer than ten sessions")
    frames=[]
    for d in days:
        x=market_day(d);x["date"]=d;frames.append(x)
    panel=pd.concat(frames,ignore_index=True)
    cap=market_cap(days[-1])
    out=[]
    for (ticker,m),g in panel.groupby(["ticker","market"]):
        if len(g)!=10 or g.date.nunique()!=10:continue
        g=g.sort_values("date")
        v=pd.to_numeric(g["거래량"],errors="coerce")
        p=pd.to_numeric(g["종가"],errors="coerce")
        if v.isna().any() or p.isna().any() or (p<=0).any() or v.iloc[:5].sum()<=0:continue
        mult=v.iloc[5:].sum()/v.iloc[:5].sum()
        change=(p.iloc[-1]/p.iloc[5]-1)*100
        if mult>=3 and -3<=change<=3:
            out.append(dict(ticker=ticker,market=m,signal_date=days[-1],multiple=mult,price_change_pct=change,signal_close=float(p.iloc[-1])))
    result=pd.DataFrame(out)
    if result.empty:return result,dict(anchor=anchor,signal_date=days[-1],matches=0)
    result=result.merge(cap,on=["ticker","market"],how="left")
    result=result[result["시가총액"].notna() & (result["시가총액"]<1_000_000_000_000)].copy()
    # Avoid hindsight: future prices used only after the signal is fixed.
    future_end=(pd.Timestamp(days[-1])+timedelta(days=55)).strftime("%Y%m%d")
    for j,row in result.iterrows():
        try:
            x=call(stock.get_market_ohlcv_by_date,days[-1],future_end,row.ticker)
            x=x.sort_index();x=x[x.index>pd.Timestamp(days[-1])]
            for h in (5,10,20):
                result.at[j,f"future_{h}"]=float(x.iloc[h-1]["종가"]/row.signal_close-1) if len(x)>=h else float("nan")
                result.at[j,f"next_close_{h}"]=float(x.iloc[h-1]["종가"]/x.iloc[0]["종가"]-1) if len(x)>=h else float("nan")
        except Exception as e:
            result.at[j,"forward_error"]=str(e)
    return result,dict(anchor=anchor,signal_date=days[-1],matches=len(result))
def main():
    diagnostic={"runs":[],"errors":{},"note":"4 monthly historical snapshots, not all trading days"}
    rows=[]
    try:
        if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):raise RuntimeError("KRX_ID/KRX_PW missing")
        print("KRX credentials present; authentication status must be verified independently",flush=True)
        for date in DATES:
            try:
                print("Snapshot",date,flush=True)
                frame,diag=run_date(date);diagnostic["runs"].append(diag)
                if not frame.empty:rows.append(frame)
            except Exception as e:
                diagnostic["errors"][date]=str(e);print("ERROR",date,str(e),flush=True)
        if not rows:raise RuntimeError("No candidates; inspect diagnostic")
        data=pd.concat(rows,ignore_index=True)
        data.to_csv(OUT/"candidates_forward.csv",index=False,encoding="utf-8-sig")
        agg=[]
        for date,g in data.groupby("signal_date"):
            for h in (5,10,20):
                r=g[f"next_close_{h}"].dropna()
                agg.append(dict(signal_date=date,horizon=h,n=len(r),mean_return=r.mean(),win_rate=(r>0).mean()))
        pd.DataFrame(agg).to_csv(OUT/"summary.csv",index=False,encoding="utf-8-sig")
    finally:
        (OUT/"diagnostic.json").write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
