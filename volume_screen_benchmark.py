"""Compare historical candidate forward returns with matching KOSPI/KOSDAQ indices."""
import json,time
from pathlib import Path
import pandas as pd
from pykrx import stock

ROOT=Path("outputs_benchmark")
ROOT.mkdir(exist_ok=True)
SRC=Path("input_artifact/candidates_forward.csv")
HORIZONS=(5,10,20)
INDEX={"KOSPI":"1001","KOSDAQ":"2001"}
def fetch_index(code,start,end):
    for attempt in range(3):
        try:
            df=stock.get_index_ohlcv_by_date(start,end,code).sort_index()
            if df.empty or "종가" not in df:raise RuntimeError("empty index "+code)
            return df
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))
def main():
    if not SRC.exists():raise FileNotFoundError(str(SRC))
    d=pd.read_csv(SRC,dtype={"ticker":str,"signal_date":str})
    d["ticker"]=d["ticker"].str.zfill(6)
    d["rank"]=d.groupby("signal_date")["multiple"].rank(method="first",ascending=False)
    diagnostics={"source_rows":len(d),"errors":[],"index_codes":INDEX}
    for date in sorted(d.signal_date.unique()):
        for market,code in INDEX.items():
            mask=(d.signal_date==date)&(d.market==market)
            if not mask.any():continue
            try:
                end=(pd.Timestamp(date)+pd.Timedelta(days=65)).strftime("%Y%m%d")
                idx=fetch_index(code,date,end)
                idx=idx[idx.index>pd.Timestamp(date)]
                if len(idx)<20:raise RuntimeError("insufficient forward index sessions")
                for h in HORIZONS:
                    # The original stock backtest uses close on first post-signal session
                    # through close on the h-th post-signal session.
                    start=float(idx.iloc[0]["종가"]);finish=float(idx.iloc[h-1]["종가"])
                    if start<=0:raise RuntimeError("nonpositive index close")
                    d.loc[mask,f"index_return_{h}"]=finish/start-1
                    d.loc[mask,f"index_entry_{h}"]=idx.index[0].strftime("%Y%m%d")
                    d.loc[mask,f"index_exit_{h}"]=idx.index[h-1].strftime("%Y%m%d")
            except Exception as exc:
                diagnostics["errors"].append({"date":date,"market":market,"error":str(exc)})
    summaries=[]
    for date,g in d.groupby("signal_date"):
        for scope,subset in [("all",g),("top10",g[g["rank"]<=10])]:
            for h in HORIZONS:
                stockret=pd.to_numeric(subset[f"next_close_{h}"],errors="coerce")
                indexret=pd.to_numeric(subset.get(f"index_return_{h}"),errors="coerce")
                valid=stockret.notna()&indexret.notna()
                s=stockret[valid];b=indexret[valid];ex=s-b
                summaries.append(dict(signal_date=date,scope=scope,horizon=h,
                    candidates=len(subset),n=len(s),stock_mean=s.mean(),benchmark_mean=b.mean(),
                    excess_mean=ex.mean(),excess_median=ex.median(),
                    stock_win_rate=(s>0).mean(),beat_market_rate=(ex>0).mean()))
    for h in HORIZONS:
        for scope,subset in [("all",d),("top10",d[d["rank"]<=10])]:
            s=pd.to_numeric(subset[f"next_close_{h}"],errors="coerce")
            b=pd.to_numeric(subset.get(f"index_return_{h}"),errors="coerce")
            valid=s.notna()&b.notna()
            x=s[valid];y=b[valid];ex=x-y
            summaries.append(dict(signal_date="ALL",scope=scope,horizon=h,
                candidates=len(subset),n=len(x),stock_mean=x.mean(),benchmark_mean=y.mean(),
                excess_mean=ex.mean(),excess_median=ex.median(),
                stock_win_rate=(x>0).mean(),beat_market_rate=(ex>0).mean()))
    for h in HORIZONS:
        d[f"excess_{h}"]=pd.to_numeric(d[f"next_close_{h}"],errors="coerce")-pd.to_numeric(d.get(f"index_return_{h}"),errors="coerce")
    d.to_csv(ROOT/"candidate_vs_index.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(ROOT/"market_comparison_summary.csv",index=False,encoding="utf-8-sig")
    (ROOT/"diagnostic.json").write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    if diagnostics["errors"]:raise RuntimeError("Index retrieval errors; inspect diagnostic.json")
if __name__=="__main__":main()
