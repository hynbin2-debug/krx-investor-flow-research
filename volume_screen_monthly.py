"""Independent monthly validation, original screening conditions unchanged."""
import json,traceback
from pathlib import Path
import pandas as pd
from volume_screen_historical import run_date
from volume_screen_benchmark import fetch_index,INDEX,HORIZONS

OUT=Path("outputs_monthly");OUT.mkdir(exist_ok=True)
MONTHS=[f"{y}{m:02d}" for y,months in [(2025,range(1,13)),(2026,range(1,9))] for m in months]
def save(rows,diagnostics):
    if rows:pd.concat(rows,ignore_index=True).to_csv(OUT/"monthly_candidates.csv",index=False,encoding="utf-8-sig")
    (OUT/"diagnostic.json").write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
    rows=[];diagnostics={"months":MONTHS,"success":[],"errors":{},"definition":"signal next close to fifth/tenth/twentieth subsequent session close; costs excluded"}
    for month in MONTHS:
        anchor=(pd.Period(month,freq="M").end_time).strftime("%Y%m%d")
        try:
            frame,diag=run_date(anchor)
            if frame.empty:
                diagnostics["success"].append({"month":month,**diag});save(rows,diagnostics);continue
            frame["month"]=month
            frame["rank"]=frame["multiple"].rank(method="first",ascending=False)
            date=diag["signal_date"]
            for market,code in INDEX.items():
                mask=frame.market.eq(market)
                if not mask.any():continue
                end=(pd.Timestamp(date)+pd.Timedelta(days=65)).strftime("%Y%m%d")
                idx=fetch_index(code,date,end)
                idx=idx[idx.index>pd.Timestamp(date)]
                if len(idx)<20:raise RuntimeError(f"{month} {market}: missing future index sessions")
                for h in HORIZONS:
                    start=float(idx.iloc[0]["종가"]);finish=float(idx.iloc[h-1]["종가"])
                    if start<=0:raise RuntimeError("invalid index close")
                    frame.loc[mask,f"benchmark_{h}"]=finish/start-1
                    frame.loc[mask,f"excess_{h}"]=pd.to_numeric(frame.loc[mask,f"next_close_{h}"],errors="coerce")-(finish/start-1)
            rows.append(frame)
            diagnostics["success"].append({"month":month,**diag})
            print("DONE",month,"candidates",len(frame),flush=True)
        except Exception as e:
            diagnostics["errors"][month]=str(e)
            print("ERROR",month,str(e),flush=True)
        save(rows,diagnostics)
    if not rows:raise RuntimeError("No valid monthly candidate data")
    d=pd.concat(rows,ignore_index=True)
    stats=[]
    for month,g in d.groupby("month"):
        for scope,x in [("all",g),("top10",g[g["rank"]<=10])]:
            for h in HORIZONS:
                r=pd.to_numeric(x[f"next_close_{h}"],errors="coerce")
                b=pd.to_numeric(x[f"benchmark_{h}"],errors="coerce")
                good=r.notna()&b.notna()
                r=r[good];b=b[good];e=r-b
                stats.append(dict(month=month,scope=scope,horizon=h,candidates=len(x),n=len(e),
                    stock_mean=r.mean(),index_mean=b.mean(),excess_mean=e.mean(),
                    beat_index_rate=(e>0).mean(),win_rate=(r>0).mean()))
    monthly=pd.DataFrame(stats)
    monthly.to_csv(OUT/"monthly_summary.csv",index=False,encoding="utf-8-sig")
    aggregate=[]
    for (scope,h),g in monthly.groupby(["scope","horizon"]):
        valid=g[g.n>0]
        aggregate.append(dict(scope=scope,horizon=h,months=len(valid),total_observations=int(valid.n.sum()),
            equal_month_excess=valid.excess_mean.mean(),median_month_excess=valid.excess_mean.median(),
            positive_month_fraction=(valid.excess_mean>0).mean(),
            equal_month_stock_return=valid.stock_mean.mean(),equal_month_index_return=valid.index_mean.mean(),
            mean_beat_index_rate=valid.beat_index_rate.mean()))
    pd.DataFrame(aggregate).to_csv(OUT/"equal_month_summary.csv",index=False,encoding="utf-8-sig")
    print(pd.DataFrame(aggregate).to_string(index=False),flush=True)
    if diagnostics["errors"]:raise RuntimeError("Some months failed; inspect diagnostic.json")
if __name__=="__main__":main()
