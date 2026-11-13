"""Bounded exploratory investor-flow study. No forward-looking data in signal."""
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from pykrx import stock

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
START, END = "20260101", "20261030"
WINDOWS = (5, 10, 20)
WAIT = 0.4


def krx_login():
    """Retained login hook. Do not claim authentication without a verified API."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    user = os.getenv("KRX_ID")
    password = os.getenv("KRX_PASSWORD")
    if user or password:
        print("KRX credentials supplied; authenticated login endpoint not verified. Public pykrx access only.")
    else:
        print("KRX login hook retained; using public pykrx data without credentials.")
    return session


def to_numeric(frame, columns):
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def select_tickers(end, n):
    cap = stock.get_market_cap_by_ticker(end, market="ALL")
    if cap.empty or "시가총액" not in cap:
        raise RuntimeError("KRX market cap response missing")
    cap = cap.sort_values("시가총액", ascending=False)
    selected = []
    for ticker in cap.index.astype(str):
        try:
            name = stock.get_market_ticker_name(ticker)
            if "스팩" in name or "SPAC" in name.upper() or name.endswith("우"):
                continue
            selected.append((ticker, name))
            if len(selected) >= n:
                break
        except Exception:
            continue
    return selected


def fetch_one(ticker, name):
    # Extra history for lookback and forward evaluation; no interpolation of missing days.
    prices = stock.get_market_ohlcv_by_date("20251101", "20261130", ticker)
    time.sleep(WAIT)
    flow = stock.get_market_trading_volume_by_date("20251101", "20261130", ticker)
    time.sleep(WAIT)
    required_price = ["종가", "거래량"]
    foreign_col = "외국인합계" if "외국인합계" in flow.columns else "외국인"
    required_flow = [foreign_col, "기관합계"]
    if prices.empty or flow.empty:
        raise ValueError("empty prices or flow")
    if any(c not in prices for c in required_price) or any(c not in flow for c in required_flow):
        raise ValueError(f"missing required columns: {list(prices.columns)} / {list(flow.columns)}")
    p = to_numeric(prices[required_price].copy(), required_price)
    f = to_numeric(flow[required_flow].copy(), required_flow).rename(columns={foreign_col: "외국인"})
    df = p.join(f, how="inner").sort_index()
    df = df.loc[~df.index.duplicated(keep=False)].dropna()
    df = df[(df["종가"] > 0) & (df["거래량"] >= 0)]
    if len(df) < 50:
        raise ValueError("less than 50 complete observations")
    for w in WINDOWS:
        df[f"foreign_{w}"] = df["외국인"].rolling(w, min_periods=w).sum()
        df[f"institution_{w}"] = df["기관합계"].rolling(w, min_periods=w).sum()
        # Signal close to future close (diagnostic, not directly executable).
        df[f"future_{w}"] = df["종가"].shift(-w) / df["종가"] - 1
        # Entry at next day's close, exit at w-th future day's close.
        df[f"next_close_{w}"] = df["종가"].shift(-w) / df["종가"].shift(-1) - 1
    df = df.loc[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))].copy()
    df["ticker"], df["name"] = ticker, name
    df["date"] = df.index.strftime("%Y-%m-%d")
    for w in WINDOWS:
        a, b = df[f"foreign_{w}"], df[f"institution_{w}"]
        df[f"group_{w}"] = "neutral"
        df.loc[(a > 0) & (b > 0), f"group_{w}"] = "A_both_buy"
        df.loc[(a > 0) & (b < 0), f"group_{w}"] = "B_foreign_buy"
        df.loc[(a < 0) & (b > 0), f"group_{w}"] = "C_institution_buy"
        df.loc[(a < 0) & (b < 0), f"group_{w}"] = "D_both_sell"
    return df.reset_index(drop=True)


def summarize(panel):
    rows = []
    for signal_days in WINDOWS:
        for horizon in WINDOWS:
            col = f"next_close_{horizon}"
            for group, part in panel.groupby(f"group_{signal_days}"):
                values = part[col].dropna()
                if values.empty:
                    continue
                rows.append({
                    "signal_days": signal_days, "future_days": horizon,
                    "group": group, "observations": int(len(values)),
                    "distinct_tickers": int(part.loc[values.index, "ticker"].nunique()),
                    "win_rate": float((values > 0).mean()),
                    "mean_return": float(values.mean()),
                    "median_return": float(values.median()),
                    "note": "overlapping signals; descriptive only",
                })
    return pd.DataFrame(rows)


def main():
    krx_login()
    sample_size = max(10, min(100, int(os.getenv("SAMPLE_SIZE", "30"))))
    diag = {"started_utc": datetime.utcnow().isoformat(), "requested": sample_size,
            "errors": {}, "data_source": "pykrx KRX", "sampling": "market cap top, not full market"}
    frames = []
    try:
        tickers = select_tickers(END, sample_size)
        diag["selected"] = tickers
        for i, (ticker, name) in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker} {name}", flush=True)
            try:
                frames.append(fetch_one(ticker, name))
            except Exception as exc:
                diag["errors"][ticker] = str(exc)
                print(f"SKIP {ticker}: {exc}", flush=True)
        if not frames:
            raise RuntimeError("No valid data; inspect diagnostic errors")
        panel = pd.concat(frames, ignore_index=True)
        panel.to_csv(OUT / "flow_panel.csv", index=False, encoding="utf-8-sig")
        summarize(panel).to_csv(OUT / "flow_summary.csv", index=False, encoding="utf-8-sig")
        diag["success_tickers"] = len(frames)
        diag["panel_rows"] = len(panel)
        diag["status"] = "completed"
    except Exception as exc:
        diag["status"] = "failed"
        diag["fatal_error"] = str(exc)
        diag["traceback"] = traceback.format_exc()
        raise
    finally:
        (OUT / "diagnostic.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
