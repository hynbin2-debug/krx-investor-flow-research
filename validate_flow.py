"""Offline robustness validation for flow_panel.csv, without KRX requests."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", help="flow_panel.csv extracted from pilot artifact")
    parser.add_argument("--out", default="validation")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.panel, dtype={"ticker": str})
    df["date"] = pd.to_datetime(df["date"])
    if df.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker-date records")
    df["month"] = df.date.dt.to_period("M").astype(str)
    summary, breakdown = [], []
    for window in (5, 10, 20):
        for horizon in (5, 10, 20):
            col = f"next_close_{horizon}"
            x = df.dropna(subset=[col, f"foreign_{window}", f"institution_{window}"]).copy()
            x["excess"] = x[col] - x.groupby("date")[col].transform("mean")
            dates = sorted(x.date.unique())
            sampled = set(dates[::horizon])
            for subset, y in (("all", x), ("nonoverlap", x[x.date.isin(sampled)])):
                for group, z in y.groupby(f"group_{window}"):
                    summary.append({"signal_days": window, "future_days": horizon,
                                    "subset": subset, "group": group, "n": len(z),
                                    "win_rate": (z[col] > 0).mean(),
                                    "mean_return": z[col].mean(),
                                    "mean_excess": z.excess.mean(),
                                    "excess_win_rate": (z.excess > 0).mean()})
            if window == 10 and horizon == 5:
                for dim in ("month", "ticker"):
                    for (key, group), z in x.groupby([dim, f"group_{window}"]):
                        breakdown.append({"dimension": dim, "value": key, "group": group,
                                          "n": len(z), "mean_excess": z.excess.mean(),
                                          "win_rate": (z[col] > 0).mean()})
    pd.DataFrame(summary).to_csv(out / "validation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(breakdown).to_csv(out / "validation_month_ticker.csv", index=False, encoding="utf-8-sig")
    print("Wrote validation_summary.csv and validation_month_ticker.csv")

if __name__ == "__main__":
    main()
