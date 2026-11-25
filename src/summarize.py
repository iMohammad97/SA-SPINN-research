"""Summarize a grid: python src/summarize.py runs/s1_grouping [group_col]"""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/s1_grouping")
col = sys.argv[2] if len(sys.argv) > 2 else "grouping"
df = pd.DataFrame([json.loads(p.read_text()) for p in (root / "results").glob("*.json")])
df[["SOH %", "SOC %"]] = df[["test_soh_rmse", "test_soc_rmse"]] * 100
g = df.groupby(["batch", col])[["SOH %", "SOC %"]].agg(["mean", "std"]).round(2)
print(g.to_string(), "\n")
print("Mean over batches:")
print(df.groupby(col)[["SOH %", "SOC %", "params", "time_s"]].mean().round(2).to_string())
g.to_csv(root / "summary.csv")
