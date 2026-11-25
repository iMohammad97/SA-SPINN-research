"""Self-validation of the HI table (independent of old CSVs; those = loose cross-check)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import GROUPS, NAMES

df = pd.read_csv("data/proc/xjtu_hi.csv")
print(f"rows={len(df):,} cells={df.cell.nunique()} NaN={int(df[NAMES].isna().sum().sum())}")

print("\n1) Ranges (p1 / median / p99)")
print(df[NAMES].quantile([0.01, 0.5, 0.99]).T.round(4).to_string())

print("\n2) Trend: median within-cell Spearman corr with SOH, per batch")
tr = df.groupby(["batch", "cell"]).apply(lambda g: g[NAMES].corrwith(g.soh, method="spearman"))
print(tr.groupby("batch").median().T.round(2).to_string())

print("\n3) Simple SOH fit (ridge), test cells = name contains 4 or 8")
test = df.cell.str.contains("4|8")
sets = {"all": NAMES, "no_T": [n for n in NAMES if n not in GROUPS["T"]]}
res = {}
for name, cols in sets.items():
    X = df[cols + ["cycle"]]
    m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(X[~test], df.soh[~test])
    err = (m.predict(X[test]) - df.soh[test]) ** 2
    res[name] = np.sqrt(err.groupby(df.batch[test]).mean())
print((pd.DataFrame(res) * 100).round(2).to_string(), "  (RMSE, % SOH)")

print("\n4) Loose cross-check vs old CSVs (per-cell medians, ratio new/old)")
old_map = {"v_mean": "voltage mean", "cc_q": "CC Q", "cc_time": "CC charge time",
           "cv_q": "CV Q", "cv_time": "CV charge time", "i_mean": "current mean"}
rat = []
for f in sorted(Path("data/xjtu_features").glob("*.csv")):
    o, n = pd.read_csv(f), df[df.cell == f.stem]
    if len(n):
        rat.append({k: n[k].median() / o[v].median() for k, v in old_map.items()})
print(pd.DataFrame(rat).describe().loc[["mean", "min", "max"]].round(3).to_string())
