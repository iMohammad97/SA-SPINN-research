"""TJU (Zhu 2022) raw parser -> per-cell npz in the xjtu.py format.

No temperature channel (chamber held constant) -> T = 0, so the T token carries no information.
Each cycle = CC-CV charge, rest, full CC discharge -> every cycle is a capacity test (anchor),
except gross outliers (|Q - rolling median| > max(3 MAD-sigma, 3 % Q_nom); rest recoveries kept), which are interpolated.
Cell name = b<batch>_<file stem> (same stems repeat across datasets).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

RAW, OUT = Path("data/raw/tju"), Path("data/proc/tju")
DIRS = {1: "Dataset_1_NCA_battery", 2: "Dataset_2_NCM_battery", 3: "Dataset_3_NCM_NCA_battery"}
Q_NOM = {1: 3.5, 2: 3.5, 3: 2.5}
DT_MAX = 600.0


def parse_cell(job):
    batch, path = job
    df = pd.read_csv(path, usecols=["time/s", "Ecell/V", "<I>/mA", "cycle number"])
    t0, V0 = df["time/s"].values, df["Ecell/V"].values
    I0, c0 = df["<I>/mA"].values / 1000.0, df["cycle number"].values.astype(np.int32)
    starts = np.flatnonzero(np.diff(c0, prepend=c0[0] - 1)).tolist() + [len(c0)]
    seg = []
    for a, b in zip(starts[:-1], starts[1:]):
        t = t0[a:b] - t0[a]
        dt = np.clip(np.diff(t), 0, DT_MAX)
        q = np.concatenate([[0.0], np.cumsum(0.5 * (I0[a + 1:b] + I0[a:b - 1]) * dt)]) / 3600.0
        pos = np.flatnonzero(I0[a:b] > 0)
        e = pos[-1] if len(pos) else -1
        seg.append((a, b, t, q, e, q[e] - q[e:].min() if e >= 0 else np.nan))
    qd = np.array([s[5] for s in seg])
    med = pd.Series(qd).rolling(9, center=True, min_periods=1).median().values
    r = qd - med
    sig = 1.4826 * np.nanmedian(np.abs(r)) + 1e-4
    anchor = np.isfinite(qd) & (np.abs(r) < max(3 * sig, 0.03 * Q_NOM[batch])) & (qd > 0.3 * Q_NOM[batch])
    k = np.arange(len(seg))
    soh = np.interp(k, k[anchor], qd[anchor] / Q_NOM[batch])
    cols = {n: [] for n in ("t", "V", "I", "T", "soc", "cycle")}
    for i, (a, b, t, q, e, _) in enumerate(seg):
        if e < 0:
            continue
        cols["t"].append(t); cols["V"].append(V0[a:b]); cols["I"].append(I0[a:b])
        cols["T"].append(np.zeros(b - a))
        cols["soc"].append(np.clip(1.0 + (q - q[e]) / (soh[i] * Q_NOM[batch]), 0.0, 1.0))
        cols["cycle"].append(np.full(b - a, i, np.int32))
    out = {n: np.concatenate(v).astype(np.int32 if n == "cycle" else np.float32) for n, v in cols.items()}
    out.update(soh=soh.astype(np.float32), anchor=anchor, qdis=qd.astype(np.float32))
    name = f"b{batch}_{path.stem}"
    np.savez(OUT / f"{name}.npz", batch=batch, **out)
    return (f"{name:28s} rows={len(out['t']):>9,} cyc={len(soh):4d} anch={anchor.sum():4d} "
            f"SOH {soh[0]:.3f}->{soh[-1]:.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, f) for b, d in DIRS.items() for f in sorted((RAW / d).glob("*.csv"))]
    with ProcessPoolExecutor(3) as ex:
        for msg in ex.map(parse_cell, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
