"""XJTU raw parser -> per-cell npz with V/I/T time series, SOC and SOH labels.

Charge is integrated from the clean recorded current; relative_time resets
per step, so resets count as one 1 Hz sample. Discharge capacity per cycle = Ah removed
after end of charge (summary file is not used).
SOH: anchor = reference full-discharge cycles; linear interp to all cycles.
SOC: 1 at end of charge, capacity = SOH_k * Q_NOM (couples SOC to SOH).
"""
from pathlib import Path
import numpy as np
import scipy.io as sio

Q_NOM = 2.0
RAW = Path("data/raw/xjtu")
OUT = Path("data/proc/xjtu")


def is_anchor(batch: int, desc: str) -> bool:
    if "0.2C discharge" in desc:  # initial low-rate capacity test
        return False
    if batch in (1, 2):
        return True
    if batch == 3:
        return "1C discharge" in desc
    return "[test capacity]" in desc


def cycle_charge(c):
    """Return t [s], I [A], cumulative charge q [Ah], end-of-charge index."""
    I = np.asarray(c["current_A"], float)
    dt = np.diff(np.asarray(c["relative_time_min"], float)) * 60.0
    dt[~(dt >= 0)] = 1.0               # step reset (negative) or nan -> one 1 Hz sample
    dt = np.clip(dt, 0.0, 2.0)         # cap logging gaps
    t = np.concatenate([[0.0], np.cumsum(dt)])
    q = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * dt)]) / 3600.0
    pos = np.flatnonzero(I > 0)
    return t, I, q, (pos[-1] if len(pos) else -1)


def parse_cell(path: Path, batch: int) -> dict:
    cycles = sio.loadmat(path, simplify_cells=True)["data"]
    n = len(cycles)
    pre = [cycle_charge(c) for c in cycles]
    qdis = np.array([q[e] - q[e:].min() if e >= 0 else np.nan for _, _, q, e in pre])
    anchor = np.array([is_anchor(batch, c["description"]) for c in cycles]) & np.isfinite(qdis)
    k = np.arange(n)
    soh = np.interp(k, k[anchor], qdis[anchor] / Q_NOM)

    cols = {n_: [] for n_ in ("t", "V", "I", "T", "soc", "cycle")}
    for i, (c, (t, I, q, e)) in enumerate(zip(cycles, pre)):
        if i == 0 or e < 0:
            continue
        cols["t"].append(t)
        cols["V"].append(c["voltage_V"])
        cols["I"].append(I)
        cols["T"].append(c["temperature_C"])
        # clip: tiny <0 at charge start after a deeper previous discharge (<0.3% pts)
        cols["soc"].append(np.clip(1.0 + (q - q[e]) / (soh[i] * Q_NOM), 0.0, 1.0))
        cols["cycle"].append(np.full(len(I), i, np.int32))
    out = {n_: np.concatenate(v).astype(np.int32 if n_ == "cycle" else np.float32) for n_, v in cols.items()}
    out.update(soh=soh.astype(np.float32), anchor=anchor, qdis=qdis.astype(np.float32),
               desc=np.array([c["description"] for c in cycles]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW.glob("Batch-*/*.mat")):
        batch = int(f.parent.name.split("-")[1])
        d = parse_cell(f, batch)
        np.savez(OUT / f"{f.stem}.npz", batch=batch, **d)
        print(f"{f.stem:24s} rows={len(d['t']):>9,} cyc={len(d['soh']):4d} anch={d['anchor'].sum():4d} "
              f"SOH {d['soh'][1]:.3f}->{d['soh'][-1]:.3f} SOC[{d['soc'].min():.3f},{d['soc'].max():.3f}]", flush=True)


if __name__ == "__main__":
    main()
