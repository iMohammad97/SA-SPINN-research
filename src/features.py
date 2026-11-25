"""Sensor-aligned health indicators from one charge (torch, differentiable in V/I/T).

Windows (end of charge, protocol-invariant):
  voltage window: CC phase with V in [V_END-0.2, CV start)
  current window: CV phase with I in [0.1, 0.5] A
Window masks are not differentiable (selection); statistics are.
"""
import numpy as np
import torch
from ds import DT_MAX, CELLS, HI_CSV

V_END, CV_TOL = 4.2, 0.005
GROUPS = {
    "V": ["v_mean", "v_std", "v_kurt", "v_skew", "v_slope", "v_entropy"],
    "I": ["i_mean", "i_std", "i_kurt", "i_skew", "i_slope", "i_entropy", "cc_q", "cv_q"],
    "T": ["t_std", "t_rise"],  # relative only: absolute T tracks seasonal ambient (Sep->Feb)
    "clock": ["cc_time", "cv_time"],  # + cycle index appended by the dataset
}
NAMES = [n for g in GROUPS.values() for n in g]


def _stats(x, t):
    m, s = x.mean(), x.std()
    z = (x - m) / (s + 1e-9)
    tc = t - t.mean()
    slope = (tc * (x - m)).sum() / (tc.pow(2).sum() + 1e-9)
    p = x.abs() / x.abs().sum()
    ent = -(p * torch.log(p + 1e-12)).sum()
    return [m, s, z.pow(4).mean() - 3.0, z.pow(3).mean(), slope, ent]


def _q(I, t):
    return (0.5 * (I[1:] + I[:-1]) * (t[1:] - t[:-1]).clamp(0, DT_MAX)).sum() / 3600.0


def charge_features(t, V, I, T):
    """t,V,I,T: 1-D tensors of one cycle up to end of charge. Returns tensor[len(NAMES)] or None."""
    chg = I > 0.05
    cv = chg & (V >= V_END - CV_TOL)
    if cv.sum() < 5:
        return None
    cv_start = int(torch.nonzero(cv)[0])
    idx = torch.arange(len(V))
    vw = chg & (idx < cv_start) & (V >= V_END - 0.2)
    iw = cv & (idx >= cv_start) & (I >= 0.1) & (I <= 0.5)
    if vw.sum() < 5 or iw.sum() < 5:
        return None
    f = _stats(V[vw], t[vw]) + _stats(I[iw], t[iw]) + [_q(I[vw], t[vw]), _q(I[iw], t[iw])]
    Tc = T[chg]
    f += [Tc.std(), Tc.max() - Tc[0]]
    f += [t[vw][-1] - t[vw][0], t[iw][-1] - t[iw][0]]
    return torch.stack(f)


def cell_features(d: dict):
    """d: loaded npz of one cell. Returns (cycle_ids, features[n, F], soh[n])."""
    t, V, I, T, cyc = (torch.from_numpy(d[k]).double() for k in ("t", "V", "I", "T", "cycle"))
    cyc = cyc.long()
    starts = torch.nonzero(torch.diff(cyc, prepend=cyc[:1] - 1)).squeeze(1).tolist() + [len(cyc)]
    ids, feats = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        pos = torch.nonzero(I[a:b] > 0).squeeze(1)
        if len(pos) == 0:
            continue
        e = a + int(pos[-1]) + 1
        f = charge_features(t[a:e], V[a:e], I[a:e], T[a:e])
        if f is not None:
            ids.append(int(cyc[a]))
            feats.append(f)
    ids = np.array(ids)
    return ids, torch.stack(feats).numpy(), d["soh"][ids]


def cell_table(f):
    import pandas as pd
    torch.set_num_threads(1)
    d = dict(np.load(f))
    ids, X, soh = cell_features(d)
    df = pd.DataFrame(X, columns=NAMES)
    df.insert(0, "cycle", ids)
    df.insert(0, "batch", int(d["batch"]))
    df.insert(0, "cell", f.stem)
    df["anchor"] = d["anchor"][ids]
    df["soh"] = soh
    print(f"{f.stem:24s} {len(ids)}/{len(d['soh'])} cycles", flush=True)
    return df


def main():
    """Build sensor-aligned HI table for all cells of DS -> data/proc/<DS>_hi.csv"""
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(cell_table, sorted(CELLS.glob("*.npz"))))
    pd.concat(rows).to_csv(HI_CSV, index=False)


if __name__ == "__main__":
    main()
