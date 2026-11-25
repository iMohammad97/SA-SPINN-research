"""Step-1 dataset: SOH cycle pairs + subsampled SOC time-step pairs, per XJTU batch.

Cache: ds.CACHE  (every CYC_EVERY-th cycle, every STRIDE rows)   Dataset: env DS=xjtu|tju (src/ds.py)
Split: ds.is_test (AutoSPINN rule per dataset); 1 train cell -> validation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from features import GROUPS
from ds import PROC, CACHE, CELLS, HI_CSV, STRIDE, CYC_EVERY, DT_MAX, MIN_DT, Q_NOM, TJU, draws_path, is_test
HI = GROUPS["V"] + GROUPS["I"] + GROUPS["T"] + GROUPS["clock"] + ["cycle"]
SOC_IN = ["V", "I", "T", "t"]


def build_cache():
    hi = pd.read_csv(HI_CSV)
    have = set(zip(hi.cell, hi.cycle))
    out = {k: [] for k in ("cell", "cycle", "V", "I", "T", "t", "q", "soc")}
    for f in sorted(CELLS.glob("*.npz")):
        d = np.load(f)
        cyc, t, I = d["cycle"], d["t"].astype(float), d["I"].astype(float)
        starts = np.flatnonzero(np.diff(cyc, prepend=-1)) .tolist() + [len(cyc)]
        for a, b in zip(starts[:-1], starts[1:]):
            k = int(cyc[a])
            if k % CYC_EVERY or (f.stem, k) not in have:
                continue
            dt = np.clip(np.diff(t[a:b]), 0, DT_MAX)
            q = np.concatenate([[0.0], np.cumsum(0.5 * (I[a + 1:b] + I[a:b - 1]) * dt)]) / 3600.0
            s = np.arange(0, b - a, STRIDE) if not MIN_DT else _min_gap(t[a:b], MIN_DT)
            out["cell"] += [f.stem] * len(s)
            out["cycle"].append(np.full(len(s), k))
            for n in ("V", "I", "T", "soc"):
                out[n].append(d[n][a:b][s])
            out["t"].append(t[a:b][s] - t[a])
            out["q"].append(q[s])
        print(f.stem, flush=True)
    np.savez(CACHE, **{k: (np.array(v) if k == "cell" else np.concatenate(v)) for k, v in out.items()})


def _min_gap(t, g):
    """Row indices with consecutive kept samples >= g seconds apart (greedy)."""
    keep, last = [0], t[0]
    for i in range(1, len(t)):
        if t[i] - last >= g:
            keep.append(i); last = t[i]
    return np.array(keep)


def _pairs(keys):
    """Indices i where row i and i+1 share the same key (consecutive within cell/cycle)."""
    return np.flatnonzero(keys[:-1] == keys[1:])


def add_uncertainty(d, batch):
    """Robust-training inputs: HI draws (s5_hidraws), Σ_HI (2nd moment of draw residuals, train cells), Σ_Z (budget)."""
    from gum import BUDGET
    D = torch.cat([torch.load(draws_path(b)) for b in d.get("batches", [batch])], 1)
    tr = torch.cat([d["train"]["soh_rows"]])
    R = (D[:, tr] - d["X"][tr]).reshape(-1, D.shape[2]).double()
    d["draws"], d["sx"] = D, R.T @ R / len(R)
    zmu, zsd = d["zstat"]
    i_rms = float(np.sqrt(np.mean((d["Z"][d["train"]["soc_rows"], 1].numpy() * zsd[1] + zmu[1]) ** 2)))
    v = [BUDGET["V_off"] ** 2 / 3 / zsd[0] ** 2,
         (BUDGET["I_off"] ** 2 + (BUDGET["I_gain"] * i_rms) ** 2) / 3 / zsd[1] ** 2,
         0.0 if d["T0"] is not None else BUDGET["T_off"] ** 2 / 3 / zsd[2] ** 2, 0.0, 0.0]   # V, I, T, t, SOH_hat
    d["sz"] = torch.tensor(v, dtype=torch.float64).diag()
    return d


def _cut(cell, cyc, th):
    """Within-cell split (protocols.within): cycles -> cell@tr / @va / @te at cycle thresholds th[c] = (a, b)."""
    cell = cell.astype(object).copy()
    for c, (a, b) in th.items():
        m = cell == c
        cell[m] = np.where(cyc[m] < a, c + "@tr", np.where(cyc[m] < b, c + "@va", c + "@te"))
    return cell


def load_batch(batch: int, val_cell: int = 0, rel_t: bool = False, split: dict = None):
    """Return dict split -> tensors. HI z-scored with train stats. rel_t: SOC sees T - T(cycle start).
    split: a published split (protocols.get) with batches / train / val / test lists and optional within-cell cut."""
    bs = split["batches"] if split else [batch]
    hi = pd.read_csv(HI_CSV)
    hi = pd.concat([hi[hi.batch == b] for b in bs], ignore_index=True)    # draws are stacked in this order
    raw = sorted(hi.cell.unique())
    cut = (split or {}).get("cut", {})
    th = {c: (k[int(0.9 * f * len(k))], k[int(f * len(k))])          # train = first 0.9 f, val = next 0.1 f
          for c, f in cut.items() for k in [np.sort(hi.cycle[hi.cell == c].values)]}
    if cut:
        hi["cell"] = _cut(hi.cell.values, hi.cycle.values, th)
    cells = sorted(hi.cell.unique())
    if split:
        train, val, test = list(split["train"]), list(split["val"]), list(split["test"])
        assert set(train + val + test) <= set(cells), set(train + val + test) - set(cells)
    else:
        test = is_test(cells)
        train = [c for c in cells if c not in test]
        val = [train.pop(i) for i in reversed(range(val_cell, len(train), 10 if TJU else len(train)))][::-1]  # TJU: every 10th (mixed conditions)
    mu, sd = hi.loc[hi.cell.isin(train), HI].mean(), hi.loc[hi.cell.isin(train), HI].std() + 1e-9
    X = torch.tensor(((hi[HI] - mu) / sd).values, dtype=torch.float32)

    c = np.load(CACHE)
    m = np.isin(c["cell"], raw)
    soc = {k: c[k][m] for k in c.files}
    if cut:
        soc["cell"] = _cut(soc["cell"], soc["cycle"], th).astype(str)
    row = pd.Series(hi.index.values, index=pd.MultiIndex.from_arrays([hi.cell, hi.cycle]))
    soc["hi_idx"] = row.loc[list(zip(soc["cell"], soc["cycle"]))].values
    key = np.char.add(soc["cell"].astype(str), soc["cycle"].astype(str))
    st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])             # cycle segments (rows contiguous)
    ln = np.diff(np.r_[st, len(key)])
    seg = np.repeat(np.arange(len(st)), ln)
    T0 = soc["T"][st[seg]] if rel_t else np.zeros_like(soc["T"])
    Z = np.stack([soc["V"], soc["I"], soc["T"] - T0, soc["t"]], 1).astype(np.float32)
    trm = np.isin(soc["cell"], train)
    zmu, zsd = Z[trm].mean(0), Z[trm].std(0) + 1e-9
    Z = (Z - zmu) / zsd

    out = {}
    for name, cs in (("train", train), ("val", val), ("test", test)):
        hm = hi.cell.isin(cs).values
        hidx = np.flatnonzero(hm)
        p = hidx[_pairs(hi.cell.values[hidx])]                       # SOH pairs k, k+1
        sm = np.isin(soc["cell"], cs)
        sidx = np.flatnonzero(sm)
        key = np.char.add(soc["cell"][sidx].astype(str), soc["cycle"][sidx].astype(str))
        sp = sidx[_pairs(key)]                                        # SOC pairs t, t+STRIDE
        out[name] = dict(
            soh_i=torch.tensor(p), soh_rows=torch.tensor(hidx),
            soh_y=torch.tensor(hi.soh.values, dtype=torch.float32),
            anchor=torch.tensor(hi.anchor.values.astype(bool)),
            dk=torch.tensor((hi.cycle.values[p + 1] - hi.cycle.values[p]) / 100.0, dtype=torch.float32),
            soc_i=torch.tensor(sp), soc_rows=torch.tensor(sidx),
            dq=torch.tensor(soc["q"][sp + 1] - soc["q"][sp], dtype=torch.float32),
            dt_h=torch.tensor((soc["t"][sp + 1] - soc["t"][sp]) / 3600.0, dtype=torch.float32),
        )
    out["X"], out["Z"] = X, torch.tensor(Z)
    out["soc_y"] = torch.tensor(soc["soc"], dtype=torch.float32)
    out["hi_idx"] = torch.tensor(soc["hi_idx"])
    out["T0"] = torch.tensor(T0, dtype=torch.float64) if rel_t else None
    out["seg0"], out["segn"] = torch.tensor(st[seg]), torch.tensor(ln[seg])
    out["q"] = torch.tensor(soc["q"], dtype=torch.float64)            # Ah since cycle start
    out["t_h"] = torch.tensor(soc["t"] / 3600.0, dtype=torch.float64)
    out["zstat"] = (zmu, zsd)
    out["q_nom"] = Q_NOM[bs[0]]
    out["batches"] = bs
    out["xstat"] = (torch.tensor(mu.values), torch.tensor(sd.values))
    out["hi"] = hi[["cell", "cycle"]]
    out["cells"] = dict(train=train, val=val, test=test)
    return out


if __name__ == "__main__":
    build_cache()
