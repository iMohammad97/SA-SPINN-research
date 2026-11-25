"""GUM propagation of a BQ76952-like instrument budget through features -> subnets -> fusion -> SOH/SOC.

MC (JCGM 101): one draw = one whole history; persistent terms fixed per draw, rectangular bounds.
Linear (JCGM 100): u^2 = sum_k (dy/dtheta_k)^2 u_k^2, forward-mode AD through feature extraction.
Usage: python src/gum.py [--draws 200] [--workers 6]      -> runs/s3_gum/   Monitor: python src/monitor.py runs/s3_gum
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

sys.path.insert(0, str(Path(__file__).parent))
from train import logger  # noqa: E402
import ds  # noqa: E402

# Declared scenario (NOT XJTU's equipment). Half-widths a; rectangular -> u = a / sqrt(3).
BUDGET = {"V_off": 5e-3,      # V, cell voltage accuracy at 25 C (BQ76952 datasheet)
          "I_off": 7.6e-3,    # A, CC offset 1 LSB with declared 1 mOhm shunt
          "I_gain": 5e-3,     # -, shunt + gain after calibration (declared assumption)
          "T_off": 1.0}       # K, thermistor channel (declared assumption)
KEYS = list(BUDGET)
A = torch.tensor([BUDGET[k] for k in KEYS], dtype=torch.float64)
OUT = Path("runs/s3_gum")


def perturb(V, I, T, th):
    return V + th[0], I * (1 + th[2]) + th[1], T + th[3]


def predict(m, d, raw, cache, th):
    """Model outputs (SOH per test cycle, SOC per test cache row) under persistent error th."""
    from features import charge_features
    X = d["X"].double().clone()
    mu, sd = d["xstat"]
    for cell, (t, V, I, T, cyc, rows) in raw.items():
        Vp, Ip, Tp = perturb(V, I, T, th)
        for r, (a, e) in rows.items():                        # hi row -> raw slice (start, end of charge)
            f = charge_features(t[a:e], Vp[a:e], Ip[a:e], Tp[a:e])
            if f is not None:
                X[r, :-1] = (f - mu[:-1]) / sd[:-1]
    Xf = X.float()
    soh_rows = d["test"]["soh_rows"]
    s = m.soh(Xf[soh_rows])
    zr, zmu, zsd = cache
    Vz, Iz, Tz = perturb(zr[:, 0], zr[:, 1], zr[:, 2], th)
    if zr.shape[1] == 4:                                      # relative T: offset hits T and T0 alike
        Tz = Tz - (zr[:, 3] + th[3])
    Z = torch.stack([(Vz - zmu[0]) / zsd[0], (Iz - zmu[1]) / zsd[1], (Tz - zmu[2]) / zsd[2],
                     d["Z"][d["test"]["soc_rows"], 3].double()], 1).float()
    sh = m.soh(Xf[d["hi_idx"][d["test"]["soc_rows"]]])
    u = m.soc(m.soc_in(Z, sh))
    return s, u


def load_raw(d, cells=None):
    """Raw cell series (default: test cells) + map hi row -> (start, end-of-charge) for HI-table cycles."""
    raw, hi = {}, d["hi"]
    for cell in cells or d["cells"]["test"]:
        z = np.load(ds.CELLS / f"{cell}.npz")
        t, V, I, T = (torch.from_numpy(z[k]).double() for k in ("t", "V", "I", "T"))
        cyc = z["cycle"]
        starts = np.flatnonzero(np.diff(cyc, prepend=cyc[0] - 1)).tolist() + [len(cyc)]
        seg = {int(cyc[a]): (a, b) for a, b in zip(starts[:-1], starts[1:])}
        rows = {}
        for r in np.flatnonzero(hi.cell.values == cell):
            a, b = seg[int(hi.cycle.values[r])]
            pos = np.flatnonzero(z["I"][a:b] > 0)
            rows[int(r)] = (a, a + int(pos[-1]) + 1)
        raw[cell] = (t, V, I, T, cyc, rows)
    return raw


def job(cfg):
    torch.set_num_threads(2)
    from data import load_batch
    from model import CoupledSPINN
    d = load_batch(cfg["batch"], rel_t=cfg["relt"])
    m = CoupledSPINN("sensor", "log_mean", "gray")
    m.load_state_dict(torch.load(cfg["model"]))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    raw = load_raw(d)
    c = np.load(ds.CACHE)
    sel = np.isin(c["cell"], d["cells"]["test"])
    zr = torch.tensor(np.stack([c["V"][sel], c["I"][sel], c["T"][sel]], 1), dtype=torch.float64)
    if cfg["relt"]:
        zr = torch.cat([zr, d["T0"][d["test"]["soc_rows"]][:, None]], 1)
    zmu, zsd = (torch.tensor(x[:3], dtype=torch.float64) for x in d["zstat"])
    cache = (zr, zmu, zsd)
    t0 = time.time()
    with torch.no_grad():
        s0, u0 = predict(m, d, raw, cache, torch.zeros(4, dtype=torch.float64))
        g = torch.Generator().manual_seed(cfg["seed"])
        S, U = [], []
        st = Path(cfg["out"]) / "status" / f"{cfg['id']}.json"
        for i in range(cfg["draws"]):
            if i % 10 == 0:
                st.write_text(json.dumps(dict(id=cfg["id"], epoch=i, epochs=cfg["draws"], elapsed=time.time() - t0,
                                              done=False)))
            th = (2 * torch.rand(4, generator=g, dtype=torch.float64) - 1) * A
            s, u = predict(m, d, raw, cache, th)
            S.append(s); U.append(u)
    S, U = torch.stack(S), torch.stack(U)
    # linear GUM: forward-mode AD, one tangent per budget term (sensitivity per unit)
    J_s, J_u = [], []
    with torch.no_grad():
        for k in range(4):
            with fwAD.dual_level():
                th = fwAD.make_dual(torch.zeros(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)[k])
                s, u = predict(m, d, raw, cache, th)
                J_s.append(fwAD.unpack_dual(s).tangent); J_u.append(fwAD.unpack_dual(u).tangent)
    J_s, J_u = torch.stack(J_s, 1), torch.stack(J_u, 1)      # [n, 4]
    uk = (A / 3 ** 0.5).float()
    lin_s, lin_u = (J_s * uk) ** 2, (J_u * uk) ** 2            # per-source variance contributions
    n = cfg["draws"]
    res = dict(cfg, time_s=time.time() - t0,
               u_mc_soh=S.std(0).mean().item(), u_lin_soh=lin_s.sum(1).sqrt().mean().item(),
               u_mc_soc=U.std(0).mean().item(), u_lin_soc=lin_u.sum(1).sqrt().mean().item(),
               u_mc_soh_half=S[: n // 2].std(0).mean().item(), u_mc_soc_half=U[: n // 2].std(0).mean().item(),
               share_soh=dict(zip(KEYS, (lin_s.mean(0) / lin_s.mean(0).sum()).tolist())),
               share_soc=dict(zip(KEYS, (lin_u.mean(0) / lin_u.mean(0).sum()).tolist())),
               bias_soh=(S.mean(0) - s0).abs().mean().item(), bias_soc=(U.mean(0) - u0).abs().mean().item())
    (Path(cfg["out"]) / "results" / f"{cfg['id']}.json").write_text(json.dumps(res))
    st.write_text(json.dumps(dict(id=cfg["id"], epoch=n, epochs=n, elapsed=res["time_s"], done=True)))
    return res


def hi_draws(cfg):
    """Robust-training input: K budget draws of the z-scored HI table (all cells; one persistent draw per cell history)."""
    torch.set_num_threads(2)
    from data import load_batch
    from features import charge_features
    d = load_batch(cfg["batch"])
    cells = [c for v in d["cells"].values() for c in v]
    raw = load_raw(d, cells)
    mu, sd = d["xstat"]
    g = torch.Generator().manual_seed(cfg["batch"])
    D = d["X"].double().repeat(cfg["k"], 1, 1)
    for k in range(cfg["k"]):
        for cell, (t, V, I, T, cyc, rows) in raw.items():
            th = (2 * torch.rand(4, generator=g, dtype=torch.float64) - 1) * A
            Vp, Ip, Tp = perturb(V, I, T, th)
            for r, (a, e) in rows.items():
                f = charge_features(t[a:e], Vp[a:e], Ip[a:e], Tp[a:e])
                if f is not None:
                    D[k, r, :-1] = (f - mu[:-1]) / sd[:-1]
    path = ds.draws_path(cfg["batch"])
    torch.save(D.float(), path)
    return cfg["batch"], (D - d["X"].double()).std(0).mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batches", default=",".join(map(str, ds.BATCHES)))
    ap.add_argument("--hi-draws", type=int, default=0, help="K HI-table draws per batch for robust training -> data/proc")
    ap.add_argument("--relt", action="store_true", help="relative-T SOC models (runs/s5_relt) -> runs/s5_relt_gum")
    a = ap.parse_args()
    if a.hi_draws:
        hd = Path(f"runs/{ds.STEP}_hidraws" if ds.TJU else "runs/s5_hidraws")   # TJU: run with env DS=tju
        hd.mkdir(parents=True, exist_ok=True)
        say = logger(hd)
        with ProcessPoolExecutor(a.workers) as ex:
            for b, u in ex.map(hi_draws, [dict(batch=b, k=a.hi_draws) for b in map(int, a.batches.split(","))]):
                say(f"b{b}: mean u per HI (z units): " + " ".join(f"{x:.3f}" for x in u))
        return
    global OUT
    OUT = Path("runs/s5_relt_gum") if a.relt else OUT
    for sub in ("status", "results"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    say = logger(OUT)
    cfgs = []
    for b in map(int, a.batches.split(",")):
        for s in (0, 1):
            i = f"b{b}_s{s}"
            if not (OUT / "results" / f"{i}.json").exists():
                mp = (f"runs/s5_relt/models/b{b}_sensor_log_mean_gray_a0.001_s{s}_relT.pt" if a.relt else
                      f"runs/s2_physics/models/b{b}_sensor_log_mean_gray_a0.001_s{s}.pt")
                cfgs.append(dict(id=i, batch=b, seed=s, draws=a.draws, relt=a.relt, model=mp, out=str(OUT)))
    say(f"{len(cfgs)} jobs -> {OUT}")
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, cfgs):
            say(f"{r['id']}: u_SOH MC {100*r['u_mc_soh']:.3f} lin {100*r['u_lin_soh']:.3f} pp | "
                  f"u_SOC MC {100*r['u_mc_soc']:.3f} lin {100*r['u_lin_soc']:.3f} pp  ({r['time_s']:.0f}s)")


if __name__ == "__main__":
    main()
