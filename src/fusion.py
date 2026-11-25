"""Fusion operators as error amplifiers / attenuators.

  swap : two-stage: 8 operators on FROZEN log_mean subnets (step2 gray models), refit fusion params only
  full : fully trained operators (s2_physics log_mean + s4_train others)
Metrics per model: clean RMSE · failed sensor (V/I/T input -> train mean) · beyond-spec stress (10x budget +
white noise) · within-spec MC u (budget A) · gain_j = E Σ_r |∂û/∂ log z_jr| per sensor token.
  robust: models trained against the budget (s5_robust, s5_uaw) -> runs/s5_eval/   baselines: s6_baselines -> runs/s6_eval/
  sweep : budget multiplier sweep (u + RMSE per level) for baselines + ours; full metrics for new models -> runs/s6_sweep/
  tju   : TJU (DS=tju set automatically): full metrics for s7 baselines + s7_main; sweep for baselines + 2 ours -> runs/s7_eval/
  final : final model + 5 baselines, 5 seeds, u from 200 draws -> runs/s10_eval[_tju]/
  extra : 5 seeds, --part ladder (s11 models, same tests as final) | budget (second budget + faults, all models) |
          sweep (budget sweep on the draws of the table) -> runs/s11_<part>[_tju]/
  s12  : selected architecture (+ temperature dropout): all tests and fault isolation -> runs/s12_eval[_tju]/
Usage: python src/fusion.py swap|full|robust|baselines|sweep|tju|final|extra|s12 [--tju] [--workers 6]   Monitor: python src/monitor.py runs/<out>
"""
import argparse, copy, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gum import A, load_raw, perturb  # noqa: E402
from train import logger  # noqa: E402
import ds  # noqa: E402

OPS = ["log_mean", "log_owa", "log_choquet", "log_trim", "log_wmean", "log_attention", "log_ivar", "log_ci"]
SENS = ["V", "I", "T"]                     # token index 0, 1, 2 in both branches
U_DRAWS, S_DRAWS, REFIT = 50, 20, 400
LEVELS, L_DRAWS = [0.5, 1, 2, 4, 10], 10   # sweep: persistent budget × k, no white noise
NOISE_V, NOISE_I = 10e-3, 0.05            # stress white noise: 10 mV, 50 mA (~1 % FS of 5 A)
A2 = torch.tensor([3.0e-3, 10e-3, 5e-3, 2.0], dtype=torch.float64)   # second budget: BQ79616-Q1 voltage (-20 to 65 C), assumed Hall current sensor
B_DRAWS, W_DRAWS = 100, 60                # draws for the second budget and per sweep level (extra)


def predict(m, d, raw, cache, th, g=None):
    """gum.predict + optional white noise (stress)."""
    from features import charge_features
    X = d["X"].double().clone()
    mu, sd = d["xstat"]
    for cell, (t, V, I, T, cyc, rows) in raw.items():
        Vp, Ip, Tp = perturb(V, I, T, th)
        if g is not None:
            Vp = Vp + NOISE_V * torch.randn(len(V), generator=g, dtype=V.dtype)
            Ip = Ip + NOISE_I * torch.randn(len(I), generator=g, dtype=I.dtype)
        for r, (a, e) in rows.items():
            f = charge_features(t[a:e], Vp[a:e], Ip[a:e], Tp[a:e])
            if f is not None:
                X[r, :-1] = (f - mu[:-1]) / sd[:-1]
    Xf = X.float()
    s = m.soh(Xf[d["test"]["soh_rows"]])
    zr, zmu, zsd = cache
    Vz, Iz, Tz = perturb(zr[:, 0], zr[:, 1], zr[:, 2], th)
    if zr.shape[1] == 4:                                      # relative T: offset hits T and T0 alike
        Tz = Tz - (zr[:, 3] + th[3])
    if g is not None:
        Vz = Vz + NOISE_V * torch.randn(len(Vz), generator=g, dtype=Vz.dtype)
        Iz = Iz + NOISE_I * torch.randn(len(Iz), generator=g, dtype=Iz.dtype)
    Z = torch.stack([(Vz - zmu[0]) / zsd[0], (Iz - zmu[1]) / zsd[1], (Tz - zmu[2]) / zsd[2],
                     d["Z"][d["test"]["soc_rows"], 3].double()], 1).float()
    u = m.soc(m.soc_in(Z, m.soh(Xf[d["hi_idx"][d["test"]["soc_rows"]]])))
    return s, u


def rmse(m, d, s, u):
    D = d["test"]
    y, a = D["soh_y"][D["soh_rows"]], D["anchor"][D["soh_rows"]]
    return ((s - y)[a] ** 2).mean().sqrt().item(), ((u - d["soc_y"][D["soc_rows"]]) ** 2).mean().sqrt().item()


def capture(branch):
    """Hook returning list of fusion inputs (tokens)."""
    buf = []
    h = branch.fusion.register_forward_hook(lambda mod, inp, out: buf.append(inp[0]))
    return buf, h


@torch.no_grad()
def token_prec(m, d, raw, cache, draws=10):
    """1 / Var of log-tokens under the budget (MC), floored by the median variance."""
    g = torch.Generator().manual_seed(0)
    L = {"soh": [], "soc": []}
    for _ in range(draws):
        th = (2 * torch.rand(4, generator=g, dtype=torch.float64) - 1) * A
        bs, hs = capture(m.soh); bc, hc = capture(m.soc)
        predict(m, d, raw, cache, th)
        hs.remove(); hc.remove()
        L["soh"].append(bs[0]); L["soc"].append(bc[-1])           # soc branch: last call = SOC tokens
    out = {}
    for k, z in L.items():
        l = torch.log(torch.nn.functional.softplus(torch.stack(z)) + 1e-4)
        v = l.var(0).mean(0)                                        # [n, p]
        out[k] = 1.0 / (v + v.median())
    return out


def refit(m, d, steps=REFIT):
    from train import BS_SOC, BS_SOH, BETA
    for p in m.parameters():
        p.requires_grad_(False)
    ps = [p for b in (m.soh, m.soc) for p in b.fusion.parameters()]
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ps, 2e-3)
    m.train()
    for _ in range(steps):
        loss, _ = m.losses(d, "train", BS_SOC, BS_SOH, 1e-3, BETA)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()


def gains(m, d):
    """E_rows Σ_r |∂û/∂ log z_jr| per token j (clean test data), both branches."""
    D = d["test"]
    out = {}
    for name, br, x in (("soh", m.soh, d["X"][D["soh_rows"]]),):
        z = torch.stack([s(x[:, gi]) for s, gi in zip(br.subnets, br.groups)], 1).detach().requires_grad_(True)
        br.fusion(z, br.token_var(x) if br.fusion.kind == "log_uaw" else None).sum().backward()
        dl_dz = torch.sigmoid(z) / (torch.nn.functional.softplus(z) + 1e-4)
        out[name] = (z.grad / dl_dz).abs().sum(2).mean(0).tolist()
    r = D["soc_rows"][torch.randperm(len(D["soc_rows"]))[:20000]]
    with torch.no_grad():
        sh = m.soh(d["X"][d["hi_idx"][r]])
    x = m.soc_in(d["Z"][r], sh)
    br = m.soc
    z = torch.stack([s(x[:, gi]) for s, gi in zip(br.subnets, br.groups)], 1).detach().requires_grad_(True)
    br.fusion(z, br.token_var(x) if br.fusion.kind == "log_uaw" else None).sum().backward()
    dl_dz = torch.sigmoid(z) / (torch.nn.functional.softplus(z) + 1e-4)
    out["soc"] = (z.grad / dl_dz).abs().sum(2).mean(0).tolist()
    return out


def spread(m, d, raw, cache, tick, nu, a, g):
    """u (SOH, SOC) over nu budget draws of half-widths a, and the bootstrap 95 % interval of u_SOH over the draws."""
    S, u1, u2 = [], 0.0, 0.0                                      # SOC: running moments (large row count)
    for i in range(nu):
        th = (2 * torch.rand(4, generator=g, dtype=torch.float64) - 1) * a
        s, u = predict(m, d, raw, cache, th); S.append(s)
        u1, u2 = u1 + u.double(), u2 + u.double() ** 2
        tick()
    S = torch.stack(S)
    var_u = ((u2 - u1 ** 2 / nu) / (nu - 1)).clamp_min(0)
    b = torch.Generator().manual_seed(3)
    ub = [S[torch.randint(nu, (nu,), generator=b)].std(0).mean().item() for _ in range(500)]
    return (S.std(0).mean().item(), var_u.sqrt().mean().item()), np.percentile(ub, [2.5, 97.5]).tolist()


KINDS = {"drift_V": lambda f, sg: (sg * 10 * A[0] * f, 0 * f, 0 * f, 0 * f),
         "drift_I": lambda f, sg: (0 * f, sg * 10 * A[1] * f, 0 * f, 0 * f),
         "step_gain": lambda f, sg: (0 * f, 0 * f, sg * 10 * A[2] * (f > 0.5), 0 * f),
         "stuck_T": None}
FAULTY = {"drift_V": 0, "drift_I": 1, "step_gain": 1, "stuck_T": 2}   # index of the faulty sensor (V, I, T)


def life(d):
    hi = d["hi"]
    return torch.tensor((hi.cycle / hi.groupby("cell").cycle.transform("max")).values, dtype=torch.float64)


def fault_inputs(d, raw, cache, name, sg):
    """Scaled HI matrix (all rows) and SOC inputs (test rows) under one fault and sign."""
    from features import charge_features
    fn, frac = KINDS[name], life(d)
    mu, sd = d["xstat"]
    zr, zmu, zsd = cache
    rs = d["test"]["soc_rows"]
    X = d["X"].double().clone()
    for cell, (t, V, I, T, cyc, rows) in raw.items():
        for r, (a, e) in rows.items():
            if fn:
                Vp, Ip, Tp = perturb(V[a:e], I[a:e], T[a:e], fn(frac[r], sg))
            else:
                Vp, Ip, Tp = V[a:e], I[a:e], T[:1].expand(e - a)
            f = charge_features(t[a:e], Vp, Ip, Tp)
            if f is not None:
                X[r, :-1] = (f - mu[:-1]) / sd[:-1]
    if fn:
        Vz, Iz, Tz = perturb(zr[:, 0], zr[:, 1], zr[:, 2], fn(frac[d["hi_idx"][rs]], sg))
    else:
        Vz, Iz, Tz = zr[:, 0], zr[:, 1], zr[:, 3] if zr.shape[1] == 4 else zr[:, 2]
    if zr.shape[1] == 4:
        Tz = Tz - zr[:, 3]
    Z = torch.stack([(Vz - zmu[0]) / zsd[0], (Iz - zmu[1]) / zsd[1], (Tz - zmu[2]) / zsd[2],
                     d["Z"][rs, 3].double()], 1).float()
    return X.float(), Z


def faults(m, d, raw, cache, tick):
    """SOH and SOC RMSE under faults that change over the cell life (frac = cycle / last cycle of the cell):
    drift_V, drift_I: offset growing linearly to 10x the budget; step_gain: current gain error of 10x the budget in the
    second half of life; stuck_T: temperature frozen at its first reading. Mean over both signs."""
    rs = d["test"]["soc_rows"]
    out = {}
    for name, fn in KINDS.items():
        E = []
        for sg in ((1, -1) if fn else (1,)):
            Xf, Z = fault_inputs(d, raw, cache, name, sg)
            s = m.soh(Xf[d["test"]["soh_rows"]])
            E.append(rmse(m, d, s, m.soc(m.soc_in(Z, m.soh(Xf[d["hi_idx"][rs]])))))
            tick()
        out[name] = np.mean(E, 0).tolist()
    return out


def isolate(m, d, raw, cache, tick):
    """Which sensor is faulty, without a clean reference? Each fusion input token (before attention) is mapped to
    SOH by least squares on training cells, giving one estimate per token. Under a fault, the token whose estimate
    departs most from the median of all tokens (in units of its training error) is blamed, per test cell and sign,
    over the second half of life. Credit = share of the faulty sensor among the indicators of that token
    (1 for a sensor-aligned token, less for a token that mixes sensors). Mean credit per fault."""
    from data import HI
    tag = lambda n: {"v": 0, "i": 1, "t": 2}.get(n[0], 1 if n in ("cc_q", "cv_q") else 3)   # 3 = clock
    share = torch.zeros(len(m.soh.groups), 4)
    for j, g in enumerate(m.soh.groups):
        for k in g:
            if HI[k] != "cycle":
                share[j, tag(HI[k])] += 1
    share = share / share.sum(1, keepdim=True)

    def tokens(X, rows):
        buf, h = capture(m.soh)
        with torch.no_grad():
            m.soh(X[rows])
        h.remove()
        z = buf[0].double()
        return torch.cat([z, torch.ones(*z.shape[:2], 1, dtype=z.dtype)], 2)

    tr, rows = d["train"]["soh_rows"], d["test"]["soh_rows"]
    Zt, y = tokens(d["X"], tr), d["train"]["soh_y"][tr].double()[:, None]
    coef = [torch.linalg.lstsq(Zt[:, j], y).solution for j in range(Zt.shape[1])]
    est = lambda Z: torch.stack([(Z[:, j] @ coef[j]).squeeze(1) for j in range(Z.shape[1])], 1)
    sig = (est(Zt) - y).std(0)
    late = (life(d)[rows] > 0.5).numpy()
    cell = d["hi"].cell.values[rows.numpy()]
    out = {}
    for name, fn in KINDS.items():
        if ds.TJU and FAULTY[name] == 2:                      # TJU has no temperature indicators
            continue
        hits = []
        for sg in ((1, -1) if fn else (1,)):
            o = est(tokens(fault_inputs(d, raw, cache, name, sg)[0], rows))
            dev = (o - o.median(1, keepdim=True).values).abs() / sig
            for cl in np.unique(cell):
                k = torch.from_numpy((cell == cl) & late)
                if k.any():
                    hits.append(float(share[int(dev[k].mean(0).argmax()), FAULTY[name]]))
            tick()
        out[name] = float(np.mean(hits))
    return out


def metrics(m, d, raw, cache, st, tick, nu=U_DRAWS):
    from model import SOH_GROUPS
    res = {}
    with torch.no_grad():
        zero = torch.zeros(4, dtype=torch.float64)
        res["clean"] = rmse(m, d, *predict(m, d, raw, cache, zero))
        for j, sname in enumerate(SENS):                          # failed sensor -> train mean (z = 0)
            d2 = dict(d, X=d["X"].clone(), Z=d["Z"].clone())
            d2["X"][:, SOH_GROUPS["sensor"][j]] = 0.0
            d2["Z"][:, j] = 0.0
            res[f"fail_{sname}"] = m.evaluate(d2, "test")
        g = torch.Generator().manual_seed(1)
        res["u"], ci = spread(m, d, raw, cache, tick, nu, A, g)   # within spec
        if nu > U_DRAWS:
            res["u_soh_ci"], res["u_soh_draws"] = ci, nu
        E = []
        for i in range(S_DRAWS):                                  # beyond spec (stress)
            th = (2 * torch.rand(4, generator=g, dtype=torch.float64) - 1) * 10 * A
            E.append(rmse(m, d, *predict(m, d, raw, cache, th, g)))
            tick()
        res["stress"] = tuple(np.mean(E, 0).tolist())
    res["gain"] = gains(m, d)
    return res


def sweep(m, d, raw, cache, tick, n=L_DRAWS, same=False):
    """Per budget multiplier k: (u_SOH, u_SOC, RMSE_SOH, RMSE_SOC); same draws for every model.
    same: every level uses the first n unit draws of the table (metrics), scaled by k."""
    out, g = {}, torch.Generator().manual_seed(2)
    with torch.no_grad():
        for k in LEVELS:
            if same:
                g = torch.Generator().manual_seed(1)
            S, U, E = [], [], []
            for _ in range(n):
                th = (2 * torch.rand(4, generator=g, dtype=torch.float64) - 1) * k * A
                s, u = predict(m, d, raw, cache, th); S.append(s); U.append(u); E.append(rmse(m, d, s, u))
                tick()
            out[str(k)] = [torch.stack(S).std(0).mean().item(), torch.stack(U).std(0).mean().item(),
                           *np.mean(E, 0).tolist()]
    return out


def job(cfg):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    from data import load_batch
    from model import CoupledSPINN
    from data import add_uncertainty
    d = add_uncertainty(load_batch(cfg["batch"], rel_t=cfg["relt"]), cfg["batch"])
    if cfg.get("part") == "uval":                         # architecture choice: evaluate on the validation cells
        d["test"], d["cells"] = d["val"], dict(d["cells"], test=d["cells"]["val"])
    raw = load_raw(d)
    c = np.load(ds.CACHE)
    sel = np.isin(c["cell"], d["cells"]["test"])
    zr = torch.tensor(np.stack([c["V"][sel], c["I"][sel], c["T"][sel]], 1), dtype=torch.float64)
    if cfg["relt"]:
        zr = torch.cat([zr, d["T0"][d["test"]["soc_rows"]][:, None]], 1)
    cache = (zr, *(torch.tensor(x[:3], dtype=torch.float64) for x in d["zstat"]))
    out = Path(cfg["dir"]); st = out / "status" / f"{cfg['id']}.json"
    nu = cfg.get("u_draws", U_DRAWS)
    part = cfg.get("part")
    total = {"budget": B_DRAWS + 7, "budgetb": B_DRAWS, "sweep": len(LEVELS) * W_DRAWS, "isolate": 7, "uval": U_DRAWS}[part] if part in ("budget", "budgetb", "sweep", "isolate", "uval") else len(cfg["ops"]) * (nu + S_DRAWS) if cfg["mode"] != "sweep" else         len(cfg["ops"]) * len(LEVELS) * cfg.get("draws", L_DRAWS) + len(cfg.get("full", [])) * (U_DRAWS + S_DRAWS)
    n = [0]; t0 = time.time()

    def tick():
        n[0] += 1
        if n[0] % 5 == 0:
            st.write_text(json.dumps(dict(id=cfg["id"], epoch=n[0], epochs=total, elapsed=time.time() - t0, done=False)))

    if cfg["mode"] == "swap":
        base = CoupledSPINN("sensor", "log_mean", "gray")
        base.load_state_dict(torch.load(cfg["base"]))
        prec = token_prec(base, d, raw, cache)
    rows = []
    for op in cfg["ops"]:
        if cfg["mode"] == "swap":
            m = CoupledSPINN("sensor", op, "gray")
            m.load_state_dict(base.state_dict(), strict=False)    # subnets + log_mean fusion; attn new
            m.soh.fusion.prec, m.soc.fusion.prec = prec["soh"], prec["soc"]
            refit(m, d)
        else:                                             # models: name -> (fusion, path)
            fus, path, *kw = cfg["models"][op]
            m = CoupledSPINN(**(kw[0] if kw else dict(fusion=fus)))
            m.load_state_dict(torch.load(path))
            m.set_sigma(d)
        m.eval()
        if part == "budget":
            with torch.no_grad():
                u, ci = spread(m, d, raw, cache, tick, B_DRAWS, A2, torch.Generator().manual_seed(1))
                r = dict(u2=u, u2_soh_ci=ci, faults=faults(m, d, raw, cache, tick))
        elif part == "budgetb":
            with torch.no_grad():
                u, ci = spread(m, d, raw, cache, tick, B_DRAWS, A2, torch.Generator().manual_seed(1))
                r = dict(u2=u, u2_soh_ci=ci)
        elif part == "sweep":
            r = dict(sweep=sweep(m, d, raw, cache, tick, W_DRAWS, same=True))
        elif part == "uval":
            with torch.no_grad():
                r = dict(zip(("u", "u_ci"), spread(m, d, raw, cache, tick, U_DRAWS, A, torch.Generator().manual_seed(1))))
        elif part == "isolate":
            r = dict(isolate=isolate(m, d, raw, cache, tick))
        elif cfg["mode"] == "sweep":
            r = dict(sweep=sweep(m, d, raw, cache, tick, cfg.get("draws", L_DRAWS)))
            if op in cfg["full"]:
                r.update(metrics(m, d, raw, cache, st, tick))
        else:
            r = metrics(m, d, raw, cache, st, tick, nu)
        rows.append(dict(op=op, **{k: v for k, v in cfg.items() if k in ("batch", "seed", "mode")}, **r))
    (out / "results" / f"{cfg['id']}.json").write_text(json.dumps(rows))
    st.write_text(json.dumps(dict(id=cfg["id"], epoch=total, epochs=total, elapsed=time.time() - t0, done=True)))
    return cfg["id"], time.time() - t0


def tju(a):
    """One job per (batch, seed, model): full metrics for all; budget sweep for baselines + mean_none + attention_pen10."""
    import os
    os.environ["DS"] = "tju"                              # spawned workers re-import ds
    from train import PRESETS
    out = Path("runs/s9_sweep_tju" if a.draws else "runs/s7_eval")
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    SW = ("mlp", "pinn", "cascade", "mean_none", "attention_pen10")
    cfgs = []
    for b in (1, 2, 3):
        for s in (0, 1):
            models = {}
            for k in ("mlp", "pinn", "cascade"):
                p = PRESETS[k]; al = 1e-3 if p["alpha"] is None else p["alpha"]
                models[k] = (p["fusion"], f"runs/s7_baselines/models/{k}_b{b}_all_in_one_mean_{p['physics']}_"
                             f"a{al:g}_s{s}_relT.pt", {q: p[q] for q in ("grouping", "fusion", "physics", "couple", "h", "depth")})
            for f in ("log_mean", "log_attention"):
                for gm in (0, 0.3, 1):
                    for rb in ("none", "pen10"):
                        models[f"{f[4:]}{'' if gm == 0 else f'_g{gm:g}'}_{rb}"] = (f, f"runs/s7_main/models/b{b}_sensor_{f}_gray_"
                            f"a0.001_s{s}" + ("" if gm == 0 else f"_g{gm:g}") + "_relT" + ("" if rb == "none" else "_pen10") + ".pt")
            if a.draws:
                models = {k: models[k] for k in ("mlp", "pinn", "cascade", "attention_none", "attention_pen10")}
            for op, spec in models.items():
                i = f"b{b}_s{s}_{op}"
                if not (out / "results" / f"{i}.json").exists():
                    cfgs.append(dict(id=i, batch=b, seed=s, mode="sweep" if op in SW or a.draws else "full", models={op: spec},
                                     ops=[op], full=[] if a.draws else [op], relt=True, dir=str(out),
                                     draws=a.draws or L_DRAWS))
                    (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: (c["batch"] != 1, c["mode"] != "sweep"))   # big batch 1 + sweeps first
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def final(a):
    """Final model + five baselines, seeds 0-4, u from --draws draws (default 200) -> runs/s10_eval[_tju].
    Jobs whose model file does not exist yet are skipped, so the command can be rerun after training."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    from train import PRESETS
    out = Path("runs/s10_eval_tju" if tju else "runs/s10_eval")
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    old, new = ("s7_main", "s10_tju") if tju else ("s6_final", "s10_xjtu")
    old_b, new_b = ("s7_baselines", "s10_tju_base") if tju else ("s6_baselines", "s10_xjtu_base")
    cfgs = []
    for b in ((1, 2, 3) if tju else range(1, 7)):
        for s in range(5):
            mid = f"b{b}_sensor_log_attention_gray_a0.001_s{s}" + ("" if tju else "_g0.3") + "_relT_pen10.pt"
            models = {"final": ("log_attention", f"runs/{old if s < 2 else new}/models/{mid}")}
            for k in ("mlp", "pinn", "cascade", "autospinn", "pinn_wide"):
                p = PRESETS[k]; al = 1e-3 if p["alpha"] is None else p["alpha"]
                folder = old_b if s < 2 and k in ("mlp", "pinn", "cascade") else new_b
                models[k] = (p["fusion"], f"runs/{folder}/models/{k}_b{b}_{p['grouping']}_{p['fusion']}_{p['physics']}_"
                             f"a{al:g}_s{s}_relT.pt", {q: p[q] for q in ("grouping", "fusion", "physics", "couple", "h", "depth") if q in p})
            for op, spec in models.items():
                i = f"b{b}_s{s}_{op}"
                if Path(spec[1]).exists() and not (out / "results" / f"{i}.json").exists():
                    cfgs.append(dict(id=i, batch=b, seed=s, mode="full", models={op: spec}, ops=[op], full=[op],
                                     relt=True, dir=str(out), u_draws=a.draws or 200))
                    (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: c["batch"] != 1)                     # largest batch first
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def extra(a):
    """Five seeds. ladder: s11 models with the tests of `final`; budget: second budget and faults for all models;
    sweep: budget sweep for five models. Jobs whose model file does not exist yet are skipped (rerun after training)."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    from train import PRESETS
    ds_ = "tju" if tju else "xjtu"
    out = Path(f"runs/s11_{a.part}" + ("_tju" if tju else ""))
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    old, old_b = ("s7_main", "s7_baselines") if tju else ("s6_final", "s6_baselines")
    gm = "" if tju else "_g0.3"
    keep = {"ladder": ("nopen", "stat_none", "stat_pen10", "pinn_pen10", "autospinn_pen10"), "budget": None,
            "sweep": ("final", "nopen", "mlp", "pinn", "autospinn"),
            "isolate": ("final", "stat_pen10", "pinn")}[a.part]
    cfgs = []
    for b in ((1, 2, 3) if tju else range(1, 7)):
        for s in range(5):
            ours = f"b{b}_sensor_log_attention_gray_a0.001_s{s}{gm}_relT"
            models = {"final": ("log_attention", f"runs/{old if s < 2 else f's10_{ds_}'}/models/{ours}_pen10.pt"),
                      "nopen": ("log_attention", f"runs/{old if s < 2 else f's11_{ds_}_nopen'}/models/{ours}.pt")}
            for rb in ("none", "pen10"):
                models[f"stat_{rb}"] = ("log_attention", f"runs/s11_{ds_}_stat/models/b{b}_statistic_log_attention_gray_a0.001_"
                                        f"s{s}{gm}_relT" + ("" if rb == "none" else "_pen10") + ".pt",
                                        dict(grouping="statistic", fusion="log_attention", physics="gray"))
            for k in ("mlp", "pinn", "cascade", "autospinn", "pinn_wide"):
                p = PRESETS[k]
                kw = {q: p[q] for q in ("grouping", "fusion", "physics", "couple", "h", "depth") if q in p}
                name = f"{k}_b{b}_{p['grouping']}_{p['fusion']}_{p['physics']}_a{1e-3 if p['alpha'] is None else p['alpha']:g}_s{s}_relT"
                folder = old_b if s < 2 and k in ("mlp", "pinn", "cascade") else f"s10_{ds_}_base"
                models[k] = (p["fusion"], f"runs/{folder}/models/{name}.pt", kw)
                if k in ("pinn", "autospinn"):
                    models[f"{k}_pen10"] = (p["fusion"], f"runs/s11_{ds_}_plug/models/{name}_pen10.pt", kw)
            for op, spec in models.items():
                i = f"b{b}_s{s}_{op}"
                if (keep is None or op in keep) and Path(spec[1]).exists() and not (out / "results" / f"{i}.json").exists():
                    cfgs.append(dict(id=i, batch=b, seed=s, mode="full", part=a.part, models={op: spec}, ops=[op], full=[op],
                                     relt=True, dir=str(out), u_draws=200))
                    (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: c["batch"] != 1)                     # largest batch first
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def s12(a):
    """Selected architecture (s12_*_final with and without penalty; XJTU also temperature dropout), five seeds:
    every test of the paper (full, budget, sweep) and fault isolation, which also runs on the statistic-grouped
    network with penalty -> runs/s12_eval[_tju]. Jobs whose model file does not exist yet are skipped."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    ds_, gm, sfx = ("tju", "", "_h32_d3") if tju else ("xjtu", "_g0.3", "_h32")
    kw = dict(grouping="sensor", fusion="log_attention", physics="gray", h=32, depth=3 if tju else 2)
    out = Path("runs/s12_eval" + ("_tju" if tju else ""))
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    cfgs = []
    for b in ((1, 2, 3) if tju else range(1, 7)):
        for s in range(5):
            base = f"b{b}_sensor_log_attention_gray_a0.001_s{s}{gm}_relT"
            models = {"final": f"runs/s12_{ds_}_final/models/{base}_pen10{sfx}.pt",
                      "nopen": f"runs/s12_{ds_}_final/models/{base}{sfx}.pt"}
            if not tju:
                models["tdrop"] = f"runs/s12_xjtu_tdrop/models/{base}_pen10{sfx}_td0.1.pt"
            jobs = [(op, (kw["fusion"], path, kw), part) for op, path in models.items()
                    for part in ("full", "budget", "sweep", "isolate")]
            jobs.append(("stat_pen10", ("log_attention", f"runs/s11_{ds_}_stat/models/b{b}_statistic_log_attention_gray_"
                                        f"a0.001_s{s}{gm}_relT_pen10.pt",
                                        dict(grouping="statistic", fusion="log_attention", physics="gray")), "isolate"))
            for op, spec, part in jobs:
                i = f"b{b}_s{s}_{op}_{part}"
                if Path(spec[1]).exists() and not (out / "results" / f"{i}.json").exists():
                    cfgs.append(dict(id=i, batch=b, seed=s, mode="full", part=None if part == "full" else part,
                                     models={op: spec}, ops=[op], full=[op], relt=True, dir=str(out), u_draws=200))
                    (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: (c["batch"] != 1, c["part"] is not None))   # largest batch and slowest part first
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def s12val(a):
    """Robustness-aware architecture choice: u_SOH on the validation cells (50 draws) for every searched model
    (s12_*_arch, seed 0) -> runs/s12_uval[_tju]. Score = val SOH RMSE + val SOC RMSE + val u_SOH."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    src = Path("runs/s12_tju_arch" if tju else "runs/s12_xjtu_arch")
    out = Path("runs/s12_uval" + ("_tju" if tju else ""))
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    cfgs = []
    for f in sorted((src / "results").glob("*.json")):
        c = json.loads(f.read_text())
        kw = dict(grouping="sensor", fusion="log_attention", physics="gray", h=c.get("h", 16), depth=c.get("depth", 2))
        i = c["id"]
        if not (out / "results" / f"{i}.json").exists():
            cfgs.append(dict(id=i, batch=c["batch"], seed=0, mode="full", part="uval", relt=True, dir=str(out),
                             models={"m": ("log_attention", str(src / "models" / f"{i}.pt"), kw)}, ops=["m"], full=["m"]))
            (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def s13(a):
    """Penalty-weight sensitivity of the final model: every s13_*_lambda model, tests of `final` -> runs/s13_eval[_tju]."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    src = Path("runs/s13_tju_lambda" if tju else "runs/s13_xjtu_lambda")
    out = Path("runs/s13_eval" + ("_tju" if tju else ""))
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    cfgs = []
    for f in sorted((src / "results").glob("*.json")):
        c = json.loads(f.read_text())
        kw = dict(grouping="sensor", fusion="log_attention", physics="gray", h=c.get("h", 16), depth=c.get("depth", 2))
        i = f"b{c['batch']}_s{c['seed']}_{c.get('robust', 'none')}"
        if not (out / "results" / f"{i}.json").exists():
            cfgs.append(dict(id=i, batch=c["batch"], seed=c["seed"], mode="full", relt=True, dir=str(out), u_draws=200,
                             models={c.get("robust", "none"): ("log_attention", str(src / "models" / f"{c['id']}.pt"), kw)},
                             ops=[c.get("robust", "none")], full=[c.get("robust", "none")]))
            (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: c["batch"] != 1)
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def s14(a):
    """Statistic-grouped ablation at the final size (s14_*_stat): full tests for both, budget and isolation with the
    penalty; XJTU λ=0 models (s13, dropout) get budget and sweep as `nopen` -> runs/s14_eval[_tju]."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    src = Path("runs/s14_tju_stat" if tju else "runs/s14_xjtu_stat")
    out = Path("runs/s14_eval" + ("_tju" if tju else ""))
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    cfgs = []
    nopen = [] if tju else [f for f in sorted(Path("runs/s13_xjtu_lambda/results").glob("*.json"))
                            if json.loads(f.read_text()).get("robust", "none") == "none"]   # XJTU λ=0: budget and sweep
    for f in sorted((src / "results").glob("*.json")) + nopen:
        c = json.loads(f.read_text())
        g = c.get("grouping", "sensor")
        op = "stat_" + c.get("robust", "none") if g == "statistic" else "nopen"
        kw = dict(grouping=g, fusion="log_attention", physics="gray", h=c["h"], depth=c["depth"])
        src = f.parents[1]
        for part in {"stat_pen10": ("full", "budget", "isolate"), "stat_none": ("full",), "nopen": ("budget", "sweep")}[op]:
            i = f"b{c['batch']}_s{c['seed']}_{op}_{part}"
            if not (out / "results" / f"{i}.json").exists():
                cfgs.append(dict(id=i, batch=c["batch"], seed=c["seed"], mode="full", part=None if part == "full" else part,
                                 models={op: ("log_attention", str(src / "models" / f"{c['id']}.pt"), kw)},
                                 ops=[op], full=[op], relt=True, dir=str(out), u_draws=200))
                (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: (c["batch"] != 1, c["part"] is not None))
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def s15(a):
    """Budget B only (u2) for the models of the fault table, seeds 0-4 -> runs/s15_budgetb[_tju]."""
    import os
    tju = a.tju
    if tju:
        os.environ["DS"] = "tju"
    from train import PRESETS
    ds_, gm = ("tju", "") if tju else ("xjtu", "_g0.3")
    out = Path("runs/s15_budgetb" + ("_tju" if tju else ""))
    for s_ in ("status", "results"):
        (out / s_).mkdir(parents=True, exist_ok=True)
    old_b = "s7_baselines" if tju else "s6_baselines"
    kw = dict(grouping="sensor", fusion="log_attention", physics="gray", h=32, depth=3 if tju else 2)
    cfgs = []
    for b in ((1, 2, 3) if tju else range(1, 7)):
        for s in range(5):
            base = f"b{b}_sensor_log_attention_gray_a0.001_s{s}{gm}_relT_pen10"
            models = {"final": ("log_attention", f"runs/s12_tju_final/models/{base}_h32_d3.pt" if tju else
                                f"runs/s12_xjtu_tdrop/models/{base}_h32_td0.1.pt", kw)}
            for k in ("mlp", "pinn", "cascade", "autospinn", "pinn_wide"):
                p = PRESETS[k]
                pk = {q: p[q] for q in ("grouping", "fusion", "physics", "couple", "h", "depth") if q in p}
                name = f"{k}_b{b}_{p['grouping']}_{p['fusion']}_{p['physics']}_a{1e-3 if p['alpha'] is None else p['alpha']:g}_s{s}_relT"
                folder = old_b if s < 2 and k in ("mlp", "pinn", "cascade") else f"s10_{ds_}_base"
                models[k] = (p["fusion"], f"runs/{folder}/models/{name}.pt", pk)
                if k == "autospinn":
                    models["autospinn_pen10"] = (p["fusion"], f"runs/s11_{ds_}_plug/models/{name}_pen10.pt", pk)
            for op, spec in models.items():
                i = f"b{b}_s{s}_{op}"
                assert Path(spec[1]).exists(), spec[1]
                if not (out / "results" / f"{i}.json").exists():
                    cfgs.append(dict(id=i, batch=b, seed=s, mode="full", part="budgetb", models={op: spec}, ops=[op],
                                     full=[op], relt=True, dir=str(out)))
                    (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    cfgs.sort(key=lambda c: c["batch"] != 1)
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["swap", "full", "robust", "baselines", "sweep", "tju", "final", "extra", "s12", "s12val", "s13", "s14", "s15"])
    ap.add_argument("--tju", action="store_true", help="final, extra: TJU instead of XJTU")
    ap.add_argument("--part", choices=["ladder", "budget", "sweep", "isolate"], default="ladder", help="extra: which evaluation")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--draws", type=int, default=0, help="sweep / tju: draws per level, final models only -> runs/s9_sweep[_tju]")
    a = ap.parse_args()
    if a.mode == "tju":
        return tju(a)
    if a.mode == "final":
        return final(a)
    if a.mode == "extra":
        return extra(a)
    if a.mode == "s12":
        return s12(a)
    if a.mode == "s12val":
        return s12val(a)
    if a.mode == "s13":
        return s13(a)
    if a.mode == "s14":
        return s14(a)
    if a.mode == "s15":
        return s15(a)
    out = Path({"robust": "runs/s5_eval", "baselines": "runs/s6_eval", "sweep": "runs/s6_sweep"}.get(a.mode, f"runs/s4_{a.mode}"))
    if a.draws:
        out = Path("runs/s9_sweep")
    for s in ("status", "results"):
        (out / s).mkdir(parents=True, exist_ok=True)
    cfgs = []
    for b in range(1, 7):
        for s in (0, 1):
            i = f"b{b}_s{s}"
            base = f"runs/s2_physics/models/b{b}_sensor_log_mean_gray_a0.001_s{s}.pt"
            if a.mode == "robust":                       # relT base vs penalty / augmentation / uaw fusion
                models = {f"{f[4:]}_{rb}": (f, f"runs/s5_robust/models/b{b}_sensor_{f}_gray_a0.001_s{s}_relT"
                                               + ("" if rb == "none" else f"_{rb}") + ".pt")
                          for f in ("log_mean", "log_attention") for rb in ("none", "pen1", "pen10", "aug")}
                models["uaw_none"] = ("log_uaw", f"runs/s5_uaw/models/b{b}_sensor_log_uaw_gray_a0.001_s{s}_relT.pt")
            full = []
            if a.mode == "sweep":                        # baselines + ours + gamma=1 + s6_final candidates
                from train import PRESETS
                models = {}
                for k in ("mlp", "pinn", "cascade"):
                    p = PRESETS[k]; al = 1e-3 if p["alpha"] is None else p["alpha"]
                    models[k] = (p["fusion"], f"runs/s6_baselines/models/{k}_b{b}_all_in_one_mean_{p['physics']}_"
                                 f"a{al:g}_s{s}_relT.pt", {q: p[q] for q in ("grouping", "fusion", "physics", "couple", "h", "depth")})
                r5 = f"runs/s5_robust/models/b{b}_sensor_{{}}_gray_a0.001_s{s}_relT{{}}.pt"
                models["mean_none"] = ("log_mean", r5.format("log_mean", ""))
                for rb in ("none", "pen10", "aug"):
                    models[f"attention_{rb}"] = ("log_attention", r5.format("log_attention", "" if rb == "none" else f"_{rb}"))
                models["mean_g1"] = ("log_mean", f"runs/s2_gamma/models/b{b}_sensor_log_mean_gray_a0.001_s{s}_g1_relT.pt")
                for gm in ("0.3", "1"):
                    for rb in ("none", "pen10"):
                        models[f"final_g{gm}_{rb}"] = ("log_attention", f"runs/s6_final/models/b{b}_sensor_log_attention_"
                                                       f"gray_a0.001_s{s}_g{gm}_relT" + ("" if rb == "none" else "_pen10") + ".pt")
                full = ["mean_g1"] + [k for k in models if k.startswith("final")]
                if a.draws:
                    models = {k: models[k] for k in ("mlp", "pinn", "cascade", "final_g0.3_none", "final_g0.3_pen10")}
                    full = []
            elif a.mode == "baselines":                  # s6: literature baselines (ours = s5_eval rows)
                from train import PRESETS
                models = {}
                for k, p in PRESETS.items():
                    al = 1e-3 if p["alpha"] is None else p["alpha"]
                    models[k] = (p["fusion"], f"runs/s6_baselines/models/{k}_b{b}_{p['grouping']}_{p['fusion']}_"
                                 f"{p['physics']}_a{al:g}_s{s}_relT.pt",
                                 {q: p[q] for q in ("grouping", "fusion", "physics", "couple", "h", "depth") if q in p})
            else:
                models = {op: (op, base if op == "log_mean" else
                               f"runs/s4_train/models/b{b}_sensor_{op}_gray_a0.001_s{s}.pt") for op in OPS[:6]}
            ops = OPS if a.mode == "swap" else list(models)
            if not (out / "results" / f"{i}.json").exists():
                cfgs.append(dict(id=i, batch=b, seed=s, mode=a.mode, base=base, models=models, ops=ops, full=full,
                                 draws=a.draws or L_DRAWS,
                                 relt=a.mode in ("robust", "baselines", "sweep"), dir=str(out)))
                (out / "status" / f"{i}.json").write_text(json.dumps(dict(id=i, epoch=0, epochs=1, elapsed=0, done=False)))
    say = logger(out)
    say(f"{len(cfgs)} jobs -> {out}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, t in ex.map(job, cfgs):
            say(f"done {i} ({t:.0f}s)")


if __name__ == "__main__":
    main()
