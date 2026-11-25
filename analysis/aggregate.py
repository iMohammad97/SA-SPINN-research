"""Collect the reported numbers from runs/ -> results/numbers.json (values in % or pp)."""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W_TJU = {1: 66, 2: 55, 3: 9}                      # cells per TJU batch
KEYS = ("clean", "u", "stress", "fail_V", "fail_I", "fail_T")


def load(folder):
    rows = []
    for f in glob.glob(str(ROOT / "runs" / folder / "results" / "*.json")):
        r = json.load(open(f))
        rows += r if isinstance(r, list) else [r]
    return rows


def agg(rows, weights=None):
    """op -> metric -> [SOH, SOC] mean over batches (cell-weighted if weights) of the seed mean; plus per batch."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["op"]][r["batch"]].append(r)
    out = {}
    for op, bs in by.items():
        o = {"batch": {}}
        w = np.array([(weights or {}).get(b, 1) for b in sorted(bs)], float)
        w /= w.sum()
        for k in KEYS:
            if k not in next(iter(bs.values()))[0]:
                continue
            pb = np.array([np.mean([r[k] for r in bs[b]], 0) for b in sorted(bs)]) * 100
            o[k] = (w[:, None] * pb).sum(0).round(3).tolist()
            o["batch"][k] = {b: v.round(3).tolist() for b, v in zip(sorted(bs), pb)}
        if "gain" in next(iter(bs.values()))[0]:
            g = np.array([np.mean([r["gain"]["soh"] for r in bs[b]], 0) for b in sorted(bs)])
            o["gain_soh"] = (w[:, None] * g).sum(0).round(3).tolist()
        if "sweep" in next(iter(bs.values()))[0]:
            o["sweep"] = {}
            for lv in next(iter(bs.values()))[0]["sweep"]:
                pb = np.array([np.mean([r["sweep"][lv] for r in bs[b]], 0) for b in sorted(bs)]) * 100
                o["sweep"][lv] = (w[:, None] * pb).sum(0).round(3).tolist()   # u_soh, u_soc, rmse_soh, rmse_soc
        out[op] = o
    return out


N = {"xjtu": {}, "tju": agg(load("s7_eval"), W_TJU)}

def merge(dst, src):
    for op, v in src.items():
        o = dst.setdefault(op, {"batch": {}})
        o["batch"].update(v.pop("batch"))
        o.update(v)


for f in ("s6_eval", "s5_eval", "s6_sweep", "s4_full", "s9_sweep"):          # s9: 50-draw sweep replaces the 10-draw one
    merge(N["xjtu"], agg(load(f)))
merge(N["tju"], agg(load("s9_sweep_tju"), W_TJU))
for name, f in (("gum_abs", "s3_gum"), ("gum_rel", "s5_relt_gum")):
    R = load(f)
    g = {k: round(100 * float(np.mean([r[k] for r in R])), 3) for k in
         ("u_mc_soh", "u_lin_soh", "u_mc_soc", "u_lin_soc", "bias_soh", "bias_soc")}
    for s in ("soh", "soc"):
        g[f"share_{s}"] = {k: round(100 * float(np.mean([r[f"share_{s}"][k] for r in R])), 1) for k in R[0][f"share_{s}"]}
        g[f"ratio_{s}"] = sorted(round(r[f"u_lin_{s}"] / r[f"u_mc_{s}"], 2) for r in R)
    N[name] = g


def flat(rows):
    """Lift fault results and sweep levels to top-level keys."""
    for r in rows:
        r.update(r.pop("faults", {}))
        r.update({"sw" + k: v for k, v in r.pop("sweep", {}).items()})
    return rows


def seeds(rows, weights=None):
    """Five-seed runs (200 draws): op -> metric -> mean and sd over seeds of the batch mean; u_SOH draw interval;
    paired wins of the final model over each baseline per (batch, seed)."""
    by = defaultdict(dict)
    for r in rows:
        by[r["op"]][(r["batch"], r["seed"])] = r
    out = {}
    for op, d in by.items():
        bs = sorted({b for b, _ in d})
        w = np.array([(weights or {}).get(b, 1) for b in bs], float); w /= w.sum()
        ss = sorted({s for _, s in d if all((b, s) in d for b in bs)})
        o = {"seeds": len(ss), "cells": len(d)}
        for k in [k for k in next(iter(d.values())) if isinstance(d[next(iter(d))][k], list) and k != "u_soh_draws"]:
            per = np.array([(w[:, None] * np.array([d[b, s][k] for b in bs])).sum(0) for s in ss]) * 100
            o[k] = per.mean(0).round(3).tolist()
            o[k + "_sd"] = per.std(0, ddof=1).round(3).tolist() if len(ss) > 1 else None
        out[op] = o
    for op in out:
        if op == "final":
            continue
        pairs = [k for k in by[op] if k in by.get("final", {})]
        if not pairs:
            continue
        for k in [k for k in ("u", "stress", "u2") if k in out[op]]:
            ratio = [by[op][q][k][0] / by["final"][q][k][0] for q in pairs]
            out[op]["wins_" + k] = [sum(x > 1 for x in ratio), len(pairs)]
            out[op]["ratio_" + k] = [round(min(ratio), 1), round(float(np.median(ratio)), 1), round(max(ratio), 1)]
    return out


def final_rows(ds):
    """Final size (s12-s14): op -> rows with every test part merged per (batch, seed). XJTU final has temperature dropout."""
    sfx = "_tju" if ds == "tju" else ""
    src = {"final": [("s12_eval", "final" if sfx else "tdrop")],
           "nopen": [("s12_eval", "nopen")] if sfx else [("s13_eval", "none"), ("s14_eval", "nopen")],
           "stat_none": [("s14_eval", "stat_none")], "stat_pen10": [("s14_eval", "stat_pen10")],
           "pen1": [("s13_eval", "pen1")], "pen3": [("s13_eval", "pen3")], "pen30": [("s13_eval", "pen30")]}
    out = {}
    for op, parts in src.items():
        m = {}
        for folder, name in parts:
            for r in flat([r for r in load(folder + sfx) if r["op"] == name]):
                r.update({"iso_" + k: [v, v] for k, v in r.pop("isolate", {}).items()})
                m.setdefault((r["batch"], r["seed"]), {}).update(r, op=op)
        if m:
            out[op] = list(m.values())
    return out


N["s10"], N["s11"], N["s13"], N["iso"] = {}, {}, {}, {}
for ds, sfx, w in (("xjtu", "", None), ("tju", "_tju", W_TJU)):
    F = final_rows(ds)
    new = lambda ops: [r for op in ops for r in F.get(op, [])]
    old = lambda rows: [r for r in rows if r["op"] not in F]
    N["s10"][ds] = seeds(old(load(f"s10_eval{sfx}")) + new(["final"]), w)
    N["s11"][ds] = {part: seeds(old(flat(load(f"s11_{part}{sfx}") + (load(f"s10_eval{sfx}") if part == "ladder" else [])))
                                + new(["final", "nopen"] + (["stat_none", "stat_pen10"] if part != "sweep" else [])), w)
                    for part in ("ladder", "budget", "sweep")}
    L = seeds(new(["nopen", "pen1", "pen3", "final", "pen30"]), w)              # penalty weight 0, 1, 3, 10, 30
    N["s13"][ds] = {lam: L[op] for lam, op in (("0", "nopen"), ("1", "pen1"), ("3", "pen3"), ("10", "final"), ("30", "pen30"))
                    if op in L}
    N["iso"][ds] = {op: {k[4:]: v[op][k][0] for k in v[op] if k.startswith("iso_") and not k.endswith("_sd")}
                    for v in [seeds(new(["final", "stat_pen10"]), w)] for op in v}
    for op, v in seeds(load(f"s15_budgetb{sfx}"), w).items():                   # budget B from the data-sheet voltage
        N["s11"][ds]["budget"].setdefault(op, {}).update({k: x for k, x in v.items() if k.startswith("u2")})
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "numbers.json").write_text(json.dumps(N, indent=1))
if __name__ == "__main__":
    for ds_, ops in (("xjtu", ["mlp", "pinn", "cascade", "ours_none", "ours_oneway", "mean_none", "attention_none",
                               "attention_pen10", "final_g0.3_none", "final_g0.3_pen10", "log_mean", "log_attention"]),
                     ("tju", ["mlp", "pinn", "cascade", "mean_none", "attention_none", "attention_pen10"])):
        for op in ops:
            v = N[ds_][op]
            print(ds_, f"{op:18}", *(f"{k}={v[k]}" for k in KEYS if k in v), "gain", v.get("gain_soh"))
    print(json.dumps(N["gum_abs"]), "\n", json.dumps(N["gum_rel"]))
    for ds_ in ("xjtu", "tju"):
        for op, v in N["s10"][ds_].items():
            print("s10", ds_, f"{op:10}", json.dumps(v))
    for ds_ in ("xjtu", "tju"):
        for part, ops in N["s11"][ds_].items():
            for op, v in ops.items():
                print("s11", ds_, part, f"{op:16}", json.dumps({k: x for k, x in v.items() if not k.endswith("_sd") or k in ("u_sd", "u2_sd")}))
    for ds_ in ("xjtu", "tju"):
        for op, v in N[ds_].items():
            if "sweep" in v:
                print(ds_, f"{op:18}", {k: [round(x, 2) for x in s] for k, s in v["sweep"].items()})
