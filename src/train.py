"""Train coupled SPINN configs in parallel. Usage: python src/train.py --grid <name>
Run folders: runs/s<step>_<what>/{status,results,models,log.txt}   Monitor: python src/monitor.py runs/<name>
"""
import argparse, itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import protocols

FINAL = dict(batch=[0], grouping=["sensor"], fusion=["log_attention"], physics=["gray"], relt=[True],
             robust=["pen10"], alpha=[1e-3], seed=[0, 1], epochs=[100])          # selected per dataset on validation

GRIDS = {
    "s1_alpha": dict(batch=[1], grouping=["sensor"], fusion=["mean"], physics=["gray"],
                  alpha=[1e-3, 1e-2, 1e-1], seed=[0], epochs=[30]),
    "s1_grouping": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor", "statistic", "all_in_one"], fusion=["mean"],
                  physics=["gray"], alpha=[None], seed=[0, 1], epochs=[100]),
    "s1_logfusion": dict(batch=[1, 4, 5], grouping=["sensor"],
                    fusion=["log_mean", "log_owa", "log_choquet", "log_trim", "log_wmean", "log_attention"],
                    physics=["gray"], alpha=[None], seed=[0], epochs=[100]),
    "s1_product": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["product"],
                   physics=["gray"], alpha=[None], seed=[0, 1], epochs=[100]),
    "s2_physics": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_mean"],
                  physics=["cc", "gray", "learned"], corrupt=["none", "input", "prior", "both"],
                  alpha=[None], seed=[0, 1], epochs=[100]),
    "s2_scarce": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_mean"],
                   physics=["cc", "gray", "learned"], corrupt=["none", "input", "prior", "both"],
                   labels=["frac1", "full"], alpha=[1e-1], seed=[0], epochs=[100]),
    "s2_fpenalty": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_mean"], physics=["gray"],
                   corrupt=["none"], labels=["frac1", "full"], lamf=[1e-2, 1e-1, 1.0],
                   alpha=[1e-1], seed=[0], epochs=[100]),
    "s4_train": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"],   # log_mean = s2_physics gray/none
                  fusion=["log_owa", "log_choquet", "log_trim", "log_wmean", "log_attention"],
                  physics=["gray"], alpha=[None], seed=[0, 1], epochs=[100]),
    "s5_relt": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_mean"], physics=["gray"],
                   relt=[True], alpha=[None], seed=[0, 1], epochs=[100]),      # SOC sees relative T
    "s5_robust": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_mean", "log_attention"],
                      physics=["gray"], relt=[True], robust=["none", "pen1", "pen10", "aug"],
                      alpha=[None], seed=[0, 1], epochs=[100]),
    "s2_gamma": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_mean"], physics=["gray"],
                     relt=[True], gamma=[0.0, 1e-2, 1e-1, 1.0], labels=["all", "rpt50"],
                     alpha=[None], seed=[0, 1], epochs=[100]),      # gamma 0 + all = s5_relt
    "s6_baselines": dict(batch=[1, 2, 3, 4, 5, 6], model=["mlp", "pinn", "cascade", "ours_oneway", "ours_none"],
                         relt=[True], seed=[0, 1], epochs=[100]),        # literature baselines in our pipeline
    "s2_coupling": dict(batch=[1, 2, 3, 4, 5, 6], model=["ours_oneway", "ours_none"], relt=[True],
                        labels=["rpt50"], seed=[0, 1], epochs=[100]),   # does SOC help SOH? (two-way = s2_gamma g0)
    "s6_final": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_attention"], physics=["gray"],
                     relt=[True], gamma=[0.3, 1.0], robust=["none", "pen10"],
                     alpha=[None], seed=[0, 1], epochs=[100]),   # best SOH+SOC candidate; pick by validation
    "s5_uaw": dict(batch=[1, 2, 3, 4, 5, 6], grouping=["sensor"], fusion=["log_uaw"], physics=["gray"],
                   relt=[True], alpha=[None], seed=[0, 1], epochs=[100]),
    # --- TJU (DS=tju, set automatically for s7_*): XJTU lessons -> attention + gamma + pen10 candidates, select on val
    "s7_main": dict(batch=[1, 2, 3], grouping=["sensor"], fusion=["log_mean", "log_attention"], physics=["gray"],
                    relt=[True], gamma=[0.0, 0.3, 1.0], robust=["none", "pen10"], alpha=[None], seed=[0, 1], epochs=[100]),
    "s7_baselines": dict(batch=[1, 2, 3], model=["mlp", "pinn", "cascade", "ours_oneway", "ours_none"],
                         relt=[True], seed=[0, 1], epochs=[100]),
    "s7_scarce": dict(batch=[1, 2, 3], grouping=["sensor"], fusion=["log_mean"], physics=["gray"], relt=[True],
                      gamma=[0.0, 1.0], labels=["rpt50"], alpha=[None], seed=[0, 1], epochs=[100]),   # coupling, 2 % SOH labels
    "s7_physics": dict(batch=[1, 2, 3], grouping=["sensor"], fusion=["log_mean"], physics=["cc", "gray", "learned"],
                       relt=[True], labels=["full"], alpha=[1e-1], seed=[0, 1], epochs=[100]),         # learned correction (= s2_scarce full)
    "s7_fpen": dict(batch=[1, 2, 3], grouping=["sensor"], fusion=["log_mean"], physics=["gray"], relt=[True],
                    labels=["full"], lamf=[1.0], alpha=[1e-1], seed=[0, 1], epochs=[100]),            # correction penalty (= s2_fpenalty)
    "s7_coupling": dict(batch=[1, 2, 3], model=["ours_oneway", "ours_none"], relt=[True],
                        labels=["rpt50"], seed=[0, 1], epochs=[100]),                                  # coupling (= s2_coupling)
    "s8_xjtu": dict(FINAL, split=list(protocols.xjtu()), gamma=[0.3]),          # other published splits, XJTU settings
    "s8_tju": dict(FINAL, split=list(protocols.tju()), gamma=[0.0]),            # other published splits, TJU settings
    # --- seeds 2-4 of the final models; baselines on 5 seeds (seeds 0-1 of mlp/pinn/cascade in s6/s7)
    "s10_xjtu": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], gamma=[0.3], seed=[2, 3, 4]),
    "s10_tju": dict(FINAL, batch=[1, 2, 3], gamma=[0.0], seed=[2, 3, 4]),
    "s10_xjtu_base": dict(batch=[1, 2, 3, 4, 5, 6], model=["mlp", "pinn", "cascade", "autospinn", "pinn_wide"],
                          relt=[True], seed=[0, 1, 2, 3, 4], epochs=[100]),
    "s10_tju_base": dict(batch=[1, 2, 3], model=["mlp", "pinn", "cascade", "autospinn", "pinn_wide"],
                         relt=[True], seed=[0, 1, 2, 3, 4], epochs=[100]),
    # --- ablation ladder and penalty as a plug-in, 5 seeds: grouping by statistic with the fusion, physics and coupling
    #     of the final model (with and without penalty), final model without penalty, baselines with penalty
    "s11_xjtu_stat": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], grouping=["statistic"], gamma=[0.3], robust=["none", "pen10"],
                          seed=[0, 1, 2, 3, 4]),
    "s11_tju_stat": dict(FINAL, batch=[1, 2, 3], grouping=["statistic"], gamma=[0.0], robust=["none", "pen10"],
                         seed=[0, 1, 2, 3, 4]),
    "s11_xjtu_nopen": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], gamma=[0.3], robust=["none"], seed=[2, 3, 4]),
    "s11_tju_nopen": dict(FINAL, batch=[1, 2, 3], gamma=[0.0], robust=["none"], seed=[2, 3, 4]),
    "s11_xjtu_plug": dict(batch=[1, 2, 3, 4, 5, 6], model=["pinn", "autospinn"], relt=[True], robust=["pen10"],
                          seed=[0, 1, 2, 3, 4], epochs=[100]),
    "s11_tju_plug": dict(batch=[1, 2, 3], model=["pinn", "autospinn"], relt=[True], robust=["pen10"],
                         seed=[0, 1, 2, 3, 4], epochs=[100]),
    # --- architecture search for the final model: width x hidden layers, seed 0, chosen on validation cells
    "s12_xjtu_arch": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], gamma=[0.3], h=[8, 16, 32], depth=[1, 2, 3], seed=[0]),
    "s12_tju_arch": dict(FINAL, batch=[1, 2, 3], gamma=[0.0], h=[8, 16, 32], depth=[1, 2, 3], seed=[0]),
    # --- chosen on validation (s12_*_arch): XJTU width 32 x 2 layers, TJU 32 x 3; 5 seeds, with and without penalty;
    #     XJTU also with temperature dropout (flat or missing sensor, 10 % each)
    "s12_xjtu_final": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], gamma=[0.3], h=[32], depth=[2], robust=["pen10", "none"],
                           seed=[0, 1, 2, 3, 4]),
    "s12_xjtu_tdrop": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], gamma=[0.3], h=[32], depth=[2], td=[0.1], seed=[0, 1, 2, 3, 4]),
    "s12_tju_final": dict(FINAL, batch=[1, 2, 3], gamma=[0.0], h=[32], depth=[3], robust=["pen10", "none"],
                          seed=[0, 1, 2, 3, 4]),
    # --- penalty-weight sensitivity of the final model (0 and 10 are in s12_*_final)
    "s13_xjtu_lambda": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], gamma=[0.3], h=[32], depth=[2], td=[0.1],
                            robust=["none", "pen1", "pen3", "pen30"], seed=[0, 1, 2, 3, 4]),
    # --- ablation at the final size: statistic grouping, same width, depth and dropout, 5 seeds
    "s14_xjtu_stat": dict(FINAL, batch=[1, 2, 3, 4, 5, 6], grouping=["statistic"], gamma=[0.3], h=[32], depth=[2],
                          td=[0.1], robust=["none", "pen10"], seed=[0, 1, 2, 3, 4]),
    "s14_tju_stat": dict(FINAL, batch=[1, 2, 3], grouping=["statistic"], gamma=[0.0], h=[32], depth=[3],
                         robust=["none", "pen10"], seed=[0, 1, 2, 3, 4]),
    "s13_tju_lambda": dict(FINAL, batch=[1, 2, 3], gamma=[0.0], h=[32], depth=[3], robust=["pen1", "pen3", "pen30"],
                           seed=[0, 1, 2, 3, 4]),
}
PRESETS = {   # s6: same split / HI / labels / sensor-error tests; h=32 -> params close to ours
    "mlp": dict(grouping="all_in_one", fusion="mean", physics="gray", alpha=0.0, beta=0.0, couple="none", h=32),
    "pinn": dict(grouping="all_in_one", fusion="mean", physics="learned", alpha=None, couple="none", h=32),  # Wang 2024
    "cascade": dict(grouping="all_in_one", fusion="mean", physics="gray", alpha=0.0, beta=0.0, couple="oneway", h=32),
    "ours_oneway": dict(grouping="sensor", fusion="log_mean", physics="gray", alpha=None, couple="oneway"),
    "ours_none": dict(grouping="sensor", fusion="log_mean", physics="gray", alpha=None, couple="none"),
    "autospinn": dict(grouping="statistic", fusion="attention", physics="learned", alpha=None, couple="none"),  # grouped by statistic
    "pinn_wide": dict(grouping="all_in_one", fusion="mean", physics="learned", alpha=None, couple="none", h=64),
}
DONE_IN = {"s10_xjtu_base": "s6_baselines", "s10_tju_base": "s7_baselines"}   # seeds already trained there
I_GAIN, I_OFF = 0.02, 0.02   # stress: current sensor +2 % gain, +20 mA offset (beyond spec)
STEPS, BS_SOC, BS_SOH, BETA, PATIENCE = 200, 1024, 256, 1.0, 15


def corrupt(d, path):
    """Persistent current error into SOC input (train+test) and/or Coulomb prior. Labels stay clean."""
    if path in ("input", "both"):
        mu, sd = d["zstat"][0][1], d["zstat"][1][1]
        d["Z"][:, 1] = d["Z"][:, 1] * (1 + I_GAIN) + (I_GAIN * mu + I_OFF) / sd
    if path in ("prior", "both"):
        for sp in ("train", "val", "test"):
            d[sp]["dq"] = d[sp]["dq"] * (1 + I_GAIN) + I_OFF * d[sp]["dt_h"]
        d["q"] = d["q"] * (1 + I_GAIN) + I_OFF * d["t_h"]


def scarce(d, mode, seed):
    """SOC labels on train rows: frac1 = random 1 %, full = only at full charge (SOC = 1).
    SOH labels: rpt50 = only every 50th cycle (1, 51, ...), like periodic capacity tests."""
    if mode == "rpt50":
        r = d["train"]["soh_rows"]
        d["train"]["soh_lab"] = r[torch.from_numpy(d["hi"].cycle.values[r.numpy()] % 50 == 1)]
        return
    r = d["train"]["soc_rows"]
    if mode == "frac1":
        g = torch.Generator().manual_seed(seed)
        d["train"]["lab"] = r[torch.randperm(len(r), generator=g)[: max(1, len(r) // 100)]]
    elif mode == "full":
        d["train"]["lab"] = r[d["soc_y"][r] >= 0.999]


def run(cfg):
    torch.set_num_threads(2)
    from data import load_batch
    from model import CoupledSPINN
    torch.manual_seed(cfg["seed"])
    out = Path(cfg["dir"])
    st = out / "status" / f"{cfg['id']}.json"
    d = load_batch(cfg["batch"], rel_t=cfg.get("relt", False),
                   split=protocols.get(cfg["split"]) if cfg.get("split") else None)
    corrupt(d, cfg["corrupt"])
    scarce(d, cfg.get("labels", "all"), cfg["seed"])
    rob = cfg.get("robust", "none")
    if rob != "none" or cfg["fusion"] == "log_uaw":
        from data import add_uncertainty
        add_uncertainty(d, cfg["batch"])
    m = CoupledSPINN(cfg["grouping"], cfg["fusion"], cfg["physics"], cfg.get("couple", "two"), cfg.get("h", 16), cfg.get("depth", 2))
    m.q_nom = d["q_nom"]
    if "sx" in d:
        m.set_sigma(d)
    pen = float(rob[3:]) if rob.startswith("pen") else 0.0
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], 1e-4)
    best, best_state, bad, t0, hist = 1e9, None, 0, time.time(), []
    for ep in range(1, cfg["epochs"] + 1):
        m.train()
        for _ in range(STEPS):
            loss, parts = m.losses(d, "train", BS_SOC, BS_SOH, cfg["alpha"], cfg.get("beta", BETA),
                                   cfg.get("gamma", 0.0), cfg.get("lamf", 0.0), pen, rob == "aug", cfg.get("td", 0.0))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        v_soh, v_soc = m.evaluate(d, "val")
        score = v_soh + v_soc
        hist.append(dict(epoch=ep, val_soh=v_soh, val_soc=v_soc, **parts))
        if score < best:
            best, bad, best_state = score, 0, {k: v.clone() for k, v in m.state_dict().items()}
            best_val = (v_soh, v_soc)
        else:
            bad += 1
        st.write_text(json.dumps(dict(id=cfg["id"], epoch=ep, epochs=cfg["epochs"], elapsed=time.time() - t0,
                                      val_soh=v_soh, val_soc=v_soc, done=False)))
        if bad >= PATIENCE:
            break
    m.load_state_dict(best_state)
    t_soh, t_soc = m.evaluate(d, "test")
    res = dict(cfg, val_soh_rmse=best_val[0], val_soc_rmse=best_val[1], test_soh_rmse=t_soh, test_soc_rmse=t_soc, epochs_ran=ep, time_s=time.time() - t0,
               params=sum(p.numel() for p in m.parameters()))
    (out / "results" / f"{cfg['id']}.json").write_text(json.dumps(res))
    st.write_text(json.dumps(dict(id=cfg["id"], epoch=ep, epochs=ep, elapsed=time.time() - t0,
                                  val_soh=v_soh, val_soc=v_soc, done=True)))
    torch.save(m.state_dict(), out / "models" / f"{cfg['id']}.pt")
    return res


def logger(out):
    """print + append to <out>/log.txt"""
    f = open(Path(out) / "log.txt", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); f.write(msg + "\n"); f.flush()
    return say


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="s1_alpha", help="one grid or comma list (one shared worker pool)")
    ap.add_argument("--alpha", type=float, default=1e-2, help="used when grid alpha is None")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.grid.startswith("s7") or "tju" in a.grid:
        os.environ["DS"] = "tju"                           # inherited by the spawned workers
    cfgs, says = [], {}
    for grid in a.grid.split(","):
        g = GRIDS[grid]
        out = Path("runs") / grid
        for s in ("status", "results", "models"):
            (out / s).mkdir(parents=True, exist_ok=True)
        n0 = len(cfgs)
        keys = list(g)
        for vals in itertools.product(*g.values()):
            c = dict(zip(keys, vals))
            if "model" in c:
                c.update(PRESETS[c["model"]])
            c.setdefault("corrupt", "none"); c.setdefault("labels", "all")
            if c["physics"] == "learned" and c["corrupt"] in ("prior", "both"):
                continue                                       # no prior -> same as none/input
            c["alpha"] = a.alpha if c["alpha"] is None else c["alpha"]
            if c.get("split"):
                c["batch"] = protocols.get(c["split"])["batches"][0]
            c["id"] = (f"{c['model']}_" if "model" in c else "") + (f"{c['split']}_" if "split" in c else "") + f"b{c['batch']}_{c['grouping']}_{c['fusion']}_{c['physics']}_a{c['alpha']:g}_s{c['seed']}"
            if c["corrupt"] != "none":
                c["id"] += f"_{c['corrupt']}"
            if c.get("gamma") == 0 and c["labels"] == "all" and grid == "s2_gamma":
                continue                                       # = s5_relt
            if c.get("gamma"):
                c["id"] += f"_g{c['gamma']:g}"
            if c.get("lamf"):
                c["id"] += f"_F{c['lamf']:g}"
            if c.get("relt"):
                c["id"] += "_relT"
            if c.get("robust", "none") != "none":
                c["id"] += f"_{c['robust']}"
            if c["labels"] != "all":
                c["id"] += f"_{c['labels']}"
            if c.get("h", 16) != 16 and "model" not in c:
                c["id"] += f"_h{c['h']}"
            if c.get("td"):
                c["id"] += f"_td{c['td']:g}"
            if c.get("depth", 2) != 2:
                c["id"] += f"_d{c['depth']}"
            c["dir"] = str(out)
            if grid in DONE_IN and (Path("runs") / DONE_IN[grid] / "results" / f"{c['id']}.json").exists():
                continue
            if not (out / "results" / f"{c['id']}.json").exists():   # resumable
                cfgs.append(c)
                (out / "status" / f"{c['id']}.json").write_text(json.dumps(
                    dict(id=c["id"], epoch=0, epochs=c["epochs"], elapsed=0, done=False)))
        says[str(out)] = logger(out)
        says[str(out)](f"{len(cfgs) - n0} runs -> {out}  (monitor: python src/monitor.py {out})")
    cfgs.sort(key=lambda c: c.get("robust", "none") == "none")   # slow pen10 runs first (better packing)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run, cfgs):
            says[r["dir"]](f"done {r['id']}: SOH {100*r['test_soh_rmse']:.2f}%  SOC {100*r['test_soc_rmse']:.2f}%")


if __name__ == "__main__":
    main()
