"""Result figures. Reads results/numbers.json and the ablation runs."""
import glob, json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from style import C, MODEL, SENSOR, ROOT, save, tag, tidy, use_style

use_style()
N = json.loads((ROOT / "results" / "numbers.json").read_text())
X, T = N["xjtu"], N["tju"]
OURS = {"xjtu": "final_g0.3_pen10", "tju": "attention_pen10"}
NOPEN = {"xjtu": "final_g0.3_none", "tju": "attention_none"}       # same model trained without the penalty
W_TJU = {1: 66, 2: 55, 3: 9}


def pick(ds, m):
    return N[ds][OURS[ds] if m == "ours" else m]


ALL = {"mlp": ("MLP", C["grey"]), "pinn": ("PINN", C["orange"]), "pinn_wide": ("Wide PINN", C["red"]),
       "cascade": ("Cascade", C["amber"]), "autospinn": ("Sep. (stat.)", C["teal"]), "final": ("SA-SPINN", C["blue"])}


def fig_noisefloor():
    S = N["s10"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.05]))
    for ax, ds, title, t in zip(axs[:2], ("xjtu", "tju"), ("XJTU (6 batches)", "TJU (3 batches)"), "ab"):
        for i, (m, (_, col)) in enumerate(ALL.items()):
            v = S[ds][m]
            lo, hi = v["u_soh_ci"]
            ax.bar(i - 0.19, v["clean"][0], 0.36, color="white", edgecolor=col, hatch="////", lw=0.9)
            ax.bar(i + 0.19, v["u"][0], 0.36, color=col, edgecolor=col)
            ax.errorbar(i + 0.19, v["u"][0], [[v["u"][0] - lo], [hi - v["u"][0]]], color=C["ink"], lw=0.8, capsize=1.8)
            ax.text(i + 0.19, hi + 0.04, f"{v['u'][0]:.2f}", ha="center", va="bottom", fontsize=6.2,
                    bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
        ax.set_xticks(range(len(ALL)), [n for n, _ in ALL.values()], fontsize=6.2, rotation=28, ha="right")
        ax.set_ylabel("SOH (%)")
        ax.set_title(title)
        ax.set_ylim(0, 3.9 if ds == "xjtu" else 2.0)
        tidy(ax); tag(ax, f"({t})", dx=-0.2)
    axs[0].bar(0, 0, color="white", edgecolor=C["ash"], hatch="////", label="test RMSE (clean data)")
    axs[0].bar(0, 0, color=C["ash"], label=r"sensor-induced uncertainty $u_{\mathrm{SOH}}$")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.2, -0.3), ncol=2, frameon=False)
    ax = axs[2]
    n = len(ALL)
    for i, (m, (name, col)) in enumerate(ALL.items()):
        for j, ds in enumerate(("xjtu", "tju")):
            v = S[ds][m]["stress"][0]
            x = j + (i - (n - 1) / 2) * 0.14
            ax.bar(x, v, 0.125, color=col, label=name if j == 0 else None)
            ax.text(x, v * 1.08, f"{v:.0f}" if v > 10 else f"{v:.1f}", ha="center", fontsize=5.6)
    ax.set_yscale("log"); ax.set_ylim(1, 90)
    ax.set_xticks([0, 1], ["XJTU", "TJU"])
    ax.set_ylabel("SOH RMSE under stress (%)")
    ax.set_title("Ten-fold budget + noise")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, columnspacing=0.6,
              handlelength=1.0, fontsize=6.4)
    tidy(ax); tag(ax, "(c)", dx=-0.22)
    save(fig, "fig_noisefloor")


SWEEP = {"mlp": ("MLP", C["grey"], "s"), "pinn": ("PINN", C["orange"], "^"),
         "autospinn": ("Sep. (stat.)", C["teal"], "D"), "final": ("SA-SPINN", C["blue"], "o"),
         "nopen": ("SA-SPINN, no penalty", C["blue"], "o")}


def fig_sweep():
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.3), sharex=True, gridspec_kw=dict(hspace=0.16, wspace=0.26))
    lv = ["0.5", "1", "2", "4", "10"]
    x = [float(v) for v in lv]
    for c, (ds, title) in enumerate((("xjtu", "XJTU"), ("tju", "TJU"))):
        for r, (idx, yl) in enumerate(((0, r"$u_{\mathrm{SOH}}$ (pp)"), (2, "SOH RMSE (%)"))):
            ax = axs[r, c]
            for m, (name, col, mk) in SWEEP.items():
                s = N["s11"][ds]["sweep"][m]
                ax.plot(x, [s["sw" + k][idx] for k in lv], marker=mk, color=col, label=name,
                        lw=2.0 if m == "final" else 1.1, ls="--" if m == "nopen" else "-",
                        mfc="white" if m == "nopen" else col, zorder=5 if m == "final" else 3)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(x, [r"0.5$\times$", r"1$\times$", r"2$\times$", r"4$\times$", r"10$\times$"])
            ax.minorticks_off()
            lo, hi = ax.get_ylim()
            ax.set_yticks([t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if lo <= t <= hi])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            if r == 0:
                ax.axhline(1, color=C["red"], lw=0.8, ls=":")
                ax.text(0.52, 1.12, "1 pp", color=C["red"], fontsize=6.8)
                ax.set_title(title)
            else:
                ax.set_xlabel("sensor error (multiple of the declared budget)")
            if c == 0:
                ax.set_ylabel(yl)
            tidy(ax, "both"); tag(ax, f"({'abcd'[2 * r + c]})", dx=-0.14, dy=1.07)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.045), fontsize=7.2)
    save(fig, "fig_sweep")


LADDER = [("Mixed inputs (PINN)", "pinn", "pinn_pen10", C["orange"]),
          ("Separable by statistic\n(as in AutoSPINN)", "autospinn", "autospinn_pen10", C["teal"]),
          ("Separable by statistic\n(SA-SPINN physics)", "stat_none", "stat_pen10", C["green"]),
          ("Separable by sensor\n(SA-SPINN)", "nopen", "final", C["blue"])]


def fig_ladder():
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 3.9), sharey=True, gridspec_kw=dict(hspace=0.5, wspace=0.08))
    y = np.arange(len(LADDER))[::-1]
    for c, (ds, title) in enumerate((("xjtu", "XJTU"), ("tju", "TJU"))):
        L = N["s11"][ds]["ladder"]
        for r, (k, xl) in enumerate((("u", r"sensor-induced $u_{\mathrm{SOH}}$ (pp)"), ("clean", "clean SOH RMSE (%)"))):
            ax = axs[r, c]
            for yy, (name, a, b, col) in zip(y, LADDER):
                va, vb = L[a][k][0], L[b][k][0]
                ax.annotate("", xy=(vb, yy), xytext=(va, yy),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=4, shrinkB=4))
                ax.plot(va, yy, "o", mfc="white", mec=col, ms=6, mew=1.3, zorder=4)
                ax.plot(vb, yy, "o", color=col, ms=6, zorder=4)
                if k == "u":
                    ax.text(vb / 1.12, yy + 0.22, f"{vb:.2f}", color=col, fontsize=6.8, ha="right")
            if k == "u":
                ax.set_xscale("log"); ax.minorticks_off()
                ax.set_xticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"]); ax.set_xlim(0.08, 3.2)
                ax.set_title(title)
            else:
                ax.set_xlim(0.65, 1.3)
            ax.set_xlabel(xl)
            ax.set_yticks(y, [n for n, *_ in LADDER])
            ax.tick_params(axis="y", labelleft=c == 0)
            ax.set_ylim(-0.6, len(LADDER) - 0.4)
            tidy(ax, "x"); tag(ax, f"({'abcd'[2 * r + c]})", dx=-0.06 if c else -0.5, dy=1.07)
    h = [plt.Line2D([], [], marker="o", mfc="white", mec=C["ink"], ls="", label="trained without the penalty"),
         plt.Line2D([], [], marker="o", color=C["ink"], ls="", label=r"with the penalty ($\lambda=10$)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.04), fontsize=7.2)
    save(fig, "fig_ladder")


def fig_budget():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.5, width_ratios=[1.25, 1, 0.9]))
    ax = axs[0]
    rows = [("SOH", N["gum_abs"]["share_soh"]), ("SOC, absolute $T$", N["gum_abs"]["share_soc"]),
            ("SOC, relative $T$", N["gum_rel"]["share_soc"])]
    src = [("V_off", "voltage offset", C["blue"]), ("I_off", "current offset", C["teal"]),
           ("I_gain", "current gain", C["green"]), ("T_off", "temperature offset", C["amber"])]
    for i, (lab, sh) in enumerate(rows):
        left = 0
        for k, name, col in src:
            ax.barh(i, sh[k], left=left, color=col, height=0.6, label=name if i == 0 else None)
            if sh[k] > 9:
                ax.text(left + sh[k] / 2, i, f"{sh[k]:.0f}", ha="center", va="center", color="white", fontsize=7)
            left += sh[k]
    ax.set_yticks(range(3), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("share of output variance (%)"); ax.set_xlim(0, 100)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.42, -0.3), frameon=False, columnspacing=0.8)
    tidy(ax, "x"); tag(ax, "(a)", dx=-0.42)
    ax = axs[1]
    rng = np.random.default_rng(0)
    for i, (k, col) in enumerate((("ratio_soh", C["blue"]), ("ratio_soc", C["teal"]))):
        y = np.array(N["gum_abs"][k])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=16, color=col, alpha=0.8, edgecolor="white", lw=0.4)
        ax.plot([i - 0.22, i + 0.22], [np.median(y)] * 2, color=C["ink"], lw=1.4)
    ax.axhline(1, color=C["red"], ls=":", lw=0.8)
    ax.set_xticks([0, 1], ["SOH", "SOC"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("linear / Monte Carlo uncertainty")
    tidy(ax); tag(ax, "(b)", dx=-0.3)
    ax = axs[2]
    g = N["gum_abs"]
    vals = [(X["log_mean"]["clean"][0], g["u_mc_soh"]), (X["log_mean"]["clean"][1], g["u_mc_soc"])]
    for i, (rm, u) in enumerate(vals):
        ax.bar(i - 0.19, rm, 0.36, color="white", edgecolor=C["slate"], hatch="////", label="test RMSE" if i == 0 else None)
        ax.bar(i + 0.19, u, 0.36, color=C["slate"], label="$u$ (Monte Carlo)" if i == 0 else None)
        ax.text(i + 0.19, u + 0.03, f"{u:.2f}", ha="center", fontsize=6.6)
        ax.text(i - 0.19, rm + 0.03, f"{rm:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks([0, 1], ["SOH", "SOC"]); ax.set_ylabel("% (RMSE) or pp ($u$)"); ax.set_ylim(0, 1.7)
    ax.legend(loc="upper left", fontsize=6.4)
    tidy(ax); tag(ax, "(c)", dx=-0.34)
    save(fig, "fig_budget")


def fig_mechanism():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.42, width_ratios=[1.25, 1.15, 1]))
    ax = axs[0]
    ops = [("log_mean", "mean"), ("log_wmean", "weighted"), ("log_owa", "OWA"), ("log_choquet", "Choquet"),
           ("log_trim", "trimmed"), ("log_attention", "attention")]
    for i, (k, lab) in enumerate(ops):
        g, bottom = X[k]["gain_soh"], 0
        for j, s in enumerate("VITt"):
            ax.bar(i, g[j], 0.62, bottom=bottom, color=SENSOR[s], label={"t": "clock"}.get(s, s) if i == 0 else None)
            bottom += g[j]
        ax.text(i, bottom + 0.05, f"{bottom:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks(range(len(ops)), [o[1] for o in ops], rotation=28, ha="right")
    ax.set_ylabel("fusion gain $g_j$ per sensor"); ax.set_ylim(0, 4.3)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, columnspacing=0.7, handlelength=1)
    tidy(ax); tag(ax, "(a)", dx=-0.2, dy=1.2)
    ax = axs[1]
    tr = [("none", "none", "#c9d2dc"), ("pen1", r"penalty $\lambda$=1", C["lightblue"]),
          ("pen10", r"penalty $\lambda$=10", C["blue"]), ("aug", "augmentation", C["teal"])]
    for i, f in enumerate(("mean", "attention")):
        for j, (k, lab, col) in enumerate(tr):
            v = X[f"{f}_{k}"]["u"][0]
            ax.bar(i + (j - 1.5) * 0.2, v, 0.18, color=col, label=lab if i == 0 else None)
            ax.text(i + (j - 1.5) * 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.0)
    ax.set_xticks([0, 1], ["mean fusion", "attention fusion"])
    ax.set_ylabel(r"$u_{\mathrm{SOH}}$ within budget (pp)"); ax.set_ylim(0, 0.78)
    ax.legend(fontsize=6.2, loc="upper right")
    tidy(ax); tag(ax, "(b)", dx=-0.22, dy=1.2)
    ax = axs[2]
    for f, mk in (("mean", "s"), ("attention", "o")):
        for k, lab, col in tr:
            v = X[f"{f}_{k}"]
            ax.scatter(sum(v["gain_soh"]), v["u"][0], marker=mk, s=34, color=col, edgecolor=C["ink"], lw=0.5, zorder=4)
    a, b = X["mean_none"], X["attention_pen10"]
    ax.annotate("", xy=(sum(b["gain_soh"]) + 0.08, b["u"][0] + 0.03), xytext=(sum(a["gain_soh"]) - 0.08, a["u"][0] - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.1))
    ax.text(2.25, 0.5, f"$-{100 * (1 - b['u'][0] / a['u'][0]):.0f}\\,\\%$", color=C["red"], fontsize=7.4)
    ax.scatter([], [], marker="s", color="white", edgecolor=C["ink"], label="mean")
    ax.scatter([], [], marker="o", color="white", edgecolor=C["ink"], label="attention")
    ax.set_xlabel("total fusion gain $\\sum_j g_j$"); ax.set_ylabel(r"$u_{\mathrm{SOH}}$ (pp)")
    ax.set_xlim(1.6, 3.8); ax.set_ylim(0, 0.75); ax.legend(fontsize=6.4, loc="lower right")
    tidy(ax, "both"); tag(ax, "(c)", dx=-0.3, dy=1.2)
    save(fig, "fig_mechanism")


def runs(folder, **flt):
    """Per-batch seed-mean test RMSE (SOH, SOC) in % for training runs that match the filter."""
    by = defaultdict(list)
    for f in glob.glob(str(ROOT / "runs" / folder / "results" / "*.json")):
        r = json.load(open(f))
        if all(r.get(k, d) == v for k, (v, d) in flt.items()):
            by[r["batch"]].append((100 * r["test_soh_rmse"], 100 * r["test_soc_rmse"]))
    return {b: np.mean(v, 0) for b, v in sorted(by.items())}


def wmean(pb, tju):
    w = np.array([W_TJU[b] if tju else 1 for b in pb], float)
    return (w[:, None] * np.array(list(pb.values()))).sum(0) / w.sum()


def fig_coupling():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    ax = axs[0]
    cp = {"xjtu": [runs("s2_coupling", couple=("none", None)), runs("s2_coupling", couple=("oneway", None)),
                   runs("s2_gamma", labels=("rpt50", "all"), gamma=(0.0, 0.0))],
          "tju": [runs("s7_coupling", couple=("none", None)), runs("s7_coupling", couple=("oneway", None)),
                  runs("s7_scarce", gamma=(0.0, 0.0))]}
    groups = [("XJTU\n(6 batches)", [wmean(p, False)[1] for p in cp["xjtu"]])]
    groups += [(f"TJU\n{n}", [p[b][1] for p in cp["tju"]]) for b, n in ((1, "NCA"), (2, "NCM"), (3, "blend"))]
    cols = ["#c9d2dc", C["lightblue"], C["blue"]]
    labs = ["no coupling (nominal capacity)", "one-way (SOH to SOC)", "two-way (joint training)"]
    for i, (name, v) in enumerate(groups):
        for j in range(3):
            ax.bar(i + (j - 1) * 0.26, v[j], 0.24, color=cols[j], label=labs[j] if i == 0 else None)
            ax.text(i + (j - 1) * 0.26, v[j] + 0.05, f"{v[j]:.2f}", ha="center", fontsize=6.2)
    ax.set_xticks(range(4), [g[0] for g in groups]); ax.set_ylabel("SOC RMSE (%)"); ax.set_ylim(0, 4.6)
    ax.legend(fontsize=6.6, loc="upper left")
    tidy(ax); tag(ax, "(a)", dx=-0.1)
    ax = axs[1]
    none = ("none", "none")
    ph = {"xjtu": [runs("s2_scarce", labels=("full", "all"), physics=(p, None), corrupt=none) for p in ("cc", "gray", "learned")]
          + [runs("s2_fpenalty", labels=("full", "all"), lamf=(1.0, 0.0))],
          "tju": [runs("s7_physics", physics=(p, None)) for p in ("cc", "gray", "learned")] + [runs("s7_fpen")]}
    names = ["Coulomb\nonly", "Coulomb +\nfree $F$", "free $F$\nonly", "Coulomb +\npenalised $F$"]
    for j, (ds, col) in enumerate((("xjtu", C["slate"]), ("tju", C["teal"]))):
        v = [wmean(p, ds == "tju")[1] for p in ph[ds]]
        ax.bar(np.arange(4) + (j - 0.5) * 0.36, v, 0.34, color=col, label=ds.upper())
        for i, y in enumerate(v):
            ax.text(i + (j - 0.5) * 0.36, y * 1.1, f"{y:.1f}", ha="center", fontsize=6.2)
    ax.set_yscale("log"); ax.set_ylim(1, 150); ax.set_xticks(range(4), names, fontsize=6.8)
    ax.set_ylabel("SOC RMSE (%)"); ax.legend(fontsize=6.6, loc="upper right")
    tidy(ax); tag(ax, "(b)", dx=-0.16)
    save(fig, "fig_coupling")
    return cp, ph


def fig_lambda():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for ds, name, col, mk in (("xjtu", "XJTU", C["blue"], "o"), ("tju", "TJU", C["teal"], "s")):
        L = N["s13"][ds]
        lam = sorted(L, key=float)
        x, y = [L[k]["clean"][0] for k in lam], [L[k]["u"][0] for k in lam]
        ax.plot(x, y, color=col, lw=1.2, zorder=3, label=name)
        for k, a, b in zip(lam, x, y):
            ax.plot(a, b, mk, color=col if k == "10" else "white", mec=col, ms=6.5 if k == "10" else 5, mew=1.2, zorder=4)
            ax.annotate(rf"$\lambda$={k}", (a, b), xytext=(6, 0), textcoords="offset points", fontsize=6.4,
                        va="center", color=C["ink"], fontweight="bold" if k == "10" else "normal")
    ax.set_yscale("log"); ax.minorticks_off()
    ax.set_yticks([0.1, 0.2, 0.5, 1], ["0.1", "0.2", "0.5", "1"]); ax.set_ylim(0.08, 1.1)
    ax.set_xlim(0.72, 1.16)
    ax.set_xlabel("clean SOH RMSE (%)"); ax.set_ylabel(r"sensor-induced $u_{\mathrm{SOH}}$ (pp)")
    ax.legend(loc="lower left", frameon=False)
    tidy(ax, "both")
    save(fig, "fig_lambda")


if __name__ == "__main__":
    fig_lambda(); fig_noisefloor(); fig_ladder(); fig_sweep(); fig_budget(); fig_mechanism()
    cp, ph = fig_coupling()
    for ds in cp:
        print(ds, "coupling SOH/SOC", [wmean(p, ds == "tju").round(2).tolist() for p in cp[ds]])
        print(ds, "physics SOH/SOC", [wmean(p, ds == "tju").round(2).tolist() for p in ph[ds]])
