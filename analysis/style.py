"""Shared plotting style for the article figures."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = {"ink": "#1b1f24", "ash": "#5b6570", "mist": "#e8ecf1", "blue": "#2d6cb5", "lightblue": "#8fb8e0",
     "teal": "#1f8a86", "green": "#3f8f4a", "amber": "#d99000", "orange": "#dd6b20", "red": "#c0392b",
     "plum": "#7d4a9e", "slate": "#43536b", "grey": "#8a94a0"}
MODEL = {"mlp": ("MLP", C["grey"], "s"), "pinn": ("PINN", C["orange"], "^"), "cascade": ("Cascade", C["amber"], "D"),
         "ours": ("SA-SPINN", C["blue"], "o")}
SENSOR = {"V": C["blue"], "I": C["teal"], "T": C["amber"], "t": C["plum"]}


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8.2, "axes.titlesize": 8.6, "axes.labelsize": 8.2, "axes.linewidth": 0.7,
        "axes.edgecolor": C["ash"], "axes.labelcolor": C["ink"], "axes.titleweight": "bold", "axes.titlepad": 5,
        "axes.grid": True, "grid.color": "#d8dee6", "grid.linewidth": 0.5,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "xtick.color": C["ash"], "ytick.color": C["ash"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "legend.fontsize": 7.0, "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#c9d2dc",
        "legend.borderpad": 0.4, "legend.handlelength": 1.5, "lines.linewidth": 1.4, "lines.markersize": 4.0,
        "patch.linewidth": 0.6, "text.color": C["ink"], "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42})


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if grid == "y":
        ax.xaxis.grid(False)
    elif grid == "x":
        ax.yaxis.grid(False)
    elif grid == "none":
        ax.grid(False)


def tag(ax, s, dx=-0.13, dy=1.09):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def box(ax, x, y, w, h, fc="white", ec=C["ash"], lw=0.9, r=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, edgecolor=ec,
                                facecolor=fc, linestyle=ls, zorder=z))


def txt(ax, x, y, s, size=6.6, color=C["ink"], weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=6, **kw)


def arrow(ax, p0, p1, color=C["slate"], lw=0.9, rad=0.0, ls="-", ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=5,
                                 shrinkA=0.5, shrinkB=0.5, connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)
