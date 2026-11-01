"""Figures for tyr_off's per-layer allocation against first-order Taylor importance.

Reads outputs/cvpr27/tyr_off_allocation/metrics.json (analyze_tyr_allocation.py) and writes two PNGs there:
  allocation_profile.png       kept share per layer: tyr_off (bars), the raw-Taylor allocation with tyr's
                               totals (FM = traj, CoC; lines), MoOP's uniform cut (dashed)
  allocation_vs_taylor_mass.png  per layer, tyr_off's cut share against the layer's share of Taylor mass
                               (FM / CoC x Q heads / MLP), dots shaded by depth, Spearman rho in each title
Layer 35 has no FM score (structurally zero) and is left out of the FM series.

  .venv/bin/python cvpr27/code/plot_tyr_allocation.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

REPO = Path(__file__).resolve().parents[2]
D = REPO / "outputs" / "cvpr27" / "tyr_off_allocation"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_TYR, C_FM, C_COC = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated all-pairs)
DEPTH = LinearSegmentedColormap.from_list("depth", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
N_LAYERS = 36
UNITS = {"q": 32, "mlp": 12288}
MOOP_KEPT = {"q": 19, "mlp": 7390}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["DejaVu Sans"],
})


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return float(np.corrcoef(r(np.asarray(a)), r(np.asarray(b)))[0, 1])


def profile(m):
    x = np.arange(N_LAYERS)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    for ax, axis, title in ((axes[0], "q", "Q heads kept per layer"),
                            (axes[1], "mlp", "MLP channels kept per layer")):
        n = UNITS[axis]
        tyr = 100 * np.array(m[f"tyr_kept_{axis}"]) / n
        fm = 100 * np.array(m[f"taylor_traj_kept_{axis}"], float) / n
        fm[35] = np.nan                                   # no FM score in layer 35
        coc = 100 * np.array(m[f"taylor_coc_kept_{axis}"]) / n
        ax.bar(x, tyr, width=0.62, color=C_TYR, label="tyr_off (searched)", zorder=2)
        ax.plot(x, fm, "-o", color=C_FM, lw=2, ms=4, label="Taylor FM, same total", zorder=3)
        ax.plot(x, coc, "-s", color=C_COC, lw=2, ms=4, label="Taylor CoC, same total", zorder=3)
        ax.axhline(100 * MOOP_KEPT[axis] / n, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1,
                   label="MoOP (uniform cut)")
        ax.set_ylim(0, 105)
        ax.set_ylabel("kept (%)")
        ax.set_title(title, loc="left", color=INK)
        # direct labels at the right edge (aqua sits below 3:1 on the surface)
        ax.annotate("CoC", (35, coc[35]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
        ax.annotate("FM", (34, fm[34]), xytext=(6, 6), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
        ax.annotate("MoOP", (35.6, 100 * MOOP_KEPT[axis] / n), xytext=(2, 4), textcoords="offset points",
                    va="bottom", color=INK2, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(s) for s in ("tyr_off (searched)", "Taylor FM, same total", "Taylor CoC, same total",
                                    "MoOP (uniform cut)")]
    axes[0].legend([h[i] for i in order], [lab[i] for i in order], ncol=4, frameon=False, fontsize=8,
                   loc="lower left", bbox_to_anchor=(0.0, 1.08))
    axes[1].set_xlabel("VLM layer")
    axes[1].set_xticks(x[::2])
    axes[1].set_xlim(-0.8, 37.2)
    fig.suptitle("tyr_off allocation vs raw first-order Taylor allocation (same kept totals: 684 heads, 266,040 ch)",
                 x=0.01, ha="left", fontsize=10.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(D / "allocation_profile.png", dpi=170)
    plt.close(fig)


def relation(m):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
    labels = {"q": {29: "29", 15: "15", 0: "0", 3: "3"}, "mlp": {1: "1", 2: "2", 3: "3", 16: "16", 0: "0"}}
    for r, axis in enumerate(("q", "mlp")):
        n = UNITS[axis]
        cut = 100 * (1 - np.array(m[f"tyr_kept_{axis}"]) / n)
        for c, (obj, name) in enumerate((("traj", "FM"), ("coc", "CoC"))):
            ax = axes[r, c]
            mass = np.array(m[f"{obj}_{axis}_mass"])
            L = np.flatnonzero(mass > 0)
            share = 100 * mass[L] / mass[L].sum()
            rho = spearman(cut[L], mass[L])
            sc = ax.scatter(share, cut[L], c=L, cmap=DEPTH, vmin=0, vmax=35, s=46,
                            edgecolors=SURFACE, linewidths=1.2, zorder=3)
            for i, t in labels[axis].items():
                if i in L:
                    j = int(np.flatnonzero(L == i)[0])
                    ax.annotate(f"L{t}", (share[j], cut[L][j]), xytext=(5, 9 if i == 16 else 4),
                                textcoords="offset points", fontsize=8, color=INK2)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(-3, 103)
            ax.set_title(f"{'Q heads' if axis == 'q' else 'MLP'} vs {name} Taylor   Spearman ρ = {rho:+.2f}",
                         loc="left", color=INK)
            if r == 1:
                ax.set_xlabel("layer's share of total Taylor mass (%, log)")
            if c == 0:
                ax.set_ylabel("tyr_off cut (%)")
            ax.axhline(100 * (1 - MOOP_KEPT[axis] / n), color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.annotate("MoOP", (ax.get_xlim()[1], 100 * (1 - MOOP_KEPT[axis] / n)), xytext=(-30, 4),
                        textcoords="offset points", fontsize=8, color=INK2)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("VLM layer (depth)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Per layer: how much tyr_off cut vs how much Taylor importance the layer holds",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.savefig(D / "allocation_vs_taylor_mass.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    m = json.loads((D / "metrics.json").read_text())["per_layer"]
    profile(m)
    relation(m)
    print("->", D / "allocation_profile.png", D / "allocation_vs_taylor_mass.png")


if __name__ == "__main__":
    main()
