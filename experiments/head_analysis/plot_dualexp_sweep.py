"""Plots for the dual+znorm sweep report (plans/2026-08-26_dual-plus-znorm.md).

Reads the paired-analysis metrics written by analyze_stepvlm_arms.py and renders
  sweep_curve.png   -- expert budget vs paired dminADE@6 (mean & median panels)
  decomposition.png -- per-step increments dual->e10->e15->e25, colored by axis

Usage:
  python experiments/head_analysis/plot_dualexp_sweep.py [--out dualexp_sweep]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

PROP_SLOPE = 0.0668 / 2_657_452_032  # dual's VLM cost per removed param


def ade(exp_id):
    m = json.loads((REPO / "outputs" / exp_id / "metrics.json").read_text())["minADE"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dualexp_sweep")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # --- sweep curve: conditional arms vs dual, with the stale e25 for reference
    budgets = [0, 220_446_720, 311_574_528, 532_021_248]  # expert params removed
    cond = [None, ade("dualexp_e10_arms_val"), ade("dualexp_e15_arms_val"),
            ade("dualexp_cond_arms_val")]
    stale = ade("dualexp_stale_arms_val")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    for ax, key, lo_k, hi_k, title in (
            (axes[0], "delta_mean", "lo", "hi", "mean paired dminADE@6 (vs dual)"),
            (axes[1], "delta_median", "median_lo", "median_hi", "median paired dminADE@6")):
        xs = [b / 1e6 for b in budgets]
        ys = [0.0] + [m[key] for m in cond[1:]]
        ax.plot(xs, ys, color=C1, lw=2, zorder=3)
        ax.scatter(xs, ys, color=C1, s=28, zorder=4)
        for b, m in zip(budgets[1:], cond[1:]):
            ax.errorbar(b / 1e6, m[key], yerr=[[m[key] - m[lo_k]], [m[hi_k] - m[key]]],
                        color=C1, lw=1.2, capsize=3, zorder=3)
        xs_stale = budgets[3] / 1e6 + 14  # offset so the marker is not hidden by cond's
        ax.errorbar(xs_stale, stale[key],
                    yerr=[[stale[key] - stale[lo_k]], [stale[hi_k] - stale[key]]],
                    color=C4, lw=1.2, capsize=3, zorder=3)
        ax.scatter([xs_stale], [stale[key]], color=C4, s=28, zorder=4)
        ax.plot([0, 560], [0, 560e6 * PROP_SLOPE], color=MUTED, lw=1.2, ls="--", zorder=2)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("expert params removed (M)")
        ax.set_title(title)
        for x, m, name, dy in ((budgets[1] / 1e6, cond[1], "e10", 8),
                               (budgets[2] / 1e6, cond[2], "e15", -14),
                               (budgets[3] / 1e6, cond[3], "e25 (cond)", 8)):
            ax.annotate(name, (x, m[key]), textcoords="offset points", xytext=(4, dy),
                        fontsize=9, color=INK)
        ax.annotate("e25 (stale)", (xs_stale, stale[key]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK)
    axes[0].annotate("proportional-cost prediction", (350, 350e6 * PROP_SLOPE),
                     textcoords="offset points", xytext=(0, -16), fontsize=9, color=MUTED)
    axes[0].set_ylabel("dminADE@6 (m)")
    fig.tight_layout()
    fig.savefig(out / "sweep_curve.png", dpi=160)
    plt.close(fig)

    # --- decomposition: per-step increments, colored by which axis the step touches
    steps = [("dual -> e10\n(+2 Q heads + 826 MLP ch/layer)", ade("dualexp_e10_arms_val"), C1),
             ("e10 -> e15\n(+412 MLP ch/layer only)", ade("dualexp_e15_vs_e10"), C2),
             ("e15 -> e25\n(+2 Q heads + 826 MLP ch/layer)", ade("dualexp_e25_vs_e15"), C1)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ys = range(len(steps))
    for y, (label, m, color) in zip(ys, steps):
        ax.barh(y, m["delta_mean"], color=color, height=0.55, zorder=3)
        ax.errorbar(m["delta_mean"], y, xerr=[[m["delta_mean"] - m["lo"]],
                                              [m["hi"] - m["delta_mean"]]],
                    color=INK, lw=1.2, capsize=3, zorder=4)
        ax.annotate(f"{m['delta_mean']:+.4f}", (max(m["hi"], m["delta_mean"]), y),
                    textcoords="offset points", xytext=(6, -3), fontsize=9, color=INK)
    ax.set_yticks(list(ys), [s[0] for s in steps])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("incremental paired dminADE@6 (m), 95% CI")
    ax.set_title("which axis pays -- only the Q-head steps cost")
    fig.tight_layout()
    fig.savefig(out / "decomposition.png", dpi=160)
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    main()
