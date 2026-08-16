"""Plots for the VQA-context importance study.

Usage: python experiments/lingoqa/plot_context.py --exp-id lingo_context_report
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

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

# every arm at the same -24.0% budget. "calib" is the data the language half of the
# criterion was measured on; the trajectory half is always PhysicalAI-AV.
ARMS = [
    ("baseline", None, 73.2, "—"),
    ("trajvqa", "LingoQA", 71.2, "max(traj, VQA)"),
    ("dual", "AV", 68.8, "max(traj, CoC)"),
    ("j_traj", "AV", 67.8, "max(traj, J)"),
    ("vqa", "LingoQA", 66.4, "VQA alone"),
    ("coclingo", "LingoQA", 65.4, "CoC alone"),
    ("traj", "AV", 37.0, "traj alone"),
    ("j", "AV", 32.2, "J alone"),
    ("coc", "AV", 30.2, "CoC alone"),
]


def arms_plot(out):
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    names = [f"{n}\n({c})" if c else n for n, c, _, _ in ARMS]
    vals = [v for _, _, v, _ in ARMS]
    cols = []
    for n, c, _, _ in ARMS:
        cols.append(MUTED if c is None else (C2 if c == "LingoQA" else C1))
    ax.bar(range(len(vals)), vals, color=cols, width=0.66)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.1, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(73.2, color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylabel("Lingo-Judge accuracy (%)")
    ax.set_ylim(0, 84)
    ax.set_title("Matched budget (-24.0%). Blue = calibrated on PhysicalAI-AV, "
                 "green = on LingoQA train")
    fig.tight_layout()
    fig.savefig(out / "context_arms.png", dpi=160)
    plt.close(fig)


def inversion_plot(out):
    """The reduced plan's inference failed: overlap did not predict effect size."""
    # (label, Jaccard Q, Jaccard MLP, measured accuracy change)
    pts = [("domain\nCoC: AV -> LingoQA", 0.753, 0.686, 35.2),
           ("context\nLingoQA: CoC -> VQA", 0.611, 0.636, 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    x = np.arange(2)
    ax.bar(x - 0.18, [p[1] for p in pts], width=0.34, color=C1, label="Q head")
    ax.bar(x + 0.18, [p[2] for p in pts], width=0.34, color=C4, label="MLP")
    ax.axhline(0.803, color=MUTED, ls=":", lw=1.2)
    ax.text(1.42, 0.815, "reproducibility ceiling", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pts], fontsize=8)
    ax.set_ylabel("kept-set Jaccard")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    ax.set_title("What the selections looked like", fontsize=10)

    ax = axes[1]
    ax.bar(x, [p[3] for p in pts], width=0.5, color=[C2, C3])
    for i, p in enumerate(pts):
        ax.text(i, p[3] + 0.9, f"+{p[3]:.1f}pp", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pts], fontsize=8)
    ax.set_ylabel("LingoQA accuracy change (pp)")
    ax.set_ylim(0, 40)
    ax.set_title("What they actually did", fontsize=10)

    fig.suptitle("The more similar selection produced the far larger effect", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "context_inversion.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="lingo_context_report")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id / "plots"
    out.mkdir(parents=True, exist_ok=True)
    arms_plot(out)
    inversion_plot(out)
    print(f"2 plots -> {out}")


if __name__ == "__main__":
    main()
