"""Ratio-sweep curves from outputs/cvpr27/ratio_sweep/metrics.json (analyze_ratio_sweep.py).

Four small multiples against the share of the model removed: test2000 minADE@6 and minFDE@6 at 6.4 s, OOD-val
minADE@6 at 6.4 s, LingoQA. Categorical slots 1-6 of the dataviz default palette in fixed order (validated: all gates
pass; contrast WARN -> each line is direct-labelled and carries its own marker). The full six-metric tables are in
the report.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "cvpr27" / "ratio_sweep" / "metrics.json"
OUT = REPO / "cvpr27" / "reports" / "figures" / "2026-11-05_ratio-sweep.png"
RATIOS = [35, 40, 45, 50]
REMOVED = {35: 20.92, 40: 23.99, 45: 26.85, 50: 29.98}
SERIES = [("MoOP l0keep", "#2a78d6", "o"), ("Tyr", "#eb6834", "s"), ("LLM-Pruner", "#1baf7a", "^"),
          ("Wanda-sp", "#eda100", "v"), ("FM-only", "#e87ba4", "D"), ("CE-only", "#008300", "P")]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
PANELS = [("test2000", "ade_6.4s", "test_2000 minADE@6, 6.4 s (m)"),
          ("test2000", "fde_6.4s", "test_2000 minFDE@6, 6.4 s (m)"),
          ("oodval", "ade_6.4s", "OOD-Val minADE@6, 6.4 s (m)"),
          ("lingoqa", "accuracy", "LingoQA accuracy (%)")]


def main():
    res = json.loads(SRC.read_text())
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = [REMOVED[r] for r in RATIOS]
    for ax, (s, key, title) in zip(axes, PANELS):
        ends = []
        for name, color, marker in SERIES:
            if s == "lingoqa":
                y = [100 * res[f"lingoqa/u{r}/{name}"]["accuracy"] for r in RATIOS]
            else:
                y = [res[f"{s}/u{r}/{name}"][key] for r in RATIOS]
            ax.plot(x, y, color=color, marker=marker, markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            ends.append([y[-1], name])
        if s == "lingoqa":
            ax.axhline(72.6, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
            ax.annotate("dense 72.6%", (x[-1] + 2.5, 72.6), xytext=(0, -10), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="right")
        # direct labels at the line ends, pushed apart so close end points stay readable
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y_lab, name in ends:
            ax.annotate(name, (x[-1] + 0.35, y_lab), va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("removed (% of model)")
        ax.set_xticks(x, [f"{v:.1f}\n(u{r})" for v, r in zip(x, RATIOS)])
        ax.set_xlim(x[0] - 0.8, x[-1] + 2.6)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("A100 ratio sweep: every method at MoOP u{N}'s budget (VLM only)", fontsize=11, color=INK)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
