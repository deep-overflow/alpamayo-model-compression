"""fig5_band_mix: why the trajectory score alone is not enough even for driving. Trunk (layers
0-21) selected by one rule, late band (22-35) by another, applied to the dense model on the same
100 held-out clips as the band ablation (outputs/armmix_v1/metrics.json, mean paired delta vs
dense with 95% bootstrap CI). Left: FM loss change; right: minADE change."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402
import fig_why_differs as fw  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
fc.OUT = OUT
m = json.loads((REPO / "outputs" / "armmix_v1" / "metrics.json").read_text())["configs"]
ROWS = (("dual", "union, all layers", fw.GREEN),
        ("traj", "traj only, all layers", fc.TRAJ),
        ("traj_trunk", "traj trunk, late intact", fc.TRAJ),
        ("traj_trunk__dual_late", "traj trunk + union late", fc.TRAJ),
        ("traj_trunk__coc_late", "traj trunk + CoC late", fc.TRAJ),
        ("traj_trunk__rand_late", "traj trunk + random late", fc.TRAJ))
fig, axes = plt.subplots(1, 2, figsize=(fc.SINGLE[0] * 2.0, fc.SINGLE[1]), sharey=True)
y = np.arange(len(ROWS))[::-1]
for ax, (key, scale, xlabel) in zip(axes, (("fm", 100.0, "FM loss change (%)"), ("ade", 1.0, "minADE change (m)"))):
    for yi, (cfg, label, col) in zip(y, ROWS):
        d = m[cfg][key]
        v = d["rel"] * scale if key == "fm" else d["delta"]
        lo, hi = ((np.array(d["ci"]) / (d["delta"] / d["rel"])) * scale if key == "fm" else np.array(d["ci"]))
        ax.barh(yi, v, color=col, alpha=0.85 if cfg in ("dual", "traj") else 0.45, height=0.62)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
        ax.text(max(hi, 0) + (0.6 if key == "fm" else 0.05), yi, f"{v:+.1f}" if key == "fm" else f"{v:+.2f}",
                va="center", fontsize=6)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, (32 if key == "fm" else 3.9))
    ax.grid(axis="x", lw=0.4, alpha=0.5)
axes[0].set_yticks(y)
axes[0].set_yticklabels([r[1] for r in ROWS], fontsize=6.5)
fig.subplots_adjust(wspace=0.08)
fc.save(fig, "fig5_band_mix")
for cfg, label, _ in ROWS:
    print(f"{label:28s} FM {m[cfg]['fm']['rel'] * 100:+6.1f}%  NLL {m[cfg]['nll']['rel'] * 100:+7.1f}%  minADE {m[cfg]['ade']['delta']:+.2f} m")
print(f"-> {OUT / 'fig5_band_mix.png'}")
