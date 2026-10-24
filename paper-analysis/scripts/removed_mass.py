"""fig5_removed_mass_by_layer: does a uniform per-layer ratio remove a uniform share of each
layer's importance? For every VLM layer, the fraction of the layer's I_traj / I_CoC mass carried by
the units the shipped dual checkpoint removes (19/32 heads and 7,390/12,288 channels kept in every
layer). Data: outputs/importance_v2/importance.npz, outputs/slim_dual_u40_v2/slim_meta.json."""
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
z = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")
meta = json.loads((REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json").read_text())["vlm"]
RATIO = 1 - 19 / 32  # = 1 - 7390 / 12288 up to rounding: the uniform removal ratio
L = np.arange(fw.LAST)

fig, axes = plt.subplots(1, 2, figsize=(fc.SINGLE[0] * 2.0, fc.SINGLE[1]), sharey=True)
for ax, (name, title) in zip(axes, (("q", "Q heads"), ("mlp", "MLP channels"))):
    ax.axvspan(21.5, 35.5, color=fc.GREY, alpha=0.13, lw=0)
    ax.axhline(RATIO, color="black", lw=0.8, ls=":", label="parameters removed (40%)")
    for obj, col, ls, label in (("traj", fc.TRAJ, "-", r"$I_{\mathrm{traj}}$"), ("coc", fc.COC, (0, (4, 2)), r"$I_{\mathrm{CoC}}$")):
        s = z[f"{obj}_vlm_{name}"]  # (36, U)
        removed = []
        for li in L:
            keep = np.zeros(s.shape[1], bool)
            keep[np.array(meta[li][name])] = True
            removed.append(s[li][~keep].sum() / s[li].sum())
        removed = np.array(removed)
        ax.plot(L, removed, color=col, lw=1.4, ls=ls, label=label)
        print(f"{title:12s} {obj}: removed share mean {removed.mean():.3f}  L6-34 range [{removed[6:].min():.2f}, {removed[6:].max():.2f}]  L1-5 mean {removed[1:6].mean():.3f}")
    ax.set_xlabel("VLM layer")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0, 0.5)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=8)
axes[0].set_ylabel("Share of layer importance removed")
axes[0].legend(loc="lower right", fontsize=6)
fig.subplots_adjust(wspace=0.08)
fc.save(fig, "fig5_removed_mass_by_layer")
print(f"-> {OUT / 'fig5_removed_mass_by_layer.png'}")
