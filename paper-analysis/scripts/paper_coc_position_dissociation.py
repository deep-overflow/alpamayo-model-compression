"""Paper-style fig10: double dissociation at generated-CoC positions. Per layer the k = 8 Q heads
with the largest rank(I_traj@CoC) - rank(I_CoC@CoC) (T) or the smallest (C), and random sets of
k (R, mean of three), are switched off at the CoC positions only, on 100 held-out clips; bars are
the paired change relative to the unmasked model in CoC NLL, FM loss and minADE (K = 8), with
clip-bootstrap 95% CIs (vlines: paper_style.export forces every Line2D solid). Left: layers 0-21;
right: all layers 0-34. Data: outputs/coc_posabl_v1_s*/metrics.json (run B of
plans/2026-09-25_coc-position-content.md). Style as fig_criterion.py --paper-style.
Usage: python paper_coc_position_dissociation.py [suffix=""]"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import BLUE, GREY, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
SHARDS = [f"coc_posabl_v1_s{i}" for i in range(4)]
METRICS = (("nll", "CoC NLL"), ("fm", "FM loss"), ("ade", "minADE"))
rng = np.random.default_rng(0)

per = {}
for s in SHARDS:
    m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
    for cfg, v in m["per_clip"].items():
        for k, arr in v.items():
            per.setdefault(cfg, {}).setdefault(k, []).extend(arr)
dense = {k: np.array(per["dense"][k], float) for k, _ in METRICS}
base = {k: float(np.nanmean(dense[k])) for k in dense}


def rel(cfgs, key):
    """Paired relative change of the mean over clips (bootstrap CI), averaged over the given configs."""
    d = np.mean([np.array(per[c][key], float) - dense[key] for c in cfgs], 0)
    ok = np.isfinite(d)
    d = d[ok]
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]) / base[key]
    return d.mean() / base[key], np.percentile(boot, 2.5), np.percentile(boot, 97.5)


fig, axes = plt.subplots(1, 2, figsize=(SINGLE[0] * 2.0, SINGLE[1]))
for ax, (band, title) in zip(axes, (("trunk", "Layers 0–21"), ("all", "Layers 0–34"))):
    sets = ((r"$I_{\mathrm{traj}}$-favored", [f"T_{band}"], BLUE), (r"$I_{\mathrm{CoC}}$-favored", [f"C_{band}"], YELLOW),
            ("Random", [f"R{i}_{band}" for i in range(3)], GREY))
    x = np.arange(len(METRICS))
    for i, (lab, cfgs, col) in enumerate(sets):
        xs = x + (i - 1) * 0.27
        vals = []
        for key, _ in METRICS:
            use = [c for c in cfgs if np.isfinite(np.array(per[c][key], float)).any()]
            vals.append(rel(use, key) if use else (np.nan, np.nan, np.nan))
        v = np.array(vals)
        ax.bar(xs, v[:, 0] * 100, 0.27, color=col, label=lab if ax is axes[0] else None)
        ax.vlines(xs, v[:, 1] * 100, v[:, 2] * 100, color="black", lw=0.8)
        ax.hlines(np.r_[v[:, 1], v[:, 2]] * 100, np.r_[xs, xs] - 0.05, np.r_[xs, xs] + 0.05, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRICS])
    ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=TEXT)
axes[0].set_ylabel("Change, heads off at CoC (%)")
axes[0].legend(loc="lower center", bbox_to_anchor=(1.02, 1.0), ncol=3, fontsize=TEXT, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2, frameon=False)
export(fig, OUT, "fig10_coc_position_dissociation" + SUFFIX)
