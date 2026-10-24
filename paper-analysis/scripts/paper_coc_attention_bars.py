"""Paper-style fig8b: attention mass of generated-CoC queries by key group, for the Q heads the two
scores rank differently at CoC positions (per layer, the 6 largest and 6 smallest
rank(I_traj@CoC) - rank(I_CoC@CoC)). Two panels (layers 0-21, 22-34), clip-bootstrap 95% CIs
(drawn with vlines: paper_style.export forces every Line2D solid, which would turn errorbar anchor
lines into a polyline), stars = post hoc two-sided Mann-Whitney over heads. Style as
fig_criterion.py --paper-style (12 pt labels, 10 pt text, BLUE = I_traj side, YELLOW = I_CoC side).
Data: outputs/coc_census_v1_s*/census.npz + outputs/gradanat_v1/anatomy_perclip_q.npz.
Usage: python paper_coc_attention_bars.py [suffix=""]"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from paper_style import BLUE, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
SHARDS = [f"coc_census_v1_s{i}" for i in range(4)]
GROUPS = ("sink", "vision", "hist", "prompt", "coc_prev", "self")
LABELS = ("Sink", "Vision", "Ego hist.", "Prompt", "Prev. CoC", "Self")
COC = 3
K = 6
rng = np.random.default_rng(0)

clips, arrs = [], []
for s in SHARDS:
    clips += json.loads((REPO / "outputs" / s / "metrics.json").read_text())["clip_ids"]
    arrs.append(np.load(REPO / "outputs" / s / "census.npz")["mass_by_clip"])
mass = np.concatenate(arrs)  # (N, L, H, 6)
M = mass.mean(0)
pq = np.load(REPO / "outputs" / "gradanat_v1" / "anatomy_perclip_q.npz")
Tc = np.abs(pq["fm_full"][:, COC].astype(np.float64)).mean(0)
Cc = np.abs(pq["ce"][:, COC].astype(np.float64)).mean(0)


def rk(x):
    return np.argsort(np.argsort(x, axis=1), axis=1) / (x.shape[1] - 1)


diff = rk(Tc) - rk(Cc)

fig, axes = plt.subplots(1, 2, figsize=(SINGLE[0] * 2.0, SINGLE[1]), sharey=True)
for ax, (bname, layers) in zip(axes, (("0–21", range(0, 22)), ("22–34", range(22, 35)))):
    sel = {}
    for name in ("T", "C"):
        hs = []
        for l in layers:
            order = np.argsort(diff[l])
            hs += [(l, h) for h in (order[-K:] if name == "T" else order[:K])]
        sel[name] = (np.array([l for l, _ in hs]), np.array([h for _, h in hs]))
    x = np.arange(len(GROUPS))
    means = {}
    for i, (name, col, lab) in enumerate((("T", BLUE, r"$I_{\mathrm{traj}}$-favored heads"), ("C", YELLOW, r"$I_{\mathrm{CoC}}$-favored heads"))):
        L_, H_ = sel[name]
        boot = np.array([mass[rng.integers(0, len(mass), len(mass))].mean(0)[L_, H_].mean(0) for _ in range(500)])
        means[name] = M[L_, H_].mean(0)
        xs = x + (i - 0.5) * 0.36
        ax.bar(xs, means[name], 0.36, color=col, label=lab)
        lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
        ax.vlines(xs, lo, hi, color="black", lw=0.8)
        ax.hlines(np.r_[lo, hi], np.r_[xs, xs] - 0.06, np.r_[xs, xs] + 0.06, color="black", lw=0.8)
    for i in range(len(GROUPS)):
        a, b = M[sel["T"][0], sel["T"][1]][:, i], M[sel["C"][0], sel["C"][1]][:, i]
        p = mannwhitneyu(a, b, alternative="two-sided")[1]
        if p < 0.01:
            ax.text(x[i], max(means["T"][i], means["C"][i]) + 0.025, "**" if p < 1e-4 else "*", ha="center", va="bottom", fontsize=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 0.72)
    ax.text(0.03, 0.97, f"Layers {bname}", transform=ax.transAxes, ha="left", va="top", fontsize=TEXT)
axes[0].set_ylabel("Attention mass of CoC queries")
axes[0].legend(loc="lower center", bbox_to_anchor=(1.02, 1.0), ncol=2, fontsize=TEXT, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2, frameon=False)
export(fig, OUT, "fig8b_coc_attention_bars" + SUFFIX)
