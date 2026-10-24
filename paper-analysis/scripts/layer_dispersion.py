"""fig5_layer_dispersion: how unevenly is importance spread within each VLM layer, and would an
allocation that equalises the removed importance share across layers differ from a uniform one?
Left: Gini coefficient of the unit importances within each layer (0 = all units equal, 1 = one unit
holds everything), per objective and unit type. Right: the per-layer removal ratio that would remove
the same share of the layer's importance as the uniform 40% cut removes on average over layers 6-34
(own-ranking bottom cut), i.e. the "importance-equalising" allocation; dotted line = uniform.
Data: outputs/importance_v2/importance.npz."""
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
RATIO = 1 - 19 / 32
L = np.arange(fw.LAST)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() <= 0:
        return np.nan
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n


def removed_share(x, ratio):
    """share of the layer's importance in the bottom `ratio` fraction of units (own ranking)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    k = int(round(ratio * len(x)))
    return x[:k].sum() / x.sum()


def equalising_ratio(x, target):
    """smallest removal ratio whose bottom cut removes at least `target` of the layer's importance."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    cum = np.cumsum(x) / x.sum()
    k = int(np.searchsorted(cum, target)) + 1
    return min(k, len(x)) / len(x)


fig, axes = plt.subplots(1, 2, figsize=(fc.SINGLE[0] * 2.0, fc.SINGLE[1]))
styles = {("traj", "q"): (fc.TRAJ, "-", r"$I_{\mathrm{traj}}$, Q heads"), ("coc", "q"): (fc.COC, "-", r"$I_{\mathrm{CoC}}$, Q heads"),
          ("traj", "mlp"): (fc.TRAJ, (0, (4, 2)), r"$I_{\mathrm{traj}}$, MLP"), ("coc", "mlp"): (fc.COC, (0, (4, 2)), r"$I_{\mathrm{CoC}}$, MLP")}
for ax in axes:
    ax.axvspan(21.5, 35.5, color=fc.GREY, alpha=0.13, lw=0)
    ax.set_xlabel("VLM layer")
    ax.set_xlim(-0.5, 35.5)
axes[1].axhline(RATIO, color="black", lw=0.8, ls=":", label="uniform (40%)")
print("layer dispersion (layers 0-34)")
for (obj, name), (col, ls, label) in styles.items():
    s = z[f"{obj}_vlm_{name}"]
    g = np.array([gini(s[li]) for li in L])
    top = np.array([np.sort(s[li])[::-1][: max(1, round(0.1 * s.shape[1]))].sum() / s[li].sum() for li in L])
    rem = np.array([removed_share(s[li], RATIO) for li in L])
    target = rem[6:].mean()
    eq = np.array([equalising_ratio(s[li], target) for li in L])
    axes[0].plot(L, g, color=col, ls=ls, lw=1.3, label=label)
    axes[1].plot(L, eq, color=col, ls=ls, lw=1.3, label=label)
    print(f"  {label:22s} Gini L6-34 mean {g[6:].mean():.2f} range [{g[6:].min():.2f}, {g[6:].max():.2f}] | L1-5 {g[1:6].mean():.2f}"
          f" | top-10% share L6-34 [{top[6:].min():.2f}, {top[6:].max():.2f}] | removed@40% L6-34 mean {target:.3f}"
          f" | equalising ratio L6-34 [{eq[6:].min():.2f}, {eq[6:].max():.2f}] (sd {eq[6:].std():.3f}), L1-5 [{eq[1:6].min():.2f}, {eq[1:6].max():.2f}]")
axes[0].set_ylabel("Gini coefficient within layer")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=6, loc="upper left")
axes[1].set_ylabel("Ratio removing the mean importance share")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=6, loc="upper right")
fig.subplots_adjust(wspace=0.3)
fc.save(fig, "fig5_layer_dispersion")
print(f"-> {OUT / 'fig5_layer_dispersion.png'}")
