"""fig6c_head_map_plain: the shipped importance maps WITHOUT the position-type split. Colour =
importance rank within the layer (0 = least important, 1 = most). Top row: Q heads (36 x 32, x =
head index; thin lines = the 8 KV groups). Bottom row: MLP channels (36 x 12,288); since channel
index carries no meaning, the channels of every layer are ordered by their I_traj rank (left = most
important under I_traj), so the I_traj panel is a ramp by construction and the I_CoC panel shows
how far the CoC ranking follows it. Third column: rank(I_traj) - rank(I_CoC) per unit (blue = the
trajectory objective ranks the unit higher, yellow = the CoC objective does). Layer 35 of I_traj
is identically zero (grey). Dashed line = the 21/22 band boundary.
Data: outputs/<run>/anatomy_perclip_{q,mlp}.npz. Usage: head_map_plain.py [run=gradanat_v1] [suffix=""]."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "gradanat_v1"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
DIV = LinearSegmentedColormap.from_list("coc_traj", [fc.COC, "white", fc.TRAJ])


def score(arr):
    g = arr.astype(np.float32)
    return np.abs(g.sum(1)).mean(0).astype(np.float64)  # E_c |sum_t G| = the shipped score, (L, U)


def rank_map(I, last):
    pct = np.full(I.shape, np.nan)
    for l in range(last):
        pct[l] = np.argsort(np.argsort(I[l])) / (I.shape[1] - 1)
    return pct


pq = np.load(REPO / "outputs" / RUN / "anatomy_perclip_q.npz")
pm = np.load(REPO / "outputs" / RUN / "anatomy_perclip_mlp.npz")
data = {}
for axis, (kt, kc, z) in (("q", ("fm_full", "ce", pq)), ("mlp", ("fm_type", "ce_type", pm))):
    It, Ic = score(z[kt]), score(z[kc])
    rt, rc = rank_map(It, 35), rank_map(Ic, 36)
    order = np.stack([np.argsort(-It[l]) if l < 35 else np.arange(It.shape[1]) for l in range(36)])
    data[axis] = dict(rt=rt, rc=rc, order=order)

fig, axes = plt.subplots(2, 3, figsize=(fc.SINGLE[0] * 2.7, fc.SINGLE[1] * 2.6))
for r, axis in enumerate(("q", "mlp")):
    d = data[axis]
    L, U = d["rt"].shape
    panels = ((d["rt"], r"rank of $I_{\mathrm{traj}}$ within layer", "viridis", (0, 1)),
              (d["rc"], r"rank of $I_{\mathrm{CoC}}$ within layer", "viridis", (0, 1)),
              (d["rt"] - d["rc"], r"rank($I_{\mathrm{traj}}$) $-$ rank($I_{\mathrm{CoC}}$)", DIV, (-1, 1)))
    for c, (M, title, cmap, lim) in enumerate(panels):
        ax = axes[r, c]
        img = M if axis == "q" else np.stack([M[l, d["order"][l]] for l in range(L)])
        im = ax.imshow(np.ma.masked_invalid(img), origin="upper", extent=(-0.5, U - 0.5, L - 0.5, -0.5), interpolation="nearest",
                       aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1])
        im.cmap.set_bad((0.85, 0.85, 0.85))
        ax.axhline(21.5, color="black", lw=0.8, ls=(0, (4, 2)))
        ax.set_yticks([0, 7, 14, 21, 28, 35])
        if axis == "q":
            for x in np.arange(3.5, U - 1, 4):
                ax.axvline(x, color="white", lw=0.5)
            ax.set_xticks([0, 7, 15, 23, 31])
            ax.set_xlabel("Q head index")
        else:
            ax.set_xticks([0, 3072, 6144, 9216, 12287])
            ax.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
            ax.set_xlabel(r"MLP channel, ordered by $I_{\mathrm{traj}}$ rank")
        if c == 0:
            ax.set_ylabel("VLM layer   (" + ("Q heads" if axis == "q" else "MLP channels") + ")")
        ax.set_title(title, fontsize=6.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6)
fig.tight_layout(pad=0.8)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig6c_head_map_plain{SUFFIX}.{ext}", dpi=300)
plt.close(fig)
print(f"-> {OUT / ('fig6c_head_map_plain' + SUFFIX + '.png')}")
for axis in ("q", "mlp"):
    d = data[axis]
    for b, sl in (("0-21", slice(0, 22)), ("22-34", slice(22, 35))):
        diff = (d["rt"] - d["rc"])[sl]
        print(f"  {axis:3s} L{b}: mean |rank diff| {np.nanmean(np.abs(diff)):.3f}, share of units with |diff| > 0.5: {np.nanmean(np.abs(diff) > 0.5):.3f}")
