"""fig6d_head_token_map_dom_{q_head,mlp}: the dominant-share panels of the head-token map alone,
I_FM (left) and I_CE (right), legend below. Colour = the unit's dominant position type (additive
split of the shipped score by the position type at which the unit's output enters the loss);
alpha = dominant share C_max / I. Q heads: x = head index (thin lines = the 8 KV groups). MLP:
channels of every layer ordered by the row's own importance rank (left = most important), dotted
line = the shipped cut (7,390 kept of 12,288). Layer 35 of I_FM is identically zero (grey); dashed
line = the 21/22 band boundary. Colours as fig4_same_position_q_head.
Data: outputs/<run>/anatomy_perclip_{q,mlp}.npz. Usage: head_token_map_dom.py [axis=q|mlp] [run=gradanat_v1] [suffix=""]."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
AXIS = sys.argv[1] if len(sys.argv) > 1 else "q"
RUN = sys.argv[2] if len(sys.argv) > 2 else "gradanat_v1"
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""
VIS, HIST, PT, COC, SINK = range(5)
TYPES = ((VIS, "Vision", "#618BC8"), (PT, "Prompt text", "#ECAE3C"), (COC, "Generated CoC", "#8E6BB3"), (HIST, "Ego history", "#C8615F"))
FILE, KT, KC, KEEP = {"q": ("anatomy_perclip_q.npz", "fm_full", "ce", 19), "mlp": ("anatomy_perclip_mlp.npz", "fm_type", "ce_type", 7390)}[AXIS]
z = np.load(REPO / "outputs" / RUN / FILE)


def analyse(arr, last):
    g = arr.astype(np.float32)
    C = (g * np.sign(g.sum(1, keepdims=True))).mean(0).astype(np.float64)  # (5, L, U)
    I = C.sum(0)
    C4 = np.stack([C[t] for t, _, _ in TYPES])
    dom = C4.argmax(0)
    share = np.clip(C4.max(0) / np.where(I > 0, I, np.nan), 0, 1)
    order = np.stack([np.argsort(-I[l]) for l in range(I.shape[0])])
    return dom, share, order, last


def rgba(dom, share, order, last, reorder):
    L, U = dom.shape
    img = np.zeros((L, U, 4))
    for k, (_, _, col) in enumerate(TYPES):
        img[dom == k, :3] = to_rgb(col)
    img[..., 3] = 0.12 + 0.88 * np.nan_to_num(share, nan=0.0)
    if reorder:
        img = np.stack([img[l, order[l]] for l in range(L)])
    img[last:] = (0.85, 0.85, 0.85, 1.0)
    return img


res = (analyse(z[KT], 35), analyse(z[KC], 36))
L, U = res[0][0].shape
fig, axes = plt.subplots(1, 2, figsize=(fc.SINGLE[0] * 2.0, fc.SINGLE[1] * 1.25))
for ax, (dom, share, order, last), title in zip(axes, res, (r"$I_{\mathrm{FM}}$", r"$I_{\mathrm{CE}}$")):
    ax.imshow(rgba(dom, share, order, last, AXIS == "mlp"), origin="lower", extent=(-0.5, U - 0.5, -0.5, L - 0.5), interpolation="nearest", aspect="auto")
    ax.axhline(21.5, color="black", lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks([0, 7, 14, 21, 28, 35])
    if AXIS == "q":
        for x in np.arange(3.5, U - 1, 4):
            ax.axvline(x, color="white", lw=0.6)
        ax.set_xticks([0, 7, 15, 23, 31])
    else:
        ax.axvline(KEEP - 0.5, color="black", lw=0.6, ls=":")
        ax.set_xticks([0, 3072, 6144, 9216, 12287])
        ax.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
    ax.set_title(title, fontsize=12)
axes[0].set_ylabel("VLM layer index")
axes[0].set_xlabel("Q head index" if AXIS == "q" else "MLP channel (ordered by own rank)")
axes[1].set_xlabel("Q head index" if AXIS == "q" else "MLP channel (ordered by own rank)")
handles = [Patch(color=col, label=name) for _, name, col in TYPES]
fig.tight_layout(pad=0.8, rect=(0, 0.1, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.6)
OUT.mkdir(parents=True, exist_ok=True)
name = f"fig6d_head_token_map_dom_{'q_head' if AXIS == 'q' else 'mlp'}{SUFFIX}"
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"{name}.{ext}", dpi=300)
plt.close(fig)
print(f"-> {OUT / (name + '.png')}")
