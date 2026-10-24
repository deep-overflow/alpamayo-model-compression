"""fig6b_head_token_map_mlp: the MLP-channel twin of fig6_head_token_map_q_head. For every channel
(36 layers x 12,288) the shipped score is split additively by the position type at which the
channel's output enters the loss (C[t,l,u] = E_c[G_{c,t,l,u} sign(sum_t G)], sum_t C = I, exact).
Colour = dominant position type (vision / prompt-text / generated CoC / ego-history; sink omitted).
Because 12,288 channels cannot be told apart by index, the channels of every layer are ORDERED BY
THE ROW'S OWN IMPORTANCE RANK (left = most important), so the maps read "which types are important
at which depth and rank". Left: alpha = importance rank within layer; middle: alpha = dominant
share C_max / I; right: channels per layer by dominant type. Rows: I_traj (layers 0-34; layer 35
identically zero) and I_CoC. The dashed line marks the 21/22 band boundary.
Data: outputs/<run>/anatomy_perclip_mlp.npz (fm_type, ce_type). Usage: head_token_map_mlp.py [run=gradanat_v1] [suffix=""]."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402
import fig_why_differs as fw  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
VIS, HIST, PT, COC, SINK = range(5)
# colours as fig4_same_position_q_head (paper_same_position.py): vision BLUE, prompt text YELLOW, CoC PURPLE, ego history BRICK
TYPES = ((VIS, "vision", "#618BC8"), (PT, "prompt-text", "#ECAE3C"), (COC, "generated CoC", "#8E6BB3"), (HIST, "ego-history", "#C8615F"))
KEEP = 7390  # channels kept per layer by the shipped uniform budget
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, 35)}
RUN = sys.argv[1] if len(sys.argv) > 1 else "gradanat_v1"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
pm = np.load(REPO / "outputs" / RUN / "anatomy_perclip_mlp.npz")


def contributions(arr):
    g = arr.astype(np.float32)
    C = (g * np.sign(g.sum(1, keepdims=True))).mean(0).astype(np.float64)  # (5, L, U)
    return C, C.sum(0)


def analyse(arr, last):
    C, I = contributions(arr)
    C4 = np.stack([C[t] for t, _, _ in TYPES])
    dom = C4.argmax(0)
    dom_share = np.clip(C4.max(0) / np.where(I > 0, I, np.nan), 0, 1)
    pct = np.full(I.shape, np.nan)
    order = np.zeros(I.shape, int)
    for l in range(I.shape[0]):
        order[l] = np.argsort(-I[l])  # most important first
        if l < last:
            pct[l] = np.argsort(np.argsort(I[l])) / (I.shape[1] - 1)
    return dict(C4=C4, I=I, dom=dom, dom_share=dom_share, pct=pct, order=order, last=last)


def rgba(dom, alpha, order, last):
    """Colour by dominant type, alpha by `alpha`, columns re-ordered per layer by `order`."""
    L, U = dom.shape
    img = np.zeros((L, U, 4))
    for k, (_, _, col) in enumerate(TYPES):
        img[dom == k, :3] = to_rgb(col)
    a = np.nan_to_num(alpha, nan=0.0)
    img[..., 3] = 0.12 + 0.88 * a
    img = np.stack([img[l, order[l]] for l in range(L)])
    img[last:] = (0.85, 0.85, 0.85, 1.0)
    return img


res = {"traj": analyse(pm["fm_type"], 35), "coc": analyse(pm["ce_type"], 36)}
label = {"traj": r"$I_{\mathrm{traj}}$", "coc": r"$I_{\mathrm{CoC}}$"}
L, U = res["coc"]["I"].shape
fig, axes = plt.subplots(2, 3, figsize=(fc.SINGLE[0] * 2.7, fc.SINGLE[1] * 2.6), gridspec_kw={"width_ratios": [1, 1, 0.75]})
for r, (obj, d) in enumerate(res.items()):
    for c, (alpha, title) in enumerate(((d["pct"], "alpha = importance rank within layer"),
                                        (d["dom_share"], "alpha = dominant share"))):
        ax = axes[r, c]
        ax.imshow(rgba(d["dom"], alpha, d["order"], d["last"]), origin="upper", extent=(-0.5, U - 0.5, L - 0.5, -0.5),
                  interpolation="nearest", aspect="auto")
        ax.axhline(21.5, color="black", lw=0.8, ls=(0, (4, 2)))
        ax.axvline(KEEP - 0.5, color="black", lw=0.6, ls=":")
        ax.set_xticks([0, 3072, 6144, 9216, 12287])
        ax.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
        ax.set_yticks([0, 7, 14, 21, 28, 35])
        ax.set_xlabel("MLP channel, ordered by own rank (0 = most important)")
        if c == 0:
            ax.set_ylabel(f"VLM layer   ({label[obj]})")
        ax.set_title(title, fontsize=6.5)
    ax = axes[r, 2]
    left = np.zeros(L)
    for k, (_, name, col) in enumerate(TYPES):
        cnt = (d["dom"] == k).sum(1).astype(float)
        cnt[d["last"]:] = 0
        ax.barh(np.arange(L), cnt, left=left, color=col, height=0.85, label=name)
        left += cnt
    ax.axhline(21.5, color="black", lw=0.8, ls=(0, (4, 2)))
    ax.set_ylim(L - 0.5, -0.5)
    ax.set_yticks([0, 7, 14, 21, 28, 35])
    ax.set_xlim(0, U)
    ax.set_xticks([0, 3072, 6144, 9216, 12288])
    ax.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
    ax.set_xlabel("Channels per layer by dominant position")
    ax.set_title("channels per layer by dominant position", fontsize=6.5)
handles = [Patch(color=col, label=name) for _, name, col in TYPES]
fig.tight_layout(pad=0.8, rect=(0, 0, 1, 0.95))
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.995),
           title="colour = dominant position type; dotted line = the shipped kept/removed cut (7,390 of 12,288)", title_fontsize=7)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig6b_head_token_map_mlp{SUFFIX}.{ext}", dpi=300)
plt.close(fig)
print(f"-> {OUT / ('fig6b_head_token_map_mlp' + SUFFIX + '.png')}")

names = [n for _, n, _ in TYPES]
print("\nshare of channels by dominant position type (all channels / top-7390 by own rank), and median dominant share")
for obj, d in res.items():
    for b, sl in BANDS.items():
        dom, I, ds = d["dom"][sl], d["I"][sl], d["dom_share"][sl]
        top = np.zeros_like(dom, bool)
        for i in range(dom.shape[0]):
            top[i, np.argsort(I[i])[::-1][:KEEP]] = True
        allc = [(dom == k).mean() for k in range(4)]
        topc = [(dom[top] == k).mean() for k in range(4)]
        print(f"  {obj:4s} L{b:5s} all: " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, allc)) + " | top-7390: " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, topc)) + f" | median dominant share {np.nanmedian(ds):.2f}")
print("\nsame dominant type under both objectives (channel-wise)")
for b, sl in BANDS.items():
    same = (res["traj"]["dom"][sl] == res["coc"]["dom"][sl]).mean()
    print(f"  L{b:5s}: all channels {same:.2f}")
