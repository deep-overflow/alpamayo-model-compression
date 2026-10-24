"""fig6_head_token_map_q_head: which Q head earns its importance at which token position.
For every head (36 layers x 32 heads) the shipped score is split additively by the position type
at which the head's output enters the loss: C[t,l,u] = E_c[G_{c,t,l,u} sign(sum_t G_{c,t,l,u})],
sum_t C = I (exact). Colour = the head's dominant position type (vision / prompt-text / generated
CoC / ego-history; sink omitted). Left column: alpha = the head's importance percentile within its
layer (faint = low-ranked head). Middle: alpha = the dominant share C_max / I (how concentrated
the head's score is on one position type). Right: heads per layer by dominant type. Rows: I_traj
(layers 0-34; layer 35 is identically zero) and I_CoC (layers 0-35). Thin vertical lines mark the
8 KV groups (4 Q heads each); the dashed line marks the 21/22 band boundary.
Data: outputs/<run>/anatomy_perclip_q.npz. Usage: head_token_map.py [run=gradanat_v1] [suffix=""]."""
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
fc.OUT = OUT
VIS, HIST, PT, COC, SINK = range(5)
# colours as fig4_same_position_q_head (paper_same_position.py): vision BLUE, prompt text YELLOW, CoC PURPLE, ego history BRICK
TYPES = ((VIS, "vision", "#618BC8"), (PT, "prompt-text", "#ECAE3C"), (COC, "generated CoC", "#8E6BB3"), (HIST, "ego-history", "#C8615F"))
KEEP = 19  # heads kept per layer by the shipped uniform budget
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, 35)}
RUN = sys.argv[1] if len(sys.argv) > 1 else "gradanat_v1"  # gradient-anatomy run to read
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""          # appended to the file name
pq = np.load(REPO / "outputs" / RUN / "anatomy_perclip_q.npz")


def contributions(arr):
    """(N, 5, L, U) signed typed gradients -> C (5, L, U) additive parts and I (L, U) = sum_t C."""
    g = arr.astype(np.float64)
    C = (g * np.sign(g.sum(1, keepdims=True))).mean(0)
    return C, C.sum(0)


def analyse(arr, last):
    C, I = contributions(arr)
    C4 = np.stack([C[t] for t, _, _ in TYPES])  # (4, L, U)
    dom = C4.argmax(0)  # (L, U) index into TYPES
    dom_share = np.clip(C4.max(0) / np.where(I > 0, I, np.nan), 0, 1)
    pct = np.full(I.shape, np.nan)
    for l in range(last):
        pct[l] = np.argsort(np.argsort(I[l])) / (I.shape[1] - 1)
    return dict(C4=C4, I=I, dom=dom, dom_share=dom_share, pct=pct, last=last)


def rgba(dom, alpha, last):
    img = np.zeros(dom.shape + (4,))
    for k, (_, _, col) in enumerate(TYPES):
        img[dom == k, :3] = to_rgb(col)
    a = np.nan_to_num(alpha, nan=0.0)
    img[..., 3] = 0.12 + 0.88 * a
    img[last:] = (0.85, 0.85, 0.85, 1.0)  # layers without a score
    return img


res = {"traj": analyse(pq["fm_full"], 35), "coc": analyse(pq["ce"], 36)}
label = {"traj": r"$I_{\mathrm{traj}}$", "coc": r"$I_{\mathrm{CoC}}$"}
L, U = 36, 32
fig, axes = plt.subplots(2, 3, figsize=(fc.SINGLE[0] * 2.7, fc.SINGLE[1] * 2.6), gridspec_kw={"width_ratios": [1, 1, 0.75]})
for r, (obj, d) in enumerate(res.items()):
    for c, (alpha, title) in enumerate(((d["pct"], "alpha = importance rank within layer"),
                                        (d["dom_share"], "alpha = dominant share"))):
        ax = axes[r, c]
        ax.imshow(rgba(d["dom"], alpha, d["last"]), origin="upper", extent=(-0.5, U - 0.5, L - 0.5, -0.5), interpolation="nearest", aspect="auto")
        for x in np.arange(3.5, U - 1, 4):
            ax.axvline(x, color="white", lw=0.6)
        ax.axhline(21.5, color="black", lw=0.8, ls=(0, (4, 2)))
        ax.set_xticks([0, 7, 15, 23, 31])
        ax.set_yticks([0, 7, 14, 21, 28, 35])
        ax.set_xlabel("Q head index")
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
    ax.set_xlabel("Heads per layer by dominant position")
    ax.set_title("heads per layer by dominant position", fontsize=6.5)
handles = [Patch(color=col, label=name) for _, name, col in TYPES]
fig.tight_layout(pad=0.8, rect=(0, 0, 1, 0.95))  # leave the top strip for the figure legend
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.995),
           title="colour = dominant position type", title_fontsize=7)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):  # not fc.save: its tight_layout ignores figure-level legends
    fig.savefig(OUT / f"fig6_head_token_map_q_head{SUFFIX}.{ext}", dpi=300)
plt.close(fig)
print(f"-> {OUT / ('fig6_head_token_map_q_head' + SUFFIX + '.png')}")

# ------------------------------------------------------------------ numbers
names = [n for _, n, _ in TYPES]
print("\nshare of heads by dominant position type (all heads / top-19 by own rank), and median dominant share")
for obj, d in res.items():
    for b, sl in BANDS.items():
        dom, I, ds = d["dom"][sl], d["I"][sl], d["dom_share"][sl]
        top = np.zeros_like(dom, bool)
        for i in range(dom.shape[0]):
            top[i, np.argsort(I[i])[::-1][:KEEP]] = True
        allc = [(dom == k).mean() for k in range(4)]
        topc = [(dom[top] == k).mean() for k in range(4)]
        print(f"  {obj:4s} L{b:5s} all: " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, allc)) + " | top-19: " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, topc)) + f" | median dominant share {np.nanmedian(ds):.2f}")
print("\nsame dominant type under both objectives (head-wise)")
for b, sl in BANDS.items():
    same = (res["traj"]["dom"][sl] == res["coc"]["dom"][sl]).mean()
    both_top = np.zeros((sl.stop - sl.start, U), bool)
    for i, l in enumerate(range(sl.start, sl.stop)):
        both_top[i] = np.isin(np.arange(U), np.argsort(res["traj"]["I"][l])[::-1][:KEEP]) & np.isin(np.arange(U), np.argsort(res["coc"]["I"][l])[::-1][:KEEP])
    same_top = (res["traj"]["dom"][sl][both_top] == res["coc"]["dom"][sl][both_top]).mean()
    print(f"  L{b:5s}: all heads {same:.2f} | heads in both top-19 sets {same_top:.2f} (n={both_top.sum()})")
# cross-type cancellation check: how often is the dominant share > 1 before clipping (other types negative)
for obj, d in res.items():
    raw = d["C4"].max(0) / np.where(d["I"] > 0, d["I"], np.nan)
    print(f"  {obj}: heads with negative contribution from some type (dominant share > 1 before clipping): {np.nanmean(raw > 1):.2f}; min share {np.nanmin(raw):.2f}")
