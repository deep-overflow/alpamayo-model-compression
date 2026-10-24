"""fig6e_head_map_combined: fig6d (dominant position type, alpha = dominant share) and fig6c (rank
within layer, and the rank difference) in one figure. Top row: Q heads (x = head index, thin lines
= the 8 KV groups). Bottom row: MLP channels; in the two type panels the channels of every layer
are ordered by that score's own rank (left = most important; dotted line = the shipped cut, 7,390
kept of 12,288), in the rank and difference panels by the I_FM rank (so the I_FM rank panel is a
ramp by construction and the I_CE panel shows how far the CE ranking follows it). Fifth column:
rank(I_FM) - rank(I_CE) per unit (blue = the trajectory objective ranks the unit higher, yellow =
the CoC objective does). Layer 35 of I_FM is identically zero (grey); dashed line = the 21/22 band
boundary. Type colours as fig4_same_position_q_head.
Flags after the second argument: "diff" adds a fifth column, rank(I_FM) - rank(I_CE) per unit (blue = the trajectory
objective ranks the unit higher, yellow = the CoC objective does), with its own colorbar, and appends _diff to the
file name; "sortq" orders the Q heads of every layer by the I_FM rank in the rank (and difference) panels, as the
MLP row does (the KV-group lines are then dropped there), and appends _sortq; "idxmlp" leaves the MLP channels in
index order in every panel (no re-ordering, no cut line) and appends _idxmlp; "typerank" orders the MLP channels of
the two position-type panels by that score's own rank (left = most important, dotted line = the shipped cut) instead
of the default index order, and appends _typerank. The default is four columns.
Data: outputs/<run>/anatomy_perclip_{q,mlp}.npz. Usage: head_map_combined.py [run=gradanat_v1] [suffix=""] [nodiff]."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "gradanat_v1"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
NODIFF = "diff" not in sys.argv[3:]
SORTQ = "sortq" in sys.argv[3:]
IDXMLP = "idxmlp" in sys.argv[3:]
TYPERANK = "typerank" in sys.argv[3:] and not IDXMLP
NCOL = 4 if NODIFF else 5
VIS, HIST, PT, COC, SINK = range(5)
TYPES = ((VIS, "Vision", "#618BC8"), (PT, "Prompt text", "#ECAE3C"), (COC, "Generated CoC", "#8E6BB3"), (HIST, "Ego history", "#C8615F"))
AXES = {"q": ("anatomy_perclip_q.npz", "fm_full", "ce", 19), "mlp": ("anatomy_perclip_mlp.npz", "fm_type", "ce_type", 7390)}
GREY = (0.85, 0.85, 0.85)
LABEL, TICK, LEGEND, TITLE = 15, 12, 15, 15  # font sizes (pt) on the 12.6 x 7.0 in canvas
DIV = LinearSegmentedColormap.from_list("ce_fm", [fc.COC, "white", fc.TRAJ])


def analyse(arr, last):
    g = arr.astype(np.float32)
    C = (g * np.sign(g.sum(1, keepdims=True))).mean(0).astype(np.float64)
    I = C.sum(0)
    C4 = np.stack([C[t] for t, _, _ in TYPES])
    dom, share = C4.argmax(0), np.clip(C4.max(0) / np.where(I > 0, I, np.nan), 0, 1)
    rank = np.full(I.shape, np.nan)
    for l in range(last):
        rank[l] = np.argsort(np.argsort(I[l])) / (I.shape[1] - 1)
    order = np.stack([np.argsort(-I[l]) for l in range(I.shape[0])])
    return dict(dom=dom, share=share, rank=rank, order=order, last=last)


def type_img(d, reorder):
    L, U = d["dom"].shape
    img = np.zeros((L, U, 4))
    for k, (_, _, col) in enumerate(TYPES):
        img[d["dom"] == k, :3] = to_rgb(col)
    img[..., 3] = 0.12 + 0.88 * np.nan_to_num(d["share"], nan=0.0)
    if reorder:
        img = np.stack([img[l, d["order"][l]] for l in range(L)])
    img[d["last"]:] = (*GREY, 1.0)
    return img


data = {}
for axis, (fn, kt, kc, keep) in AXES.items():
    z = np.load(REPO / "outputs" / RUN / fn)
    data[axis] = {"fm": analyse(z[kt], 35), "ce": analyse(z[kc], 36), "keep": keep}

fig, axes = plt.subplots(2, NCOL, figsize=(fc.SINGLE[0] * (4.2 if NODIFF else 5.1), fc.SINGLE[1] * 2.9))
titles = (r"$I_{\mathrm{FM}}$: position type", r"$I_{\mathrm{CE}}$: position type",
          r"$I_{\mathrm{FM}}$: rank in layer", r"$I_{\mathrm{CE}}$: rank in layer",
          r"rank($I_{\mathrm{FM}}$) $-$ rank($I_{\mathrm{CE}}$)")
ims = {}
for r, axis in enumerate(("q", "mlp")):
    d, keep = data[axis], data[axis]["keep"]
    L, U = d["fm"]["dom"].shape
    ext = (-0.5, U - 0.5, -0.5, L - 0.5)  # layer 0 at the bottom, layer 35 at the top
    sort_rank = (axis == "mlp" and not IDXMLP) or (axis == "q" and SORTQ)  # rank / difference panels ordered by the I_FM rank
    ordered = (lambda M: np.stack([M[l, d["fm"]["order"][l]] for l in range(L)])) if sort_rank else (lambda M: M)
    panels = [type_img(d["fm"], axis == "mlp" and TYPERANK), type_img(d["ce"], axis == "mlp" and TYPERANK),
              np.ma.masked_invalid(ordered(d["fm"]["rank"])), np.ma.masked_invalid(ordered(d["ce"]["rank"])),
              np.ma.masked_invalid(ordered(d["fm"]["rank"] - d["ce"]["rank"]))]
    for c, P in enumerate(panels[:NCOL]):
        ax = axes[r, c]
        if c < 2:
            ax.imshow(P, origin="lower", extent=ext, interpolation="nearest", aspect="auto")
        elif c < 4:
            ims["rank"] = ax.imshow(P, origin="lower", extent=ext, interpolation="nearest", aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ims["rank"].cmap.set_bad(GREY)
        else:
            ims["diff"] = ax.imshow(P, origin="lower", extent=ext, interpolation="nearest", aspect="auto", cmap=DIV, vmin=-1, vmax=1)
            ims["diff"].cmap.set_bad(GREY)
        ax.axhline(21.5, color="black", lw=0.8, ls=(0, (4, 2)))
        ax.set_yticks([0, 7, 14, 21, 28, 35])
        if axis == "q":
            if not (SORTQ and c >= 2):
                for x in np.arange(3.5, U - 1, 4):
                    ax.axvline(x, color="white", lw=0.5)
            ax.set_xticks([0, 7, 15, 23, 31])
            ax.set_xlabel(r"Q head ($I_{\mathrm{FM}}$ rank)" if (SORTQ and c >= 2) else "Q head index")
            ax.set_title(titles[c], fontsize=TITLE)
        else:
            if c < 2 and TYPERANK:
                ax.axvline(keep - 0.5, color="black", lw=0.6, ls=":")
            ax.set_xticks([0, 3072, 6144, 9216, 12287])
            ax.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
            ax.set_xlabel("MLP ch. index" if (IDXMLP or (c < 2 and not TYPERANK)) else ("MLP ch. (own rank)" if c < 2 else r"MLP ch. ($I_{\mathrm{FM}}$ rank)"))
        if c == 0:
            ax.set_ylabel("Layer (" + ("Q heads" if axis == "q" else "MLP") + ")")
for ax in axes.flat:
    ax.xaxis.label.set_size(LABEL)
    ax.yaxis.label.set_size(LABEL)
    ax.tick_params(axis="both", labelsize=TICK)
handles = [Patch(color=col, label=name) for _, name, col in TYPES]
fig.tight_layout(pad=0.8, rect=(0, 0.08, 0.925 if NODIFF else 0.93, 1))
cax1 = fig.add_axes([0.932, 0.20, 0.010, 0.62] if NODIFF else [0.937, 0.56, 0.008, 0.34])
cb1 = fig.colorbar(ims["rank"], cax=cax1)
cb1.set_label("Rank in layer", fontsize=LABEL)
cb1.ax.tick_params(labelsize=TICK)
if not NODIFF:
    cax2 = fig.add_axes([0.937, 0.12, 0.008, 0.34])
    cb2 = fig.colorbar(ims["diff"], cax=cax2)
    cb2.set_label(r"rank($I_{\mathrm{FM}}$) $-$ rank($I_{\mathrm{CE}}$)", fontsize=LABEL)
    cb2.ax.tick_params(labelsize=TICK)
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=LEGEND, frameon=False, bbox_to_anchor=(0.475, 0.0), handlelength=1.4, columnspacing=1.6)
NAME = "fig6e_head_map_combined" + ("_sortq" if SORTQ else "") + ("_idxmlp" if IDXMLP else "") + ("_typerank" if TYPERANK else "") + ("" if NODIFF else "_diff")
OUT.mkdir(parents=True, exist_ok=True)
for ext_ in ("png", "pdf"):
    fig.savefig(OUT / f"{NAME}{SUFFIX}.{ext_}", dpi=300)
plt.close(fig)
print(f"-> {OUT / (NAME + SUFFIX + '.png')}")
