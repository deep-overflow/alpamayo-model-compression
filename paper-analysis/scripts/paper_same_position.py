"""Paper-style fig4_same_position_{q_head,mlp}: ceiling-corrected within-layer rank agreement
between I_traj and I_CoC with BOTH scores restricted to one kind of token position (vision /
prompt text / generated CoC / ego history) next to the pooled (shipped) agreement. Same data and
computation as paper-analysis/scripts/same_token5.py (stored P2 series of the anatomy analysis;
missing series recomputed with the analyzer's split-half procedure in the same rng order), styled
like fig1_rank_agreement (experiments/paper/fig_criterion.py --paper-style).
A fourth argument selects "raw" (default: Spearman between the two 100-clip mean scores, the value the
paper quotes) or "corrected" (split-half ceiling-corrected, appends _corrected to the file name).
Usage: python paper_same_position.py [run=gradanat_v1] [suffix=""] [axis=q|mlp] [raw|corrected]."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from paper_style import BLUE, BRICK, GREY, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "gradanat_v1"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
AXIS = sys.argv[3] if len(sys.argv) > 3 else "q"
MODE = sys.argv[4] if len(sys.argv) > 4 else "raw"
PURPLE = "#8E6BB3"  # fifth colour, not in the paper palette (generated-CoC positions)
VIS, HIST, PT, COC, SINK = range(5)
LAST = 35
N_SPLIT = {"q": 50, "mlp": 10}[AXIS]
PERCLIP = {"q": ("anatomy_perclip_q.npz", "fm_full", "ce"), "mlp": ("anatomy_perclip_mlp.npz", "fm_type", "ce_type")}[AXIS]
NAME = {"q": "fig4_same_position_q_head", "mlp": "fig4_same_position_mlp"}[AXIS]


def rho_layers(a, b):
    out = np.full(a.shape[0], np.nan)
    for li in range(a.shape[0]):
        if np.ptp(a[li]) > 0 and np.ptp(b[li]) > 0:
            out[li] = spearmanr(a[li], b[li])[0]
    return out


def split_half(t, c, n_split, rng):
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)
        st.append(rho_layers(ta, tb)); sc_.append(rho_layers(ca, cb)); cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def corrected(st, sc_, cr):
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


m = json.loads((REPO / "outputs" / RUN / "metrics_analysis.json").read_text())["P2"][AXIS]
stored = {"All": "pooled (the shipped score)", "Vision": "both at VISION positions",
          "Prompt text": "both at prompt-text positions only", "CoC": "both at CoC positions only", "Ego history": None}
series = {}
with np.load(REPO / "outputs" / RUN / PERCLIP[0]) as pq:
    fm, ce = pq[PERCLIP[1]], pq[PERCLIP[2]]  # (N, 5, 36, U) signed per-clip typed gradients
    if MODE == "raw":
        fm32, ce32 = fm.astype(np.float32), ce.astype(np.float32)
        series["All"] = rho_layers(np.abs(fm32.sum(1)).mean(0), np.abs(ce32.sum(1)).mean(0))
        for label, ty in (("Vision", VIS), ("Prompt text", PT), ("CoC", COC), ("Ego history", HIST)):
            series[label] = rho_layers(np.abs(fm32[:, ty]).mean(0), np.abs(ce32[:, ty]).mean(0))
    rng = np.random.default_rng(0)
    if MODE == "raw":
        pass
    else:
      split_half(np.abs(fm[:, VIS])[:, :LAST], np.abs(ce[:, VIS])[:, :LAST], N_SPLIT, rng)  # same rng order as same_token5.py
      for label, key in stored.items():
        if key in m:
            series[label] = corrected(*(np.array(m[key]["per_layer"][k]) for k in ("self_traj", "self_coc", "cross")))
        else:
            ty = {"Prompt text": PT, "Ego history": HIST}[label]
            series[label] = corrected(*split_half(np.abs(fm[:, ty])[:, :LAST], np.abs(ce[:, ty])[:, :LAST], N_SPLIT, rng))

colors = {"All": "black", "Vision": BLUE, "Prompt text": YELLOW, "CoC": PURPLE, "Ego history": BRICK}
L = np.arange(LAST)
fig, ax = plt.subplots(figsize=SINGLE)
ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
for label, y in series.items():
    ax.plot(L, y[:LAST], color=colors[label], lw=1.8 if label == "All" else 1.3, label=label)
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("VLM layer index")
ax.set_ylabel(r"Rank agreement $\rho$" if MODE == "raw" else r"Corrected $\rho$")
ax.set_xlim(-0.5, 35.5)
ax.set_xticks([0, 7, 14, 21, 28, 35])
ax.set_ylim(-0.25, 1.05)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=TEXT, handlelength=1.0,
          handletextpad=0.4, columnspacing=0.7, borderaxespad=0.2)
export(fig, OUT, NAME + ("" if MODE == "raw" else "_corrected") + SUFFIX)
late = slice(22, LAST)
print("  late-band (22-34) means: " + ", ".join(f"{k} {np.nanmean(v[late]):.2f}" for k, v in series.items()))
