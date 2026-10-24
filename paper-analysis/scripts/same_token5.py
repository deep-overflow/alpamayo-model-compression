"""fig4_same_token_{q_head,mlp} split by token type: ceiling-corrected within-layer rank agreement
between I_traj and I_CoC with BOTH restricted to the same token type -- vision / prompt text /
generated CoC / ego history -- next to the pooled (shipped) agreement.

Series that analyze_gradient_anatomy.py already stored (P2 of outputs/gradanat_v1/metrics_analysis.json;
50 split-half draws for Q heads, 10 for MLP channels) are read from there. The missing ones
(ego history on both axes, prompt text on MLP) are computed here from the per-clip typed gradients
with the same procedure (rho_layers / split_half copied from the analyzer, same draw count). The
vision series is recomputed the same way as a check of the port.

usage: same_token5.py {q,mlp}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402
import fig_why_differs as fw  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
fc.OUT = OUT
VIS, HIST, PT, COC, SINK = range(5)  # run_gradient_anatomy.TYPES order
LAST = 35
EARLY, LATE = slice(0, 22), slice(22, LAST)
AXIS = sys.argv[1] if len(sys.argv) > 1 else "q"
N_SPLIT = {"q": 50, "mlp": 10}[AXIS]  # analyzer defaults: --n-split 50, --n-split-mlp 10
PERCLIP = {"q": ("anatomy_perclip_q.npz", "fm_full", "ce"), "mlp": ("anatomy_perclip_mlp.npz", "fm_type", "ce_type")}[AXIS]
FNAME = {"q": "fig4_same_token5_q_head", "mlp": "fig4_same_token5_mlp"}[AXIS]


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
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)  # (L, U)
        st.append(rho_layers(ta, tb))
        sc_.append(rho_layers(ca, cb))
        cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def per_layer_corrected(st, sc_, cr):
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


def band(st, sc_, cr, sl):
    y = per_layer_corrected(st[sl], sc_[sl], cr[sl])
    return float(np.nanmean(y)), int(np.isfinite(y).sum()), float(np.nanmean(st[sl])), float(np.nanmean(sc_[sl]))


m = json.loads((REPO / "outputs" / "gradanat_v1" / "metrics_analysis.json").read_text())["P2"][AXIS]
stored = {"pooled": "pooled (the shipped score)", "vision tokens": "both at VISION positions",
          "prompt-text tokens": "both at prompt-text positions only", "CoC tokens": "both at CoC positions only",
          "ego-history tokens": None}
series = {}
with np.load(REPO / "outputs" / "gradanat_v1" / PERCLIP[0]) as pq:
    fm, ce = pq[PERCLIP[1]], pq[PERCLIP[2]]  # (100, 5, 36, U) signed per-clip typed gradients
    rng = np.random.default_rng(0)
    chk = split_half(np.abs(fm[:, VIS])[:, :LAST], np.abs(ce[:, VIS])[:, :LAST], N_SPLIT, rng)
    for label, key in stored.items():
        if key in m:
            series[label] = tuple(np.array(m[key]["per_layer"][k]) for k in ("self_traj", "self_coc", "cross"))
        else:
            ty = {"prompt-text tokens": PT, "ego-history tokens": HIST}[label]
            series[label] = split_half(np.abs(fm[:, ty])[:, :LAST], np.abs(ce[:, ty])[:, :LAST], N_SPLIT, rng)
            print(f"computed here: {label} ({N_SPLIT} splits)")

# port check: the recomputed vision series against the stored one (fresh rng, so Monte-Carlo noise only)
ref, new = per_layer_corrected(*series["vision tokens"]), per_layer_corrected(*chk)
print(f"port check (vision, {N_SPLIT} fresh splits vs stored): max |diff| per layer {np.nanmax(np.abs(ref - new)):.3f}, "
      f"late mean stored {np.nanmean(ref[LATE]):.3f} vs recomputed {np.nanmean(new[LATE]):.3f}")

styles = (("black", "-", 1.6), (fc.TRAJ, "-", 1.1), (fw.PURPLE, (0, (4, 2)), 1.1),
          (fc.COC, (0, (1, 1)), 1.3), (fw.GREEN, (0, (3, 1, 1, 1)), 1.1))
L = np.arange(fw.N_L)
fig, ax = fw.depth_axes(r"Corrected rank agreement ($\rho$)", (-0.3, 1.1))
print(f"{'series':26s} early corr (layers)  late corr (layers)   late ceilings traj/coc")
for (label, (st, sc_, cr)), (col, ls, lw) in zip(series.items(), styles):
    y = per_layer_corrected(st, sc_, cr)
    ax.plot(L[:LAST], y, color=col, lw=lw, ls=ls, label=label)
    e, le = band(st, sc_, cr, EARLY), band(st, sc_, cr, LATE)
    print(f"{label:26s} {e[0]:5.3f} ({e[1]:2d}/22)          {le[0]:5.3f} ({le[1]:2d}/13)           {le[2]:.2f}/{le[3]:.2f}")
ax.axhline(0, color="black", lw=0.7)
fw.legend_above(ax, 3)
fc.save(fig, FNAME)
print(f"-> {OUT / (FNAME + '.png')}")
