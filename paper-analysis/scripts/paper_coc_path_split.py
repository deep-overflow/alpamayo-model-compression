"""Paper-style fig8c: within-layer rank agreement of I_traj at generated-CoC positions with the
full I_CoC at those positions and with its two parts (own-token path: the position's own logits;
cross-position path: the position's K/V read by later CoC tokens). Split-half ceiling-corrected
(as fig4_same_position_q_head) by default; `raw` as the second argument plots the raw Spearman on
the 100-clip means. Data: outputs/coc_pathsplit_v1_s*/pathsplit_perclip.npz + gradanat_v1.
Usage: python paper_coc_path_split.py [suffix=""] [corrected|raw]"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from paper_style import BLUE, BRICK, GREY, REPO, SINGLE, TEXT, export

OUT = Path(__file__).resolve().parent / "paperfigs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "corrected"
SHARDS = [f"coc_pathsplit_v1_s{i}" for i in range(4)]
COC, LAST, N_SPLIT = 3, 35, 50
rng = np.random.default_rng(0)


def rho_layers(a, b):
    return np.array([spearmanr(a[l], b[l])[0] if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan for l in range(a.shape[0])])


def corrected(t, c):
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        st.append(rho_layers(t[a].mean(0), t[b].mean(0)))
        sc_.append(rho_layers(c[a].mean(0), c[b].mean(0)))
        cr.append(0.5 * (rho_layers(t[a].mean(0), c[b].mean(0)) + rho_layers(t[b].mean(0), c[a].mean(0))))
    st, sc_, cr = (np.nanmean(v, 0) for v in (st, sc_, cr))
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


clips, qf, qd = [], [], []
for s in SHARDS:
    clips += json.loads((REPO / "outputs" / s / "metrics.json").read_text())["clip_ids"]
    z = np.load(REPO / "outputs" / s / "pathsplit_perclip.npz")
    qf.append(z["q_full"][:, COC])
    qd.append(z["q_direct"][:, COC])
qf, qd = np.concatenate(qf).astype(np.float64), np.concatenate(qd).astype(np.float64)  # (N, L, H)
anat_ids = [r["clip_id"] for r in json.loads((REPO / "outputs" / "gradanat_v1" / "metrics.json").read_text())["per_clip"]]
order = [anat_ids.index(c) for c in clips]
T = np.abs(np.load(REPO / "outputs" / "gradanat_v1" / "anatomy_perclip_q.npz")["fm_full"][order][:, COC].astype(np.float64))
series = {"Full": np.abs(qf), "Cross-position": np.abs(qf - qd), "Own token": np.abs(qd)}
L = np.arange(LAST)
fig, ax = plt.subplots(figsize=SINGLE)
ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
for (name, X), col, lw in zip(series.items(), ("black", BLUE, BRICK), (1.8, 1.3, 1.3)):
    y = corrected(T[:, :LAST], X[:, :LAST]) if MODE == "corrected" else rho_layers(T.mean(0), X.mean(0))[:LAST]
    ax.plot(L, y, color=col, lw=lw, label=name)
    print(f"  {name:16s} {MODE}: 0-21 {np.nanmean(y[:22]):+.2f}, 22-34 {np.nanmean(y[22:]):+.2f}")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("VLM layer index")
ax.set_ylabel(r"Corrected $\rho$" if MODE == "corrected" else r"Spearman’s $\rho$")
ax.set_xlim(-0.5, 35.5)
ax.set_xticks([0, 7, 14, 21, 28, 35])
ax.set_ylim(-0.25, 1.05)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=TEXT, handlelength=1.0,
          handletextpad=0.4, columnspacing=0.9, borderaxespad=0.2)
export(fig, OUT, "fig8c_coc_path_split" + ("" if MODE == "corrected" else "_raw") + SUFFIX)
