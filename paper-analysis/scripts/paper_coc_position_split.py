"""Paper-style fig9: the CoC-position exception is a token mixture inside the CoC. Left: share of
each score's |Q-head contribution| at generated-CoC positions on the head clause / rest of the
sentence / <|cot_end|> / <|traj_future_start|> (stacked, per band). Right: raw within-layer rank
agreement of I_traj and I_CoC with both restricted to the same CoC sub-span (all CoC positions =
the shipped CoC row; words only; head clause). Data: outputs/coc_posanat_v1_s*/ (run A of
plans/2026-09-25_coc-position-content.md). Style as fig_criterion.py --paper-style.
Usage: python paper_coc_position_split.py [suffix=""]"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from paper_style import BLUE, BRICK, GREY, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
SHARDS = [f"coc_posanat_v1_s{i}" for i in range(4)]
LAST = 35
BANDS = (("0–21", slice(0, 22)), ("22–34", slice(22, LAST)))

rows, per = [], {}
for s in SHARDS:
    rows += json.loads((REPO / "outputs" / s / "metrics.json").read_text())["per_clip"]
    z = np.load(REPO / "outputs" / s / "posanat_perclip.npz")
    for k in ("q_ce_pos", "q_fm_pos"):
        per.setdefault(k, []).append(z[k])
qce, qfm = (np.nan_to_num(np.concatenate(per[k]).astype(np.float64)) for k in ("q_ce_pos", "q_fm_pos"))  # (N, PAD, L, H)
N = len(rows)
n_coc = np.array([r["n_coc"] for r in rows])
fine = np.full((N, qce.shape[1]), -1)
for i, r in enumerate(rows):
    fine[i, : r["n_coc"]] = r["sub"]  # 0 head clause, 1 rest, 2 <|traj_future_start|>
    if r["n_coc"] >= 2 and fine[i, r["n_coc"] - 2] == 1:
        fine[i, r["n_coc"] - 2] = 3  # <|cot_end|>
SPANS = ((0, "Head clause"), (1, "Rest of sentence"), (3, "⟨cot_end⟩"), (2, "⟨traj_start⟩"))
SPAN_COLORS = (BLUE, "#A9C3E6", YELLOW, BRICK)


def rho_layers(a, b):
    out = np.full(a.shape[0], np.nan)
    for li in range(a.shape[0]):
        if np.ptp(a[li]) > 0 and np.ptp(b[li]) > 0:
            out[li] = spearmanr(a[li], b[li])[0]
    return out


fig, axes = plt.subplots(1, 2, figsize=(SINGLE[0] * 2.0, SINGLE[1]), gridspec_kw={"width_ratios": [1.0, 1.25]})

# left: token weighting inside the CoC
ax = axes[0]
bars = [(r"$I_{\mathrm{CoC}}$", qce, sl, b) for b, sl in BANDS] + [(r"$I_{\mathrm{traj}}$", qfm, sl, b) for b, sl in BANDS]
ylabels = []
for yi, (score, X, sl, b) in enumerate(bars):
    arr = np.abs(X).sum(3).transpose(0, 2, 1)  # (N, L, PAD)
    left = 0.0
    for (fi, lab), col in zip(SPANS, SPAN_COLORS):
        num = np.array([[arr[i, l][fine[i] == fi].sum() for l in range(sl.start, sl.stop)] for i in range(N)])
        den = np.array([[arr[i, l][fine[i] >= 0].sum() for l in range(sl.start, sl.stop)] for i in range(N)])
        sh = float((num / np.maximum(den, 1e-30)).mean())
        ax.barh(yi, sh, left=left, color=col, edgecolor="white", lw=0.5, label=lab if yi == 0 else None)
        left += sh
    ylabels.append(f"{score}\nlayers {b}")
ax.set_yticks(range(len(bars)))
ax.set_yticklabels(ylabels, fontsize=TEXT)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Share of the score at CoC positions")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=TEXT, handlelength=1.0,
          handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2, frameon=False)

# right: same-sub-span agreement
ax = axes[1]
ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
sels = (("All CoC", fine >= 0, "black", 1.8), ("Words only", (fine == 0) | (fine == 1), BLUE, 1.3),
        ("Head clause", fine == 0, YELLOW, 1.3))
for label, m, col, lw in sels:
    msk = m[:, :, None, None]
    T = np.abs((qfm * msk).sum(1)).mean(0)
    C = np.abs((qce * msk).sum(1)).mean(0)
    ax.plot(np.arange(LAST), rho_layers(T, C)[:LAST], color=col, lw=lw, label=label)
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("VLM layer index")
ax.set_ylabel(r"Rank agreement $\rho$")
ax.set_xlim(-0.5, 35.5)
ax.set_xticks([0, 7, 14, 21, 28, 35])
ax.set_ylim(-0.25, 1.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=TEXT, handlelength=1.0,
          handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2, frameon=False)
export(fig, OUT, "fig9_coc_position_split" + SUFFIX)
