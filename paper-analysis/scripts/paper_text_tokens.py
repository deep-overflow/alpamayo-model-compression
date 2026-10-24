"""Paper-style fig13: which text tokens carry I_traj and I_CoC. Left: share of each score's Q-head
contribution over the non-vision positions by token role (layers 0-21 and 22-34 side by side per
role; token share as a grey tick). Right: share of the ego-history block by timestep (16 steps x 3
tokens; oldest to newest), both scores, both bands. Data: outputs/textpos_v1/metrics.json
(analyze_text_position_anatomy.py, run D). Style as fig_criterion.py --paper-style.
Usage: python paper_text_tokens.py [suffix=""]"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import BLUE, BRICK, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
m = json.loads((REPO / "outputs" / "textpos_v1" / "metrics.json").read_text())
ROLES = (("hist", "Ego-history tokens"), ("vision markers", "⟨vision_end/start⟩"), ("camera text", "Camera labels"),
         ("instruction", "Instruction"), ("system text", "System text"), ("chat markers", "Turn markers"),
         ("cot_start", "⟨cot_start⟩"), ("coc words", "CoC words"), ("cot_end", "⟨cot_end⟩"), ("traj_start", "⟨traj_start⟩"))

fig, axes = plt.subplots(1, 2, figsize=(SINGLE[0] * 2.0, SINGLE[1] * 1.25), gridspec_kw={"width_ratios": [1.35, 1.0]})
ax = axes[0]
y = np.arange(len(ROLES))
for k, (b, alpha) in enumerate((("0-21", 1.0), ("22-34", 0.45))):
    r = m["bands"][b]["roles_nonvis"]
    off = -0.2 if k == 0 else 0.2
    ax.barh(y + off - 0.1, [r[k_]["FM_share"] for k_, _ in ROLES], 0.2, color=BLUE, alpha=alpha, label=(r"$I_{\mathrm{traj}}$ " + b.replace("-", "–")))
    ax.barh(y + off + 0.1, [r[k_]["CE_share"] for k_, _ in ROLES], 0.2, color=YELLOW, alpha=alpha, label=(r"$I_{\mathrm{CoC}}$ " + b.replace("-", "–")))
r = m["bands"]["0-21"]["roles_nonvis"]
ax.scatter([r[k_]["token_share"] for k_, _ in ROLES], y, marker="|", s=80, color="black", zorder=3, label="token share")
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab in ROLES], fontsize=TEXT - 1)
ax.invert_yaxis()
ax.set_xlabel("Share of score (non-vision)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=TEXT - 2, handlelength=1.0,
          handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2, frameon=False)

ax = axes[1]
for b, ls in (("0-21", "-"), ("22-34", "--")):
    p = m["bands"][b]["hist_profile"]
    for L, col in (("FM", BLUE), ("CE", YELLOW)):
        v = np.array(p[L]).reshape(16, 3).sum(1)
        ax.plot(np.arange(16), v / v.sum(), ls, color=col, lw=1.4 if b == "0-21" else 1.0, marker="o", ms=2.5,
                label=(r"$I_{\mathrm{traj}}$" if L == "FM" else r"$I_{\mathrm{CoC}}$") + f" {b.replace('-', '–')}")
ax.axhline(1 / 16, color=BRICK, lw=0.7, ls=":")
ax.set_xlabel("History step (old→new)")
ax.set_ylabel("Share of history")
ax.set_xticks([0, 5, 10, 15])
ax.set_ylim(0, 0.17)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=TEXT - 2, handlelength=1.4,
          handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2, frameon=False)
export(fig, OUT, "fig13_text_tokens" + SUFFIX)
