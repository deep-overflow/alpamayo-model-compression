"""Paper-style fig11c: the CoC words that carry I_traj (and I_CoC), per occurrence. Words of the
generated CoC merged from subwords; share of the clip's CoC-word total (words-only normalisation,
Q heads of the band); mean share per occurrence for the words of the action clause (top 8, n >= 3)
and of the cause clause (top 10, n >= 3), ranked by I_traj. Data: outputs/coc_tokimp_v1/metrics.json
(analyze_coc_token_importance.py, run A shards). Style as fig_criterion.py --paper-style.
Usage: python paper_coc_words.py [suffix=""]"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import BLUE, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
m = json.loads((REPO / "outputs" / "coc_tokimp_v1" / "metrics.json").read_text())
N_DEC, N_CAUSE = 8, 10

fig, axes = plt.subplots(1, 2, figsize=(SINGLE[0] * 2.0, SINGLE[1] * 1.35), sharex=True)
for ax, b in zip(axes, ("0-21", "22-34")):
    tw = m["bands"][b]["top_words"]
    dec = [w for w in tw["decision clause"] if w["n"] >= 3][:N_DEC]
    cause = [w for w in tw["cause clause"] if w["n"] >= 3][:N_CAUSE]
    items = dec + cause
    y = np.arange(len(items), dtype=float)
    y[len(dec):] += 0.8  # gap between the two clauses
    ax.barh(y - 0.19, [w["FM_per_occ"] for w in items], 0.38, color=BLUE, label=r"$I_{\mathrm{traj}}$")
    ax.barh(y + 0.19, [w["CE_per_occ"] for w in items], 0.38, color=YELLOW, label=r"$I_{\mathrm{CoC}}$")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{w['word']} ({w['n']})" for w in items], fontsize=TEXT - 1)
    ax.invert_yaxis()
    ax.text(0.98, 0.99, f"Layers {b.replace('-', '–')}", transform=ax.transAxes, ha="right", va="top", fontsize=TEXT)
    sep = len(dec) - 0.5 + 0.4
    ax.axhline(sep, color="#777777", lw=0.6, ls=":")
    ax.text(0.98, sep - 0.15, "action clause", ha="right", va="bottom", fontsize=TEXT - 2, color="#555555", transform=ax.get_yaxis_transform())
    ax.text(0.98, sep + 0.15, "cause clause", ha="right", va="top", fontsize=TEXT - 2, color="#555555", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Share per occurrence")
axes[0].legend(loc="lower center", bbox_to_anchor=(1.02, 1.0), ncol=2, fontsize=TEXT, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2, frameon=False)
export(fig, OUT, "fig11c_coc_words" + SUFFIX)
