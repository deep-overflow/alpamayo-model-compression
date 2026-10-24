"""Paper-style fig1_rank_agreement with the y-axis cut to [-0.25, 1]: within-layer Spearman
correlation between the I_traj and I_CoC rankings, Q heads and MLP channels. Same data and
computation as experiments/paper/fig_criterion.py (rho_per_layer on outputs/importance_v2).
Usage: python paper_rank_agreement.py [run=importance_v2] [suffix=""]. The run may be an
importance run (importance.npz) or a gradient-anatomy run (anatomy.npz): the anatomy's full-seed
gate gradients summed over positions are the same scores (typed-sum vs shipped score 6e-06)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import BLUE, GREY, REPO, SINGLE, TEXT, YELLOW, export, fc

OUT = Path(__file__).resolve().parent / "paperfigs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "importance_v2"  # run to read
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""            # appended to the file name
run_dir = REPO / "outputs" / RUN
if (run_dir / "importance.npz").exists():
    with np.load(run_dir / "importance.npz") as z:
        rq = fc.rho_per_layer(z["traj_vlm_q"], z["coc_vlm_q"])
        rm = fc.rho_per_layer(z["traj_vlm_mlp"], z["coc_vlm_mlp"])
else:
    with np.load(run_dir / "anatomy.npz") as z:
        rq = fc.rho_per_layer(z["fm_full_q"], z["ce_full_q"])
        rm = fc.rho_per_layer(z["fm_full_mlp"], z["ce_full_mlp"])
L = np.arange(len(rq))
fig, ax = plt.subplots(figsize=SINGLE)
ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
ax.plot(L, rq, color=BLUE, lw=1.4, label="Q head")
ax.plot(L, rm, color=YELLOW, lw=1.4, label="MLP channel")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("VLM layer index")
ax.set_ylabel(r"Spearman’s $\rho$")
ax.set_xlim(-0.5, 35.5)
ax.set_xticks([0, 7, 14, 21, 28, 35])
ax.set_ylim(-0.25, 1.0)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=TEXT, handlelength=1.2,
          handletextpad=0.45, columnspacing=1.2, borderaxespad=0.2)
export(fig, OUT, "fig1_rank_agreement" + SUFFIX)
print(f"  {RUN}: Q mean L0-21 {np.nanmean(rq[:22]):.3f}, L22-34 {np.nanmean(rq[22:35]):.3f} | "
      f"MLP {np.nanmean(rm[:22]):.3f}, {np.nanmean(rm[22:35]):.3f}")
