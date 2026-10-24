"""fig4_token_handoff_{mlp,q_head} extended to every token type: the additive share of each
layer's shipped score earned at vision / prompt-text / generated-CoC / ego-history positions
(sink omitted, < 1%). One figure per axis, two rows: I_traj on top, I_CoC below, same colours
as fig4_ratio_by_token5_* and fig4_same_token5_*. Data: outputs/gradanat_v1/metrics_analysis.json
(additive_by_type, share_by_type_and_layer), exactly what the two-line panels read."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402
import fig_why_differs as fw  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
fc.OUT = OUT
VIS, HIST, PT, COC, SINK = range(5)  # run_gradient_anatomy.TYPES order
TYPES = ((VIS, "vision tokens", fc.TRAJ, "-"), (PT, "prompt-text tokens", fw.PURPLE, (0, (4, 2))),
         (COC, "CoC tokens", fc.COC, (0, (1, 1))), (HIST, "ego-history tokens", fw.GREEN, (0, (3, 1, 1, 1))))
L = np.arange(fw.N_L)

m = json.loads((REPO / "outputs" / "gradanat_v1" / "metrics_analysis.json").read_text())["additive_by_type"]
for name, fname in (("mlp", "fig4_token_handoff5_mlp"), ("q", "fig4_token_handoff5_q_head")):
    fm = np.array(m[f"{name}_fm"]["share_by_type_and_layer"])  # (5, 36); layer 35 is all-zero (I_traj = 0)
    ce = np.array(m[f"{name}_ce"]["share_by_type_and_layer"])  # (5, 36)
    fig, axes = plt.subplots(2, 1, figsize=(fc.SINGLE[0], fc.SINGLE[1] * 1.85), sharex=True)
    print(f"== {name}   share L6-17 -> L27-34")
    for ax, (share, last, title) in zip(axes, ((fm, fw.LAST, r"$I_{\mathrm{traj}}$"), (ce, fw.N_L, r"$I_{\mathrm{CoC}}$"))):
        ax.axvspan(21.5, 35.5, color=fc.GREY, alpha=0.13, lw=0)
        for ti, label, col, ls in TYPES:
            ax.plot(L[:last], share[ti][:last], color=col, lw=1.2 if ti != VIS else 1.4, ls=ls, label=label)
            print(f"  {title:18s} {label:20s} {share[ti][6:18].mean():.3f} -> {share[ti][27:35].mean():.3f}")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlim(-0.5, 35.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"Share of {title}")
        ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    axes[1].set_xlabel("VLM layer")
    axes[1].set_xticks([0, 7, 14, 21, 28, 35])
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6,
                   handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)
    fig.subplots_adjust(hspace=0.12)
    fc.save(fig, fname)
    print(f"  -> {OUT / (fname + '.png')}")
