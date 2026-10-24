"""fig4_ratio_by_token_{mlp,q_head} with two more token types: generated CoC tokens and
ego-history tokens. Same data (outputs/gradanat_v1/anatomy.npz), same helpers and style as
experiments/paper/fig_why_differs.py; writes next to this script, not into figures/."""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402
import fig_why_differs as fw  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
fc.OUT = OUT
T_VIS, T_HIST, T_PT, T_COC = 0, 1, 2, 3  # run_gradient_anatomy.TYPES = (vision, hist, prompt_text, coc, sink)
L = np.arange(fw.N_L)

with np.load(REPO / "outputs" / "gradanat_v1" / "anatomy.npz") as z:
    for name, fname in (("mlp", "fig4_ratio_by_token5_mlp"), ("q", "fig4_ratio_by_token5_q_head")):
        parts = {"pooled": (z[f"fm_full_{name}"], z[f"ce_full_{name}"]),
                 "vision tokens": (z[f"fm_type_{name}"][T_VIS], z[f"ce_{name}"][T_VIS]),
                 "text tokens (prompt + CoC)": (z[f"fm_text_{name}"], z[f"ce_text_{name}"]),
                 "CoC tokens": (z[f"fm_type_{name}"][T_COC], z[f"ce_{name}"][T_COC]),
                 "ego-history tokens": (z[f"fm_type_{name}"][T_HIST], z[f"ce_{name}"][T_HIST])}
        styles = (("black", "-", 1.6), (fc.TRAJ, "-", 1.2), (fw.PURPLE, (0, (4, 2)), 1.2),
                  (fc.COC, (0, (1, 1)), 1.4), (fw.GREEN, (0, (3, 1, 1, 1)), 1.2))
        fig, ax = fw.depth_axes(r"$I_{\mathrm{traj}}\,/\,I_{\mathrm{CoC}}$")
        print(f"== {name}")
        for (label, (t, c)), (col, ls, lw) in zip(parts.items(), styles):
            tl, cl = t.sum(-1)[:fw.LAST], c.sum(-1)[:fw.LAST]  # (35,) layer sums
            y = np.log(tl / cl)
            tau, _, _, r2 = fw.step_fit(y)
            ax.plot(L[:fw.LAST], np.exp(y), color=col, lw=lw, ls=ls, label=label)
            print(f"  {label:28s} L0-21 {np.exp(y[:22].mean()):6.2f}x  L23-34 {np.exp(y[23:].mean()):6.2f}x  "
                  f"step {np.exp(y[:22].mean() - y[23:].mean()):5.2f}x  best step tau {tau:2d} R2 {r2:.2f}")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_ylim(0.6, 90)
        fw.legend_above(ax, 3)
        fc.save(fig, fname)
        print(f"  -> {OUT / (fname + '.png')}")
