"""Review figure (2026-11-08): the A2 'late band' of the rank-agreement comparison, relative depth vs step-aligned.

(a, b) paper Fig. 3(a) (ceiling-corrected within-layer rank agreement, Q heads) for Alpamayo 1.5 and Alpamayo 2 Super, with
       the relative-depth late band (hatched; A2 39-62 = A1.5's 22-34 scaled by 64/36) and the post-step band (grey,
       tau..last-1, tau = fitted I_traj/I_CoC step: 23 / 51)
(c)    pooled agreement on the shipped scores (Finding 1, Spearman of I_traj vs I_CoC per layer, Q heads solid, MLP dashed)
       against the layer offset from each model's step

Reads outputs/iclr27-rebuttal/{a2_fig3_token_type,a2_depth_structure}/metrics.json.
Usage: .venv/bin/python iclr27-rebuttal/code/a2_rank_band_review.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "iclr27-rebuttal" / "a2_rank_band_review"
COLORS = {"All": "black", "Vision": "#618BC8", "Prompt text": "#ECAE3C", "CoC": "#8E6BB3", "Ego history": "#C0504D"}
MODELS = {"a15": {"name": "Alpamayo 1.5", "n": 36, "rel_late": (22, 34), "tau": {"q": 23, "mlp": 23}, "color": "#1f4e9c"},
          "a2": {"name": "Alpamayo 2 Super", "n": 64, "rel_late": (39, 62), "tau": {"q": 51, "mlp": 50}, "color": "#c2410c"}}


def main():
    fig3 = json.loads((REPO / "outputs/iclr27-rebuttal/a2_fig3_token_type/metrics.json").read_text())["per_layer"]
    depth = json.loads((REPO / "outputs/iclr27-rebuttal/a2_depth_structure/metrics.json").read_text())["per_layer"]
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1, 1.5, 1.15]})
    for ax, key in zip(axes[:2], MODELS, strict=True):
        m, ag = MODELS[key], fig3[f"{key}/q"]["agreement"]
        last, tau = m["n"] - 1, m["tau"]["q"]
        lo, hi = m["rel_late"]
        ax.axvspan(lo - 0.5, hi + 0.5, facecolor="none", edgecolor="0.55", hatch="///", lw=0, zorder=0)
        ax.axvspan(tau - 0.5, last - 0.5, color="0.85", lw=0, zorder=0)
        ax.axvline(tau - 0.5, color="black", lw=0.8, ls=":")
        for lab, col in COLORS.items():
            y = np.array(ag[lab]["corrected"], float)
            ax.plot(np.arange(len(y)), y, color=col, lw=2.0 if lab == "All" else 1.2, label=lab)
        post = np.nanmean(np.array(ag["All"]["corrected"], float)[tau:last])
        rel = np.nanmean(np.array(ag["All"]["corrected"], float)[lo:hi + 1])
        ax.text(0.02, 0.05, f"All, mean over\n  hatched (rel. depth {lo}-{hi}): {rel:.2f}\n  grey (post-step {tau}-{last - 1}): {post:.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-0.25, 1.08)
        ax.set_xlim(-0.5, last - 0.5)
        ax.set_xlabel("VLM layer index")
        ax.set_title(f"({'a' if key == 'a15' else 'b'}) {m['name']}: corrected ρ, Q heads", fontsize=10, loc="left")
    axes[0].set_ylabel("Corrected ρ (I_FM vs I_CE)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 0.3), fontsize=8, frameon=False, ncol=1)
    ax = axes[2]
    for key, m in MODELS.items():
        for axis, ls in (("q", "-"), ("mlp", "--")):
            tau = m["tau"][axis]
            r = np.array([np.nan if v is None else v for v in depth[key][f"rho_{axis}"]], float)[: m["n"] - 1]
            ax.plot(np.arange(len(r)) - tau, r, ls, color=m["color"], lw=1.6,
                    label=f"{m['name']} {'Q' if axis == 'q' else 'MLP'} (post {np.nanmean(r[tau:]):.2f})")
    ax.axvline(-0.5, color="black", lw=0.8, ls=":")
    ax.set_xlabel("layer − step layer τ")
    ax.set_ylabel("Spearman ρ, shipped scores")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Finding 1 aligned at the step", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.text(0.5, 0.005, "hatched = late band by relative depth (used in the 09-27 report and the first 11-08 report);  "
             "grey = after the fitted step τ;  dotted = τ", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "rank_band_review.png", dpi=180)
    fig.savefig(OUT / "rank_band_review.pdf")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
