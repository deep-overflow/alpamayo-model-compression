"""Two presentations of the Q-head token shares that separate structure from clip-sampling noise:
  _ci    per-layer lines with a 95% bootstrap band over the 100 calibration clips
  _bands faint per-layer lines with bold band means (0-5 / 6-17 / 18-21 / 22-26 / 27-34),
         the bands whose values the text quotes
Same two-row layout and colours as fig4_token_handoff5_q_head."""
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
VIS, HIST, PT, COC, SINK = range(5)
TYPES = ((VIS, "vision tokens", fc.TRAJ, "-"), (PT, "prompt-text tokens", fw.PURPLE, (0, (4, 2))),
         (COC, "CoC tokens", fc.COC, (0, (1, 1))), (HIST, "ego-history tokens", fw.GREEN, (0, (3, 1, 1, 1))))
BANDS = ((0, 6), (6, 18), (18, 22), (22, 27), (27, 35))
L = np.arange(fw.N_L)
B = 500


def shares(G):
    """G (N, 5, L, U) signed -> additive share (5, L)."""
    full = G.sum(1)
    num = (G * np.sign(full)[:, None]).sum((0, 3))
    den = np.abs(full).sum((0, 2))
    return num / np.where(den > 0, den, np.nan)


with np.load(REPO / "outputs" / "gradanat_v1" / "anatomy_perclip_q.npz") as z:
    data = {r"$I_{\mathrm{traj}}$": z["fm_full"].astype(np.float64), r"$I_{\mathrm{CoC}}$": z["ce"].astype(np.float64)}
m = json.loads((REPO / "outputs" / "gradanat_v1" / "metrics_analysis.json").read_text())["additive_by_type"]
ref = {r"$I_{\mathrm{traj}}$": np.array(m["q_fm"]["share_by_type_and_layer"]), r"$I_{\mathrm{CoC}}$": np.array(m["q_ce"]["share_by_type_and_layer"])}

rng = np.random.default_rng(0)
est = {}
for loss, G in data.items():
    point = shares(G)
    last = fw.LAST if "traj" in loss else fw.N_L
    print(f"{loss}: point estimate vs stored shares, max |diff| over layers<{last}: {np.nanmax(np.abs(point[:, :last] - ref[loss][:, :last])):.2e}")
    boot = np.stack([shares(G[rng.integers(0, len(G), len(G))]) for _ in range(B)])
    est[loss] = (point, np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0), last)


def layout():
    fig, axes = plt.subplots(2, 1, figsize=(fc.SINGLE[0], fc.SINGLE[1] * 1.85), sharex=True)
    for ax, loss in zip(axes, est):
        ax.axvspan(21.5, 35.5, color=fc.GREY, alpha=0.13, lw=0)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlim(-0.5, 35.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"Share of {loss}")
        ax.text(0.02, 0.95, loss, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    axes[1].set_xlabel("VLM layer")
    axes[1].set_xticks([0, 7, 14, 21, 28, 35])
    fig.subplots_adjust(hspace=0.12)
    return fig, axes


# (A) per-layer lines + 95% bootstrap band
fig, axes = layout()
for ax, (loss, (point, lo, hi, last)) in zip(axes, est.items()):
    for ti, label, col, ls in TYPES:
        ax.fill_between(L[:last], lo[ti][:last], hi[ti][:last], color=col, alpha=0.18, lw=0)
        ax.plot(L[:last], point[ti][:last], color=col, lw=1.2, ls=ls, label=label)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6,
               handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)
fc.save(fig, "fig4_token_handoff5_q_head_ci")

# (B) faint per-layer lines + bold band means
fig, axes = layout()
for ax, (loss, (point, lo, hi, last)) in zip(axes, est.items()):
    print(f"== {loss} band means " + " / ".join(f"L{a}-{b - 1}" for a, b in BANDS))
    for ti, label, col, ls in TYPES:
        ax.plot(L[:last], point[ti][:last], color=col, lw=0.7, ls="-", alpha=0.3)
        means = []
        for a, b in BANDS:
            b = min(b, last)
            mu = np.nanmean(point[ti][a:b])
            means.append(mu)
            # plot() rather than hlines(): fig_criterion.save inspects legend handles as Line2D
            ax.plot([a - 0.5, b - 0.5], [mu, mu], color=col, lw=2.0, ls=ls, solid_capstyle="butt",
                    label=label if (a, b) == BANDS[0] else None)
        print(f"  {label:20s} " + " / ".join(f"{v:.2f}" for v in means))
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6,
               handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)
fc.save(fig, "fig4_token_handoff5_q_head_bands")
print(f"-> {OUT}")
