"""Paper figures for the criterion and the expert's pruned axis, one panel per file.

Every figure is written on its own so the manuscript can place and size them
independently; nothing here composes a multi-panel sheet.

Figure 1 -- the two objectives do not weight the network the same way.
  fig1_depth_q_head / fig1_depth_mlp
      Per-layer importance of each objective, one panel per axis. Each curve is
      normalised to its own maximum because the two Taylor scores are in different loss
      units; what is comparable is the SHAPE over depth, and the shapes peak in
      different places (Q head 19 vs 24, MLP 18 vs 27). The trajectory objective is
      exactly zero at the last layer -- structurally, not numerically: layer 35's
      o_proj/down_proj never reaches the KV cache the expert reads, so the trajectory
      loss has no path to it.

  fig1_rank_agreement / fig1_kept_overlap  (supplementary)
      The same disagreement read as rank correlation and as shared kept sets. These are
      what make the union criterion quantitative rather than illustrative: after layer
      ~22 a single objective discards a third of the other's picks.

Figure 2 -- why only the expert's MLP is pruned, never its Q heads.
  fig2_steps_q_head / fig2_steps_mlp
      Rank agreement between denoising steps, per axis. Q-head rankings move from step
      to step; MLP rankings barely do.
  fig2_mass_curve
      Cumulative importance mass when units are removed lowest-score-first.
  fig2_stability_vs_mass
      The link between the two, tested WITHIN each axis. Comparing the axes alone would
      confound the mechanism with everything else that differs between a head and a
      channel, so both quantities are measured per layer: layers whose step rankings are
      less stable also spread their mass more, inside the Q-head axis (rho -0.45,
      p=0.006) and inside the MLP axis (-0.44, p=0.008) separately.

Sources: outputs/importance_v2 (the run the shipped dual_u40_v2 was built from; v1 and
importance_v2_ada reproduce every number to within 0.02) and
outputs/stepimp_fm_perstep_v2 (ten per-step expert gradients).

Usage:
  .venv/bin/python experiments/paper/fig_criterion.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
# figures/ is tracked, unlike outputs/ -- these are paper artifacts, not run output, and
# they have to survive with the manuscript rather than with the experiment directories
OUT = REPO / "figures"
IMP = REPO / "outputs" / "importance_v2" / "importance.npz"
STEP = REPO / "outputs" / "stepimp_fm_perstep_v2" / "step_importance.npz"

# shipped dual_u40_v2: uniform 0.3985632694 -> 19/32 Q heads, 7390/12288 MLP channels
KEEP_Q, N_Q = 19, 32
KEEP_M, N_M = 7390, 12288
TRAJ, COC, GREY = "#0072B2", "#D55E00", "#777777"
SINGLE = (3.5, 2.8)  # 89 mm, one manuscript column
WIDE = SINGLE

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "font.style": "normal", "font.weight": "normal",
    # Keep math symbols and log ticks in the same upright family as axis text.
    "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
    "text.usetex": False,
    "axes.titleweight": "semibold", "axes.titlepad": 10,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    "axes.axisbelow": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
})


def save(fig, name):
    """Export fixed-size, embedded-font artwork and a high-resolution preview."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        if ax.get_xlabel() == "VLM layer":
            ax.set_xticks([0, 7, 14, 21, 28, 35])
        if (ax.lines or ax.collections) and not ax.images and ax.get_xlabel():
            ax.grid(axis="y")
        # Encode the second series by dash / marker as well as by colour.
        for line in ax.lines:
            if line.get_color() == COC and line.get_label() in (
                "MLP channel", r"$I_{\mathrm{CoC}}$"
            ):
                line.set_linestyle((0, (4, 2)))
        legend = ax.get_legend()
        if legend is not None:
            for handle in legend.legend_handles:
                if isinstance(handle, matplotlib.lines.Line2D) and handle.get_color() == COC:
                    handle.set_linestyle((0, (4, 2)))
    fig.tight_layout(pad=0.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf / .svg")


def rho_per_layer(a, b):
    return np.array([stats.spearmanr(a[l], b[l]).statistic
                     if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan
                     for l in range(a.shape[0])])


def kept_overlap(a, b, keep):
    out = []
    for l in range(a.shape[0]):
        if np.ptp(a[l]) == 0 or np.ptp(b[l]) == 0:
            out.append(np.nan)
            continue
        ka = set(np.argsort(-a[l])[:keep])
        kb = set(np.argsort(-b[l])[:keep])
        out.append(len(ka & kb) / keep)
    return np.array(out)


def depth_panel(traj, coc, fname):
    """Per-layer importance of both objectives, each normalised to its own maximum."""
    t, c = traj.sum(1), coc.sum(1)
    tn, cn = t / t.max(), c / c.max()
    L = np.arange(len(t))
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(L, tn, color=TRAJ, lw=1.5, label=r"$I_{\mathrm{traj}}$")
    ax.plot(L, cn, color=COC, lw=1.5, label=r"$I_{\mathrm{CoC}}$")
    for v, col in ((int(t.argmax()), TRAJ), (int(c.argmax()), COC)):
        ax.axvline(v, color=col, lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("Normalized importance")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, fname)
    return {"peak_traj": int(t.argmax()), "peak_coc": int(c.argmax()),
            "com_traj": float((L * tn).sum() / tn.sum()),
            "com_coc": float((L * cn).sum() / cn.sum())}


def figure1(z):
    print("Figure 1 -- objective disagreement")
    q = depth_panel(z["traj_vlm_q"], z["coc_vlm_q"], "fig1_depth_q_head")
    m = depth_panel(z["traj_vlm_mlp"], z["coc_vlm_mlp"], "fig1_depth_mlp")

    rq = rho_per_layer(z["traj_vlm_q"], z["coc_vlm_q"])
    rm = rho_per_layer(z["traj_vlm_mlp"], z["coc_vlm_mlp"])
    L = np.arange(len(rq))
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, rq, color=TRAJ, lw=1.4, label="Q head")
    ax.plot(L, rm, color=COC, lw=1.4, label="MLP channel")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel(r"Rank agreement ($\rho$)")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig1_rank_agreement")

    oq = kept_overlap(z["traj_vlm_q"], z["coc_vlm_q"], KEEP_Q)
    om = kept_overlap(z["traj_vlm_mlp"], z["coc_vlm_mlp"], KEEP_M)
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, oq, color=TRAJ, lw=1.4, label="Q head")
    ax.plot(L, om, color=COC, lw=1.4, label="MLP channel")
    ax.axhline(KEEP_Q / N_Q, color=TRAJ, lw=0.9, ls=":")
    ax.axhline(KEEP_M / N_M, color=COC, lw=0.9, ls="--")
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("Retained-set overlap")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0.45, 1.03)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig1_kept_overlap")

    print(f"    Q head : peak traj {q['peak_traj']} vs CoC {q['peak_coc']}  "
          f"(centre of mass {q['com_traj']:.1f} vs {q['com_coc']:.1f})")
    print(f"    MLP    : peak traj {m['peak_traj']} vs CoC {m['peak_coc']}  "
          f"(centre of mass {m['com_traj']:.1f} vs {m['com_coc']:.1f})")
    print(f"    rho Q  {np.nanmean(rq[:22]):+.3f} (layers 0-21) -> {np.nanmean(rq[22:]):+.3f} (22-35)")
    print(f"    rho MLP {np.nanmean(rm[:22]):+.3f} -> {np.nanmean(rm[22:]):+.3f}")
    print(f"    late-layer kept-set overlap  Q {np.nanmean(oq[22:]):.3f} "
          f"(chance {KEEP_Q / N_Q:.3f}), MLP {np.nanmean(om[22:]):.3f} (chance {KEEP_M / N_M:.3f})")


def figure2(s):
    print("Figure 2 -- expert axis")
    q, m = s["q_abs_step"], s["mlp_abs_step"]
    n_step = q.shape[0]

    def step_matrix(a):
        M = np.ones((n_step, n_step))
        for i in range(n_step):
            for j in range(i + 1, n_step):
                M[i, j] = M[j, i] = np.nanmean(
                    [stats.spearmanr(a[i, l], a[j, l]).statistic for l in range(a.shape[1])])
        return M

    def mass_curve(a):
        cur = [np.cumsum(np.sort(a[l])) / max(a[l].sum(), 1e-12) for l in range(a.shape[0])]
        return np.mean(np.stack(cur), axis=0)

    def per_layer(a):
        stab, mass = [], []
        for l in range(a.shape[1]):
            rs = [stats.spearmanr(a[i, l], a[j, l]).statistic
                  for i in range(n_step) for j in range(i + 1, n_step)]
            stab.append(np.nanmean(rs))
            srt = np.sort(a[:, l].sum(0))
            mass.append((np.cumsum(srt) / max(srt.sum(), 1e-12))[int(0.9 * len(srt)) - 1])
        return np.array(stab), np.array(mass)

    off = ~np.eye(n_step, dtype=bool)
    for a, nm, fname in ((q, "Q head", "fig2_steps_q_head"), (m, "MLP channel", "fig2_steps_mlp")):
        M = step_matrix(a)
        fig, ax = plt.subplots(figsize=SINGLE)
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(n_step))
        ax.set_yticks(range(n_step))
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Denoising step")
        for sp in ax.spines.values():
            sp.set_visible(True)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"rank agreement $\rho$", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        fig.tight_layout()
        save(fig, fname)
        print(f"    {nm:11s} step-to-step rho {M[off].mean():.3f} (min {M[off].min():.3f})")

    cq, cm = mass_curve(q.sum(0)), mass_curve(m.sum(0))
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--", label="uniform")
    ax.plot(np.arange(len(cq) + 1) / len(cq), np.r_[0, cq], color=TRAJ, lw=1.5, label="Q head")
    ax.plot(np.arange(len(cm) + 1) / len(cm), np.r_[0, cm], color=COC, lw=1.5, label="MLP channel")
    ax.axvline(0.9, color="black", lw=0.7, ls=":")
    q90, m90 = cq[int(0.9 * len(cq)) - 1], cm[int(0.9 * len(cm)) - 1]
    xq, xm = int(0.9 * len(cq)) / len(cq), int(0.9 * len(cm)) / len(cm)
    ax.scatter([xq], [q90], color=TRAJ, s=18, zorder=4)
    ax.scatter([xm], [m90], color=COC, marker="s", s=18, zorder=4)
    ax.annotate(f"{100 * q90:.0f}%", (xq, q90), textcoords="offset points",
                xytext=(-30, 2), color=TRAJ, fontsize=7.5)
    ax.annotate(f"{100 * m90:.1f}%", (xm, m90), textcoords="offset points",
                xytext=(-32, 6), color=COC, fontsize=7.5)
    ax.set_xlabel("Fraction of units removed")
    ax.set_ylabel("Importance removed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_mass_curve")
    print(f"    bottom 90% carries  Q {100 * q90:.1f}%  MLP {100 * m90:.1f}%")

    fig, ax = plt.subplots(figsize=SINGLE)
    for a, col, nm in ((q, TRAJ, "Q head"), (m, COC, "MLP channel")):
        xs, ys = per_layer(a)
        r = stats.spearmanr(xs, ys)
        ax.scatter(xs, ys, s=18, color=col, alpha=0.8,
                   marker="o" if nm == "Q head" else "^", edgecolors="white", linewidths=0.3,
                   label=nm)
        print(f"    within-axis {nm:11s} rho {r.statistic:+.3f}  p={r.pvalue:.3f}")
    ax.set_xlabel(r"Step rank agreement ($\rho$)")
    ax.set_ylabel("Importance in lowest-ranked 90%")
    ax.set_yscale("log")
    ax.legend(loc="center left")
    fig.tight_layout()
    save(fig, "fig2_stability_vs_mass")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance", type=Path, default=IMP)
    parser.add_argument("--step-importance", type=Path, default=STEP)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_dir
    with np.load(args.importance) as importance, np.load(args.step_importance) as steps:
        figure1(importance)
        figure2(steps)


if __name__ == "__main__":
    main()
