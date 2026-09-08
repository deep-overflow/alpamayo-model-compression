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

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
# figures/ is tracked, unlike outputs/ -- these are paper artifacts, not run output, and
# they have to survive with the manuscript rather than with the experiment directories
OUT = REPO / "figures"
IMP = REPO / "outputs" / "importance_v2" / "importance.npz"
STEP = REPO / "outputs" / "stepimp_fm_perstep_v2" / "step_importance.npz"

# shipped dual_u40_v2: uniform 0.3985632694 -> 19/32 Q heads, 7390/12288 MLP channels
KEEP_Q, N_Q = 19, 32
KEEP_M, N_M = 7390, 12288
TRAJ, COC, GREY = "#2F6FBF", "#D97757", "#8C8878"
SINGLE = (3.4, 2.4)   # one column
WIDE = (4.6, 2.6)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def rho_per_layer(a, b):
    return np.array([stats.spearmanr(a[l], b[l]).statistic
                     if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan
                     for l in range(a.shape[0])])


def kept_overlap(a, b, keep):
    out = []
    for l in range(a.shape[0]):
        ka = set(np.argsort(-a[l])[:keep])
        kb = set(np.argsort(-b[l])[:keep])
        out.append(len(ka & kb) / keep)
    return np.array(out)


def depth_panel(traj, coc, axis_name, fname):
    """Per-layer importance of both objectives, each normalised to its own maximum."""
    t, c = traj.sum(1), coc.sum(1)
    tn, cn = t / t.max(), c / c.max()
    L = np.arange(len(t))
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(L, tn, color=TRAJ, lw=1.5, label=r"$I_{\mathrm{traj}}$  (trajectory FM)")
    ax.plot(L, cn, color=COC, lw=1.5, label=r"$I_{\mathrm{CoC}}$  (reasoning NLL)")
    for v, col in ((int(t.argmax()), TRAJ), (int(c.argmax()), COC)):
        ax.axvline(v, color=col, lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("layer importance (max-normalised)")
    ax.set_title(axis_name, loc="left")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    ax.annotate(f"peak {int(t.argmax())}", (t.argmax(), 1.03), color=TRAJ, fontsize=7,
                ha="right", xytext=(-2, 0), textcoords="offset points")
    ax.annotate(f"peak {int(c.argmax())}", (c.argmax(), 1.03), color=COC, fontsize=7,
                ha="left", xytext=(2, 0), textcoords="offset points")
    fig.tight_layout()
    save(fig, fname)
    return {"peak_traj": int(t.argmax()), "peak_coc": int(c.argmax()),
            "com_traj": float((L * tn).sum() / tn.sum()),
            "com_coc": float((L * cn).sum() / cn.sum())}


def figure1(z):
    print("Figure 1 -- objective disagreement")
    q = depth_panel(z["traj_vlm_q"], z["coc_vlm_q"], "Q head", "fig1_depth_q_head")
    m = depth_panel(z["traj_vlm_mlp"], z["coc_vlm_mlp"], "MLP channel", "fig1_depth_mlp")

    rq = rho_per_layer(z["traj_vlm_q"], z["coc_vlm_q"])
    rm = rho_per_layer(z["traj_vlm_mlp"], z["coc_vlm_mlp"])
    L = np.arange(len(rq))
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, rq, color=TRAJ, lw=1.4, label="Q head")
    ax.plot(L, rm, color=COC, lw=1.4, label="MLP channel")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel(r"Spearman $\rho$ between the two")
    ax.set_xlim(-0.5, 35.5)
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
    ax.text(0.5, KEEP_Q / N_Q - 0.035, "chance", color=GREY, fontsize=7)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("shared fraction of the kept set")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0.45, 1.0)
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
        fig, ax = plt.subplots(figsize=(2.9, 2.5))
        im = ax.imshow(M, vmin=0.3, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(0, n_step, 3)); ax.set_yticks(range(0, n_step, 3))
        ax.set_xlabel("denoising step"); ax.set_ylabel("denoising step")
        ax.set_title(f"{nm}   mean $\\rho$ = {M[off].mean():.2f}", loc="left")
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
    ax.plot(np.arange(1, len(cq) + 1) / len(cq), cq, color=TRAJ, lw=1.5, label="Q head")
    ax.plot(np.arange(1, len(cm) + 1) / len(cm), cm, color=COC, lw=1.5, label="MLP channel")
    ax.axvline(0.9, color="black", lw=0.7, ls=":")
    q90, m90 = cq[int(0.9 * len(cq)) - 1], cm[int(0.9 * len(cm)) - 1]
    ax.annotate(f"{100 * q90:.0f}%", (0.9, q90), textcoords="offset points",
                xytext=(-30, 2), color=TRAJ, fontsize=7.5)
    ax.annotate(f"{100 * m90:.1f}%", (0.9, m90), textcoords="offset points",
                xytext=(-32, 6), color=COC, fontsize=7.5)
    ax.set_xlabel("fraction of units removed (lowest score first)")
    ax.set_ylabel("importance mass removed")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_mass_curve")
    print(f"    bottom 90% carries  Q {100 * q90:.1f}%  MLP {100 * m90:.1f}%")

    fig, ax = plt.subplots(figsize=SINGLE)
    for a, col, nm in ((q, TRAJ, "Q head"), (m, COC, "MLP channel")):
        xs, ys = per_layer(a)
        r = stats.spearmanr(xs, ys)
        ax.scatter(xs, ys, s=10, color=col, alpha=0.75, edgecolors="none",
                   label=f"{nm}  $\\rho$={r.statistic:+.2f}")
        print(f"    within-axis {nm:11s} rho {r.statistic:+.3f}  p={r.pvalue:.3f}")
    ax.set_xlabel(r"step-to-step rank agreement $\rho$ (per layer)")
    ax.set_ylabel("mass in the bottom 90%")
    ax.set_yscale("log")
    ax.legend(loc="center left")
    fig.tight_layout()
    save(fig, "fig2_stability_vs_mass")


def main():
    figure1(np.load(IMP))
    figure2(np.load(STEP))


if __name__ == "__main__":
    main()
