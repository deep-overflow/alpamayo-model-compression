"""The two figures that justify the criterion and the expert's pruned axis.

Figure 1 -- why the VLM criterion is a UNION of two objectives.
  (a) per-layer Spearman between the trajectory and CoC Taylor importances. They agree
      in early layers and come apart after layer ~22, so neither objective is a proxy
      for the other where it matters.
  (b) how much the two objectives' kept sets actually share at the shipped budget.
      Below chance would mean anti-correlated; near chance means a single-objective
      criterion discards the other objective's top units.

Figure 2 -- why only the expert's MLP is pruned, never its Q heads.
  (a,b) rank agreement between denoising steps, per axis. Q-head rankings move from
      step to step; MLP rankings barely do.
  (c) cumulative importance mass when units are removed lowest-score-first. The MLP's
      mass sits in a few channels, so most of the width is nearly free; the Q head's is
      spread out, so there is no cheap set to remove.
  (d) the link between (a,b) and (c), tested WITHIN each axis. Comparing the two axes
      alone would confound the mechanism with everything else that differs between a
      head and a channel, so the same two quantities are measured per layer: layers whose
      step rankings are less stable also spread their mass more, and that holds inside
      the Q-head axis (rho -0.45, p=0.006) and inside the MLP axis (-0.44, p=0.008)
      separately, not only across them.

The claim Figure 2 makes is therefore mechanistic, not just descriptive: an axis whose
ranking is stable across steps can concentrate its mass, because every step agrees on
which units matter. An axis whose ranking moves cannot -- a head that is idle at one step
carries the load at another, so the union over steps covers nearly every head.

Sources: outputs/importance_v2 (the run the shipped dual_u40_v2 was built from; v1 and
importance_v2_ada reproduce every number here to within 0.02) and
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

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200,
})


def rho_per_layer(a, b):
    return np.array([stats.spearmanr(a[l], b[l]).statistic
                     if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan
                     for l in range(a.shape[0])])


def kept_overlap(a, b, keep):
    """Fraction of one objective's kept set that the other would also keep."""
    out = []
    for l in range(a.shape[0]):
        ka = set(np.argsort(-a[l])[:keep])
        kb = set(np.argsort(-b[l])[:keep])
        out.append(len(ka & kb) / keep)
    return np.array(out)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / name}.png / .pdf")


def figure1(z):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    rq = rho_per_layer(z["traj_vlm_q"], z["coc_vlm_q"])
    rm = rho_per_layer(z["traj_vlm_mlp"], z["coc_vlm_mlp"])
    L = np.arange(len(rq))
    ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, rq, color=TRAJ, lw=1.4, label="Q head")
    ax.plot(L, rm, color=COC, lw=1.4, label="MLP channel")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel(r"Spearman $\rho$   ($I_{\mathrm{traj}}$ vs $I_{\mathrm{CoC}}$)")
    ax.set_title("(a) the two objectives rank alike only early", loc="left")
    ax.set_xlim(-0.5, 35.5)
    ax.legend(loc="lower left")
    ax.text(28.5, 0.93, "late layers", ha="center", color=GREY, fontsize=7.5,
            transform=ax.get_xaxis_transform())

    ax = axes[1]
    oq = kept_overlap(z["traj_vlm_q"], z["coc_vlm_q"], KEEP_Q)
    om = kept_overlap(z["traj_vlm_mlp"], z["coc_vlm_mlp"], KEEP_M)
    ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, oq, color=TRAJ, lw=1.4, label="Q head")
    ax.plot(L, om, color=COC, lw=1.4, label="MLP channel")
    ax.axhline(KEEP_Q / N_Q, color=TRAJ, lw=0.9, ls=":")
    ax.axhline(KEEP_M / N_M, color=COC, lw=0.9, ls="--")
    ax.text(0.5, KEEP_Q / N_Q - 0.035, "chance", color=GREY, fontsize=7)
    ax.set_xlabel("VLM layer")
    ax.set_ylabel("shared fraction of the kept set")
    ax.set_title("(b) a single objective discards the other's picks", loc="left")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig1_dual_objective")
    return {"rho_q_early": float(np.nanmean(rq[:22])), "rho_q_late": float(np.nanmean(rq[22:])),
            "rho_m_early": float(np.nanmean(rm[:22])), "rho_m_late": float(np.nanmean(rm[22:])),
            "ov_q_late": float(np.nanmean(oq[22:])), "ov_m_late": float(np.nanmean(om[22:])),
            "chance_q": KEEP_Q / N_Q, "chance_m": KEEP_M / N_M}


def figure2(s):
    q, m = s["q_abs_step"], s["mlp_abs_step"]          # (10, L, U)
    n_step = q.shape[0]

    def step_matrix(a):
        M = np.ones((n_step, n_step))
        for i in range(n_step):
            for j in range(i + 1, n_step):
                r = np.nanmean([stats.spearmanr(a[i, l], a[j, l]).statistic
                                for l in range(a.shape[1])])
                M[i, j] = M[j, i] = r
        return M

    def mass_curve(a):
        """Fraction of first-order importance mass removed, lowest score first."""
        cur = []
        for l in range(a.shape[0]):
            srt = np.sort(a[l])
            cur.append(np.cumsum(srt) / max(srt.sum(), 1e-12))
        # layers have equal unit counts, so a plain mean over layers is well defined
        return np.mean(np.stack(cur), axis=0)

    def per_layer(a):
        """Per layer: mean step-to-step rank agreement, and bottom-90% mass share."""
        stab, mass = [], []
        for l in range(a.shape[1]):
            rs = [stats.spearmanr(a[i, l], a[j, l]).statistic
                  for i in range(n_step) for j in range(i + 1, n_step)]
            stab.append(np.nanmean(rs))
            srt = np.sort(a[:, l].sum(0))
            c = np.cumsum(srt) / max(srt.sum(), 1e-12)
            mass.append(c[int(0.9 * len(srt)) - 1])
        return np.array(stab), np.array(mass)

    Mq, Mm = step_matrix(q), step_matrix(m)
    off = ~np.eye(n_step, dtype=bool)
    fig, axes2 = plt.subplots(2, 2, figsize=(6.6, 4.6))
    axes = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
    for ax, M, nm in ((axes[0], Mq, "Q head"), (axes[1], Mm, "MLP channel")):
        im = ax.imshow(M, vmin=0.3, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(0, n_step, 3)); ax.set_yticks(range(0, n_step, 3))
        ax.set_xlabel("denoising step"); 
        ax.set_title(f"{nm}   mean $\\rho$ = {M[off].mean():.2f}", loc="left")
        for sp in ax.spines.values():
            sp.set_visible(True)
    axes[0].set_ylabel("denoising step")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label(r"rank agreement $\rho$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    axes[0].text(-0.28, 1.22, "(a)", transform=axes[0].transAxes, fontsize=9)
    axes[1].text(-0.18, 1.22, "(b)", transform=axes[1].transAxes, fontsize=9)

    ax = axes[2]
    cq, cm = mass_curve(q.sum(0)), mass_curve(m.sum(0))
    xq = np.arange(1, len(cq) + 1) / len(cq)
    xm = np.arange(1, len(cm) + 1) / len(cm)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--", label="uniform")
    ax.plot(xq, cq, color=TRAJ, lw=1.5, label="Q head")
    ax.plot(xm, cm, color=COC, lw=1.5, label="MLP channel")
    ax.axvline(0.9, color="black", lw=0.7, ls=":")
    ax.annotate(f"{100 * cq[int(0.9 * len(cq)) - 1]:.0f}%", (0.9, cq[int(0.9 * len(cq)) - 1]),
                textcoords="offset points", xytext=(-30, 2), color=TRAJ, fontsize=7.5)
    ax.annotate(f"{100 * cm[int(0.9 * len(cm)) - 1]:.1f}%", (0.9, cm[int(0.9 * len(cm)) - 1]),
                textcoords="offset points", xytext=(-32, 6), color=COC, fontsize=7.5)
    ax.set_xlabel("fraction of units removed (lowest score first)")
    ax.set_ylabel("importance mass removed")
    ax.set_title("(c) where each axis keeps its mass", loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    ax = axes[3]
    sq, mq_ = per_layer(q)
    sm, mm_ = per_layer(m)
    stats_within = {}
    for (xs, ys, col, nm) in ((sq, mq_, TRAJ, "Q head"), (sm, mm_, COC, "MLP channel")):
        r = stats.spearmanr(xs, ys)
        stats_within[nm] = (float(r.statistic), float(r.pvalue))
        ax.scatter(xs, ys, s=9, color=col, alpha=0.75, edgecolors="none",
                   label=f"{nm}  $\\rho$={r.statistic:+.2f}")
    ax.set_xlabel(r"step-to-step rank agreement $\rho$ (per layer)")
    ax.set_ylabel("mass in the bottom 90%")
    ax.set_title("(d) the link holds inside each axis", loc="left")
    ax.set_yscale("log")
    ax.legend(loc="center left")
    fig.tight_layout()
    save(fig, "fig2_expert_axis")
    return {"step_rho_q": float(Mq[off].mean()), "step_rho_m": float(Mm[off].mean()),
            "step_rho_q_min": float(Mq[off].min()), "step_rho_m_min": float(Mm[off].min()),
            "mass90_q": float(cq[int(0.9 * len(cq)) - 1]),
            "mass90_m": float(cm[int(0.9 * len(cm)) - 1]),
            "within_axis": stats_within}


def main():
    z, s = np.load(IMP), np.load(STEP)
    print("Figure 1 (dual objective)")
    f1 = figure1(z)
    print(f"  rho Q  layers 0-21 {f1['rho_q_early']:+.3f} -> 22-35 {f1['rho_q_late']:+.3f}")
    print(f"  rho MLP layers 0-21 {f1['rho_m_early']:+.3f} -> 22-35 {f1['rho_m_late']:+.3f}")
    print(f"  late-layer kept-set overlap: Q {f1['ov_q_late']:.3f} (chance {f1['chance_q']:.3f}), "
          f"MLP {f1['ov_m_late']:.3f} (chance {f1['chance_m']:.3f})")
    print("Figure 2 (expert axis)")
    f2 = figure2(s)
    print(f"  step-to-step rho: Q {f2['step_rho_q']:.3f} (min {f2['step_rho_q_min']:.3f}), "
          f"MLP {f2['step_rho_m']:.3f} (min {f2['step_rho_m_min']:.3f})")
    print(f"  mass lost removing the bottom 90%: Q {100*f2['mass90_q']:.1f}%, "
          f"MLP {100*f2['mass90_m']:.1f}%")
    for nm, (r, p) in f2["within_axis"].items():
        print(f"  within-axis stability vs concentration: {nm:12s} rho {r:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
