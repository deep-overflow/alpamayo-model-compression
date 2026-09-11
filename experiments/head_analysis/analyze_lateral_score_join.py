"""Does the lane-excursion delta travel with the closed-loop score delta, scene by scene?

`analyze_lateral.py` shows that `dual+h4` spends +4.4pp of its time out of lane and that
its longest excursion grows 4.87 s -> 5.64 s. That is two aggregates moving in the same
run; it is not yet evidence that the lane failures are what the -0.091 score gap is made
of. This joins them per scene.

**The score is not independent of the lane.** `score_criteria` reads

    collision_at_fault == 0 ,  offroad == 0 ,
    progress_score = min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)

so `offroad` is a hard gate inside the score. That is not circular with what is measured
here -- time at zero lane margin is 41% of steps while `offroad` fires on only 3.46% of
them -- but it is a direct channel, and an unconditional correlation cannot tell the two
apart. So the question is split:

  (a) overall     rho(delta excursion time, delta score) over the 150 paired scenes
  (b) gate-free   the same, on scenes where NEITHER arm tripped offroad -- here the only
                  route left to the score is progress, so a surviving association means
                  lane behaviour costs score without the gate
  (c) specificity does delta score track the lateral delta more than a longitudinal delta
                  that did NOT move the score story (braking rate)?
  (d) control     `dualexp_em93p75` moved the score (-0.035) but nothing lateral. If the
                  association is a generic artefact of two noisy runs it should show up
                  there too; if it is lateral-specific it should not.

Spearman throughout: both deltas are heavy-tailed and half the scenes tie at zero.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_lateral_score_join.py \
      --lateral .../outputs/headmlp_lateral --out .../outputs/headmlp_lateral_join
"""

import argparse
import collections
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
PREFIX = "m2601_merged_"
REF = "slim_dual_u40_v2"
ARMS = ["slim_dual_u40_qcut4_v2", "slim_dualexp_u40_em93p75"]
BOOT = 20000

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ORANGE, BLUE, GREEN, RED = "#D97757", "#2a78d6", "#008300", "#b3261e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scene_scores(cfg):
    f = RUNS / f"{PREFIX}{cfg}" / "aggregate" / "results-summary.json"
    per = collections.defaultdict(list)
    for r in json.loads(f.read_text())["rollouts"]:
        per[r["clipgt_id"]].append(float(r["score"]))
    return {k: float(np.mean(v)) for k, v in per.items()}


def scene_offroad(cfg):
    f = RUNS / f"{PREFIX}{cfg}" / "aggregate" / "results-summary.json"
    per = collections.defaultdict(int)
    for r in json.loads(f.read_text())["rollouts"]:
        sm = r.get("score_metrics") or {}
        if sm.get("offroad") is not None:
            per[r["clipgt_id"]] += float(sm["offroad"]) > 0
    return per


def spearman_ci(x, y, seed=0):
    r = stats.spearmanr(x, y)
    g = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    idx = g.integers(0, len(x), (BOOT, len(x)))
    b = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx[:2000]])
    b = b[np.isfinite(b)]
    return (float(r.statistic), float(r.pvalue),
            float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lateral", type=Path, required=True,
                    help="analyze_lateral.py --out dir (needs per_scene.json)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lat = json.loads((args.lateral / "per_scene.json").read_text())
    sc = {c: scene_scores(c) for c in [REF] + ARMS}
    off = {c: scene_offroad(c) for c in [REF] + ARMS}
    scenes = sorted(set(lat[REF]) & set(sc[REF]))

    res = {"n_scenes": len(scenes), "reference": REF,
           "score_criteria": "collision_at_fault==0, offroad==0, progress_score", "arms": {}}
    L = [f"lane excursion vs score, per scene (n={len(scenes)}, reference {REF})",
         "score contains `offroad == 0` as a hard gate -- (b) removes that channel", ""]

    for arm in ARMS:
        d_score = np.array([sc[arm][s] - sc[REF][s] for s in scenes])
        d_exc = np.array([lat[arm][s]["frac_out_of_lane"] - lat[REF][s]["frac_out_of_lane"]
                          for s in scenes])
        d_long = np.array([lat[arm][s]["plan_dev_median"] - lat[REF][s]["plan_dev_median"]
                           for s in scenes])
        gate_free = np.array([(off[arm].get(s, 0) == 0) and (off[REF].get(s, 0) == 0)
                              for s in scenes])

        cell = {}
        r, p, lo, hi = spearman_ci(d_exc, d_score)
        cell["overall"] = {"rho": r, "p": p, "ci": [lo, hi], "n": len(scenes)}
        r2, p2, lo2, hi2 = spearman_ci(d_exc[gate_free], d_score[gate_free])
        cell["gate_free"] = {"rho": r2, "p": p2, "ci": [lo2, hi2], "n": int(gate_free.sum())}
        r3, p3, lo3, hi3 = spearman_ci(d_long, d_score)
        cell["plan_dev_control"] = {"rho": r3, "p": p3, "ci": [lo3, hi3], "n": len(scenes)}

        # how much of the aggregate score gap sits in the scenes whose excursion time rose
        up = d_exc > 0
        cell["decomposition"] = {
            "scenes_excursion_up": int(up.sum()),
            "mean_dscore_where_up": float(d_score[up].mean()) if up.any() else float("nan"),
            "mean_dscore_where_not_up": float(d_score[~up].mean()) if (~up).any() else float("nan"),
            "share_of_total_gap": (float(d_score[up].sum() / d_score.sum())
                                   if d_score.sum() else float("nan")),
            "mannwhitney_p": (float(stats.mannwhitneyu(d_score[up], d_score[~up]).pvalue)
                              if up.any() and (~up).any() else float("nan")),
        }
        res["arms"][arm] = cell

        L.append(f"[{arm}]  mean dscore {d_score.mean():+.4f}")
        for k, lab in (("overall", "(a) all scenes"), ("gate_free", "(b) no-offroad scenes"),
                       ("plan_dev_control", "(c) plan-deviation control")):
            e = cell[k]
            L.append(f"  {lab:28s} rho {e['rho']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}] "
                     f"p={e['p']:.4g}  n={e['n']}")
        d = cell["decomposition"]
        L.append(f"  excursion-time rose on {d['scenes_excursion_up']}/{len(scenes)} scenes; "
                 f"mean dscore there {d['mean_dscore_where_up']:+.4f} vs "
                 f"{d['mean_dscore_where_not_up']:+.4f} elsewhere "
                 f"(Mann-Whitney p={d['mannwhitney_p']:.4g})")
        L.append(f"  those scenes carry {100 * d['share_of_total_gap']:.0f}% of the total gap")
        L.append("")

    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))
    (args.out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    plots = args.out / "plots"
    plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for j, arm in enumerate(ARMS):
        d_score = np.array([sc[arm][s] - sc[REF][s] for s in scenes])
        d_exc = np.array([lat[arm][s]["frac_out_of_lane"] - lat[REF][s]["frac_out_of_lane"]
                          for s in scenes])
        gf = np.array([(off[arm].get(s, 0) == 0) and (off[REF].get(s, 0) == 0)
                       for s in scenes])
        ax[j].scatter(d_exc[gf], d_score[gf], s=16, color=BLUE, alpha=0.6,
                      label="no offroad either arm")
        ax[j].scatter(d_exc[~gf], d_score[~gf], s=26, color=RED, alpha=0.8,
                      label="offroad in one arm")
        ax[j].axhline(0, color=MUTED, lw=0.8)
        ax[j].axvline(0, color=MUTED, lw=0.8)
        e = res["arms"][arm]["overall"]
        ax[j].set_title(f"{arm.replace('slim_', '')}\nrho {e['rho']:+.3f} "
                        f"[{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]", fontsize=9)
        ax[j].set_xlabel("Δ time out of lane vs dual")
        if j == 0:
            ax[j].set_ylabel("Δ closed-loop score vs dual")
            ax[j].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(plots / "join.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
