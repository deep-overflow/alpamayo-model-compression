"""Does higher progress come with LOWER distance-to-GT in closed loop? (rollout-free)

Follow-up to the offroad forensics (plans/2026-08-31_offroad-forensics.md), which found
that reconstruction arms lose to `dual` on progress, not on lane keeping, and that dual
has both the highest progress and the WORST d2gt. That pairing is the question here: is
"follow the GT path" aligned with "make progress", or do they trade?

Three levels, all from the stored aggregate summaries (no GPU, no re-simulation):
  A  arm level (5 points): mean progress vs mean d2gt
  B  within arm, scene level: Spearman(progress, d2gt), and the same partialling out
     dist_traveled_m (d2gt is a path-integrated quantity, so a car that drives further
     mechanically accumulates more of it)
  C  paired scene deltas vs a reference arm: does an arm's progress gain travel with a
     d2gt gain?
plus the distance normalisation d2gt / 100 m and dist_traveled / gt_dist_traveled.

Usage (alpasim venv, for the summary parsing only):
  cd /home/cvlab21/project/chan/alpasim && uv run python \
      .../analyze_progress_d2gt.py --runs-root /home/cvlab21/project/chan/alpasim-runs \
      --out .../outputs/alpasim_progress_d2gt
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#E8E6DC",
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
ARMS = {"baseline": "baseline", "slim_dual_u40_v2": "dual", "slim_dualr_u40": "dualr",
        "slim_tyr_u40_r": "tyr_r", "slim_dualr_wl_u40": "wl"}
KEYS = ["progress_clipped_rel", "progress_rel", "dist_to_gt_trajectory", "dist_to_gt_location",
        "dist_traveled_m", "gt_dist_traveled_m", "score", "offroad", "collision_at_fault"]


def scene_means(rollouts, key):
    out = {}
    for r in rollouts:
        v = r["metrics"].get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        out.setdefault(r["clipgt_id"], []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def partial_spearman(x, y, z):
    """Spearman of x,y with the rank-linear effect of z removed from both."""
    X, Y, Z = (rankdata(np.asarray(v, float)) for v in (x, y, z))

    def resid(a):
        return a - np.polyval(np.polyfit(Z, a, 1), Z)

    return spearmanr(resid(X), resid(Y))


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), *[float(q) for q in np.percentile(b, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--prefix", default="m2601_merged_")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="dual")
    args = ap.parse_args()
    root, out = Path(args.runs_root), Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    per, tab = {}, {}
    for cfg, lab in ARMS.items():
        f = root / f"{args.prefix}{cfg}" / "aggregate" / "results-summary.json"
        if not f.exists():
            print(f"!! missing {f}")
            continue
        rollouts = json.loads(f.read_text())["rollouts"]
        # `score` lives on the rollout, not in metrics; mirror it into metrics for scene_means
        for r in rollouts:
            if r.get("score") is not None:
                r["metrics"]["score"] = r["score"]
        per[lab] = {k: scene_means(rollouts, k) for k in KEYS}
        tab[lab] = {k: float(np.mean(list(v.values()))) for k, v in per[lab].items() if v}
        t = tab[lab]
        t["d2gt_per_100m"] = 100 * t["dist_to_gt_trajectory"] / t["dist_traveled_m"]
        t["dist_ratio"] = t["dist_traveled_m"] / t["gt_dist_traveled_m"]

    res = {"arms": tab, "within": {}, "paired_vs_ref": {}, "ref": args.ref}
    lines = ["progress vs distance-to-GT in closed loop", "",
             "A. arm level (scene means):"]
    order = sorted(tab, key=lambda a: tab[a]["progress_clipped_rel"])
    for lab in order:
        t = tab[lab]
        lines.append(f"  {lab:9s} progress {t['progress_clipped_rel']:.3f} d2gt "
                     f"{t['dist_to_gt_trajectory']:.3f} (per 100 m {t['d2gt_per_100m']:.3f}) "
                     f"| dist {t['dist_traveled_m']:.1f}/{t['gt_dist_traveled_m']:.1f} = "
                     f"{t['dist_ratio']:.3f} | endpoint gap {t['dist_to_gt_location']:.2f} "
                     f"| score {t['score']:.3f}")
    xs = [tab[a]["progress_clipped_rel"] for a in tab]
    ys = [tab[a]["dist_to_gt_trajectory"] for a in tab]
    yn = [tab[a]["d2gt_per_100m"] for a in tab]
    res["arm_level"] = {"spearman_progress_d2gt": list(map(float, spearmanr(xs, ys))),
                        "spearman_progress_d2gt_per_100m": list(map(float, spearmanr(xs, yn))),
                        "n": len(xs)}
    lines.append(f"  arm-level Spearman(progress, d2gt) "
                 f"{res['arm_level']['spearman_progress_d2gt'][0]:+.3f}; normalised per 100 m "
                 f"{res['arm_level']['spearman_progress_d2gt_per_100m'][0]:+.3f} (n={len(xs)})")

    lines += ["", "B. within arm, scene level: Spearman(progress, d2gt), raw and | dist_traveled:"]
    for lab in order:
        p_, d_, t_ = (per[lab][k] for k in
                      ("progress_clipped_rel", "dist_to_gt_trajectory", "dist_traveled_m"))
        ids = sorted(set(p_) & set(d_) & set(t_))
        raw = spearmanr([p_[i] for i in ids], [d_[i] for i in ids])
        par = partial_spearman([p_[i] for i in ids], [d_[i] for i in ids], [t_[i] for i in ids])
        res["within"][lab] = {"raw": list(map(float, raw)), "partial": list(map(float, par)),
                              "n": len(ids)}
        lines.append(f"  {lab:9s} raw {raw[0]:+.3f} (p={raw[1]:.2g}) -> partial {par[0]:+.3f} "
                     f"(p={par[1]:.2g}), n={len(ids)}")

    lines += ["", f"C. paired scene deltas vs {args.ref}:"]
    for lab in order:
        if lab == args.ref:
            continue
        pa, pb = per[lab]["progress_clipped_rel"], per[args.ref]["progress_clipped_rel"]
        da, db = per[lab]["dist_to_gt_trajectory"], per[args.ref]["dist_to_gt_trajectory"]
        ids = sorted(set(pa) & set(pb) & set(da) & set(db))
        dp = np.array([pa[i] - pb[i] for i in ids])
        dd = np.array([da[i] - db[i] for i in ids])
        rho = spearmanr(dp, dd)
        res["paired_vs_ref"][lab] = {"dprogress": boot_ci(dp), "dd2gt": boot_ci(dd),
                                     "spearman": list(map(float, rho)), "n": len(ids)}
        r = res["paired_vs_ref"][lab]
        lines.append(f"  {lab:9s} dprogress {r['dprogress'][0]:+.4f} "
                     f"[{r['dprogress'][1]:+.4f},{r['dprogress'][2]:+.4f}] dd2gt "
                     f"{r['dd2gt'][0]:+.3f} [{r['dd2gt'][1]:+.3f},{r['dd2gt'][2]:+.3f}] | "
                     f"Spearman(dprogress, dd2gt) {rho[0]:+.3f} (p={rho[1]:.2g})")

    text = "\n".join(lines)
    print(text)
    (out / "progress_d2gt_summary.txt").write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"baseline": MUTED, "dual": C1, "dualr": C2, "tyr_r": C3, "wl": C4}
    ax = axes[0]
    for lab in tab:
        ax.scatter(tab[lab]["progress_clipped_rel"], tab[lab]["dist_to_gt_trajectory"],
                   s=70, color=cols.get(lab, INK), label=lab)
        ax.annotate(lab, (tab[lab]["progress_clipped_rel"], tab[lab]["dist_to_gt_trajectory"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("progress (scene mean)")
    ax.set_ylabel("d2gt (m)")
    ax.set_title("A. arm level: more progress, worse d2gt")
    ax = axes[1]
    ref = args.ref
    for lab in tab:
        if lab == ref:
            continue
        p_, d_ = per[lab]["progress_clipped_rel"], per[lab]["dist_to_gt_trajectory"]
        ids = sorted(set(p_) & set(d_))
        ax.scatter([p_[i] for i in ids], [d_[i] for i in ids], s=8, alpha=0.35,
                   color=cols.get(lab, INK), label=lab)
    p_, d_ = per[ref]["progress_clipped_rel"], per[ref]["dist_to_gt_trajectory"]
    ids = sorted(set(p_) & set(d_))
    ax.scatter([p_[i] for i in ids], [d_[i] for i in ids], s=8, alpha=0.5, color=cols[ref], label=ref)
    ax.set_xlabel("progress (scene)")
    ax.set_ylabel("d2gt (m)")
    ax.set_title("B. scene level, all arms (positive slope)")
    ax.legend(fontsize=7, markerscale=2)
    ax = axes[2]
    for lab in tab:
        ax.scatter(tab[lab]["dist_ratio"], tab[lab]["d2gt_per_100m"], s=70, color=cols.get(lab, INK))
        ax.annotate(lab, (tab[lab]["dist_ratio"], tab[lab]["d2gt_per_100m"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("distance travelled / GT distance")
    ax.set_ylabel("d2gt per 100 m driven")
    ax.set_title("C. normalised by distance")
    fig.tight_layout()
    fig.savefig(out / "plots" / "progress_vs_d2gt.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
