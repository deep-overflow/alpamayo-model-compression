"""Difficulty-stratified closed-loop comparison over the 150-scene alpasim matrix.

Splits the 150 scenes of `public_2601` into easy / medium / hard by the unpruned
baseline's own scene score, then re-reads every arm inside each stratum. Because
grouping on the baseline's own rollouts and then reading a delta against that same
baseline is regression to the mean, the split is also done the debiased way --
rollout 1 defines the stratum, rollout 2 supplies the baseline value -- and both are
reported. The composite scene score gates on offroad / at-fault collision, so the gate
rate is decomposed too: two arms can tie on score with opposite safety profiles.

Reads soowon's LLM-Pruner runs from a second runs-root; they cover the same 150 scenes.

`--full-suite-run` points at sangoh's unpruned run over all 913 `public_2601` scenes
(1 rollout each). The static-difficulty features were chosen by looking at the 150 run
scenes, so that panel is in-sample on its own; the 913-scene run supplies the 763 scenes
the matrix never touched and turns it into an out-of-sample test.

Usage:
  python analyze_difficulty_strat.py \
      --runs-root /home/cvlab21/project/chan/alpasim-runs \
      --lp-root /mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-runs \
      --full-suite-run /mnt/nvme1n1/ad_vla/results/sangoh/alpasim_runs/eval2601_a1_5 \
      --out outputs/difficulty_strat
"""

import argparse
import collections
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
ACC = "#D97757"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# usdz-derived static difficulty (ego GT kinematics), built by outputs/scene_difficulty
FEATS = REPO / "outputs" / "scene_difficulty" / "scene_feats_public2601.json"


def load_run(d):
    per = collections.defaultdict(list)
    for r in json.loads((d / "aggregate" / "results-summary.json").read_text())["rollouts"]:
        per[r["clipgt_id"]].append(r)
    return per


def gated(r):
    return r["metrics"]["offroad"] > 0 or r["metrics"]["collision_at_fault"] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--lp-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-suite-run", type=Path, default=None,
                    help="unpruned baseline over all 913 public_2601 scenes (1 rollout); turns the "
                         "static-difficulty panel into an out-of-sample test on the 763 scenes the "
                         "150-scene matrix never touched")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    runs = {d.name.replace("m2601_merged_", ""): load_run(d)
            for d in sorted(args.runs_root.glob("m2601_merged_*"))}
    for name, sub in [("lp_r50", "cl150_merged_lp_r50"), ("lp_r50_dual", "cl150_merged_lp_r50_dual")]:
        runs[name] = load_run(args.lp_root / sub)

    scenes = sorted(runs["baseline"])
    runs = {k: v for k, v in runs.items() if set(v) >= set(scenes)}
    arms = [a for a in runs if a != "baseline"]
    SC = {a: np.array([np.mean([r["score"] for r in runs[a][s]]) for s in scenes]) for a in runs}
    b = SC["baseline"]
    r1 = np.array([runs["baseline"][s][0]["score"] for s in scenes])
    r2 = np.array([runs["baseline"][s][1]["score"] for s in scenes])

    def rate(a, key):
        rr = [r for s in scenes for r in runs[a][s]]
        if key == "gate":
            return float(np.mean([gated(r) for r in rr]))
        return float(np.mean([r["metrics"][key] > 0 for r in rr]))

    # --- strata: raw (biased) and debiased (rollout1 splits, rollout2 reports)
    g_raw = np.where(b >= 0.9, 0, np.where(b >= 0.4, 1, 2))
    g_deb = np.where(r1 >= 0.9, 0, np.where(r1 >= 0.4, 1, 2))
    GL = ["easy", "medium", "hard"]

    M = {"n_scenes": len(scenes), "n_arms": len(arms),
         "strata": {"raw": {GL[i]: int((g_raw == i).sum()) for i in range(3)},
                    "debiased": {GL[i]: int((g_deb == i).sum()) for i in range(3)}},
         "baseline": {"overall": float(b.mean()),
                      "raw": {GL[i]: float(b[g_raw == i].mean()) for i in range(3)},
                      "debiased": {GL[i]: float(r2[g_deb == i].mean()) for i in range(3)},
                      "gate": rate("baseline", "gate"), "offroad": rate("baseline", "offroad"),
                      "at_fault": rate("baseline", "collision_at_fault")},
         "arms": {}}
    for a in arms:
        v = SC[a]
        d = v - b
        p = float(wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
        M["arms"][a] = {
            "overall": float(v.mean()), "delta": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))), "wilcoxon_p": p,
            "raw": {GL[i]: float(v[g_raw == i].mean()) for i in range(3)},
            "raw_delta": {GL[i]: float(v[g_raw == i].mean() - b[g_raw == i].mean()) for i in range(3)},
            "debiased": {GL[i]: float(v[g_deb == i].mean()) for i in range(3)},
            "debiased_delta": {GL[i]: float(v[g_deb == i].mean() - r2[g_deb == i].mean()) for i in range(3)},
            "gate": rate(a, "gate"), "offroad": rate(a, "offroad"),
            "at_fault": rate(a, "collision_at_fault"),
        }

    # --- static (usdz GT geometry) difficulty, for the selection-rule validation panel
    if FEATS.exists():
        A = json.loads(FEATS.read_text())
        def z(v):
            return (v - v.mean()) / (v.std() + 1e-9)

        hs = z(np.array([r["v_mean"] for r in A])) + z(np.array([r["yaw_total_deg"] for r in A]))
        H = {r["scene_id"]: hs[i] for i, r in enumerate(A)}
        h = np.array([H[s] for s in scenes])
        cuts = np.percentile(hs, [0, 20, 40, 60, 80, 90, 100])
        bands, static = ["0-20%", "20-40%", "40-60%", "60-80%", "80-90%", "90-100%"], []

        # the 913-scene unpruned run turns this panel from in-sample (the features were picked by
        # looking at these 150) into a genuine out-of-sample test on the 763 never-run scenes
        full = load_run(args.full_suite_run) if args.full_suite_run else {}
        FS = {k: float(np.mean([r["score"] for r in v])) for k, v in full.items()}
        unseen = sorted(set(FS) - set(scenes))

        for i, lab in enumerate(bands):
            m = (h >= cuts[i]) & ((h <= cuts[i + 1]) if i == 5 else (h < cuts[i + 1]))
            ma = (hs >= cuts[i]) & ((hs <= cuts[i + 1]) if i == 5 else (hs < cuts[i + 1]))
            row = {"band": lab, "n_150": int(m.sum()), "n_913": int(ma.sum()),
                   "baseline_score": float(b[m].mean()) if m.any() else None}
            if unseen:
                u = [k for k in unseen
                     if H[k] >= cuts[i] and (H[k] <= cuts[i + 1] if i == 5 else H[k] < cuts[i + 1])]
                row["n_unseen"] = len(u)
                row["unseen_score"] = float(np.mean([FS[k] for k in u])) if u else None
            static.append(row)
        M["static_bands"] = static

        if unseen:
            xf = np.array([H[k] for k in FS])
            yf = np.array([FS[k] for k in FS])
            xu = np.array([H[k] for k in unseen])
            yu = np.array([FS[k] for k in unseen])
            gp = {r["scene_id"]: r["gt_path_m"] for r in A}
            pool = [k for k in unseen if gp[k] >= 5.0]          # <5 m GT auto-scores 1.0
            top = sorted(pool, key=lambda k: -H[k])[:100]
            bot = sorted(pool, key=lambda k: H[k])[:100]
            yt = np.array([FS[k] for k in top])
            yb = np.array([FS[k] for k in bot])
            shared = [k for k in FS if k in set(scenes)]
            a_sh = np.array([FS[k] for k in shared])
            b_sh = np.array([SC["baseline"][scenes.index(k)] for k in shared])
            M["full_suite"] = {
                "run": str(args.full_suite_run), "n_scenes": len(FS),
                "n_rollouts_per_scene": int(np.median([len(v) for v in full.values()])),
                "mean_score_913": float(yf.mean()),
                "spearman_913": float(spearmanr(xf, yf).statistic),
                "spearman_unseen": float(spearmanr(xu, yu).statistic),
                "spearman_unseen_p": float(spearmanr(xu, yu).pvalue),
                "pearson_unseen": float(np.corrcoef(xu, yu)[0, 1]),
                "hard100_score": float(yt.mean()), "easy100_score": float(yb.mean()),
                "hard_minus_easy": float(yt.mean() - yb.mean()),
                "hard_minus_easy_p": float(mannwhitneyu(yt, yb).pvalue),
                # does the 150-scene matrix represent the suite it is drawn from?
                "first150_score": float(a_sh.mean()), "rest763_score": float(yu.mean()),
                "first150_vs_rest_p": float(mannwhitneyu(a_sh, yu).pvalue),
                # cross-run agreement on the shared 150, against this run's own repeat ceiling
                "cross_run_r": float(np.corrcoef(a_sh, b_sh)[0, 1]),
                "cross_run_wilcoxon_p": float(wilcoxon(a_sh - b_sh).pvalue),
                "within_run_r1r2_r": float(np.corrcoef(r1, r2)[0, 1]),
            }

    (out / "metrics.json").write_text(json.dumps(M, indent=2))
    (out / "config.json").write_text(json.dumps(
        {"runs_root": str(args.runs_root), "lp_root": str(args.lp_root),
         "full_suite_run": str(args.full_suite_run) if args.full_suite_run else None,
         "suite": "public_2601", "n_scenes": len(scenes), "n_rollouts": 2,
         "strata_cuts": {"easy": ">=0.9", "medium": "0.4-0.9", "hard": "<0.4"}}, indent=2))

    order = sorted(arms, key=lambda a: -SC[a].mean())

    # 1) per-arm delta inside each stratum (debiased)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(order))
    w = 0.27
    for k, (i, c, lab) in enumerate([(0, C1, "easy"), (1, C4, "medium"), (2, C2, "hard")]):
        ax.bar(x + (k - 1) * w, [M["arms"][a]["debiased_delta"][GL[i]] for a in order],
               w, color=c, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("slim_", "") for a in order], rotation=45, ha="right", fontsize=8)
    for i, a in enumerate(order):
        if a.startswith("lp_"):
            ax.get_xticklabels()[i].set_color(ACC)
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_ylabel("scene score delta vs baseline")
    ax.set_title("difficulty stratum x arm (debiased: rollout1 splits, rollout2 reports)"
                 "   orange = LLM-Pruner")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(plots / "arm_by_difficulty.png", dpi=150)
    plt.close(fig)

    # 2) safety profile: offroad vs at-fault
    def prof(a):
        m = M["baseline"] if a == "baseline" else M["arms"][a]
        return 100 * m["offroad"], 100 * m["at_fault"]

    fig, ax = plt.subplots(figsize=(7.2, 6))
    labelled = {"baseline", "lp_r50", "lp_r50_dual", "slim_dual_u40_v2", "slim_j_u40_v2",
                "slim_wanda_u40_v2", "slim_coc_u40_v2", "slim_recover_dual_u55"}
    for a in ["baseline"] + order:
        ox, ay = prof(a)
        lp = a.startswith("lp_")
        col = ACC if lp else (INK if a == "baseline" else C1)
        ax.scatter(ox, ay, s=110 if (lp or a == "baseline") else 45, color=col, zorder=3,
                   edgecolor=BG, linewidth=1.2)
        if a in labelled:
            ax.annotate(a.replace("slim_", ""), (ox, ay), fontsize=8.5, xytext=(6, 4),
                        textcoords="offset points", color=col,
                        fontweight="bold" if (lp or a == "baseline") else "normal")
    bx, by = prof("baseline")
    ax.axvline(bx, color=MUTED, lw=0.7, ls=":")
    ax.axhline(by, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("offroad rate (%, per rollout)")
    ax.set_ylabel("at-fault collision rate (%, per rollout)")
    ax.set_title("safety profile - dotted lines are the unpruned baseline")
    fig.tight_layout()
    fig.savefig(plots / "safety_profile.png", dpi=150)
    plt.close(fig)

    # 3) our arms trade score against collisions monotonically; LP does not sit on that line
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    names = order + ["baseline"]
    sc = np.array([(M["baseline"] if a == "baseline" else M["arms"][a])["overall"] for a in names])
    af = np.array([100 * (M["baseline"] if a == "baseline" else M["arms"][a])["at_fault"]
                   for a in names])
    ours = np.array([not a.startswith("lp_") for a in names])
    slope, icept = np.polyfit(sc[ours], af[ours], 1)
    r_ours = float(np.corrcoef(sc[ours], af[ours])[0, 1])
    xs = np.linspace(sc.min() - 0.01, sc.max() + 0.01, 50)
    ax.plot(xs, slope * xs + icept, color=MUTED, lw=1.1, ls="--", zorder=1,
            label=f"trend over our 15 arms (r = {r_ours:+.2f})")
    for s, f, a in zip(sc, af, names):
        lp = a.startswith("lp_")
        col = ACC if lp else (INK if a == "baseline" else C1)
        ax.scatter(s, f, s=110 if (lp or a == "baseline") else 45, color=col, zorder=3,
                   edgecolor=BG, linewidth=1.2)
        if lp:  # residual above the trend is the point of the panel
            ax.plot([s, s], [slope * s + icept, f], color=ACC, lw=1.0, ls=":", zorder=2)
        if a in labelled:
            ax.annotate(a.replace("slim_", ""), (s, f), fontsize=8.5, xytext=(6, 4),
                        textcoords="offset points", color=col,
                        fontweight="bold" if (lp or a == "baseline") else "normal")
    resid = {a: af[i] - (slope * sc[i] + icept) for i, a in enumerate(names) if a.startswith("lp_")}
    M["lp_atfault_residual_pp"] = {k: float(v) for k, v in resid.items()}
    ax.set_xlabel("scene score (mean over 150 scenes)")
    ax.set_ylabel("at-fault collision rate (%)")
    ax.set_title("our arms buy score by avoiding collisions; LLM-Pruner does not")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "score_vs_atfault.png", dpi=150)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(M, indent=2))

    # 4) static difficulty band validation, in-sample and (with --full-suite-run) out-of-sample
    if "static_bands" in M:
        sb = M["static_bands"]
        oos = "full_suite" in M
        fig, axes = plt.subplots(1, 2 if oos else 1, figsize=(12.4 if oos else 7.6, 4.6))
        ax = axes[0] if oos else axes
        xs = np.arange(len(sb))
        if oos:
            w = 0.38
            ax.bar(xs - w / 2, [s["baseline_score"] for s in sb], w, color=C1,
                   label="150 run scenes (in-sample, 2 rollouts)")
            ax.bar(xs + w / 2, [s["unseen_score"] for s in sb], w, color=C2,
                   label="763 never-run scenes (out-of-sample, 1 rollout)")
            ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        else:
            ax.bar(xs, [s["baseline_score"] for s in sb], 0.6, color=C1)
            for i, s in enumerate(sb):
                ax.text(i, s["baseline_score"] + 0.015, f"n={s['n_150']}", ha="center",
                        fontsize=8.5, color=MUTED)
        ax.set_xticks(xs)
        ax.set_xticklabels([s["band"] for s in sb], fontsize=9)
        ax.set_xlabel("static difficulty band (hard_score percentile over 913 scenes, usdz GT only)")
        ax.set_ylabel("measured baseline scene score")
        ax.set_ylim(0, 1.05)
        ax.set_title("difficulty scored without running the model tracks the measured score")

        if oos:
            fsm = M["full_suite"]
            ax2 = axes[1]
            vals = [fsm["easy100_score"], fsm["mean_score_913"], fsm["hard100_score"]]
            labs = ["easy100\n(pool bottom)", "all 913", "hard100\n(pool top)"]
            ax2.bar(np.arange(3), vals, 0.55, color=[C2, MUTED, ACC])
            for i, v in enumerate(vals):
                ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=10, color=INK)
            ax2.set_xticks(np.arange(3))
            ax2.set_xticklabels(labs, fontsize=9)
            ax2.set_ylabel("measured baseline scene score")
            ax2.set_ylim(0, 1.05)
            ax2.set_title(f"hard100 selection holds: {fsm['hard_minus_easy']:+.3f} "
                          f"(p={fsm['hard_minus_easy_p']:.1e})")
        fig.tight_layout()
        fig.savefig(plots / "static_validation.png", dpi=150)
        plt.close(fig)

    head = (f"baseline {b.mean():.3f}  (easy {M['baseline']['raw']['easy']:.3f} / "
            f"medium {M['baseline']['raw']['medium']:.3f} / "
            f"hard {M['baseline']['raw']['hard']:.3f})")
    lines = [f"difficulty-stratified arms - {len(scenes)} scenes x 2 rollouts, {len(arms)} arms",
             head, ""]
    lines.append(f"{'arm':28s}{'score':>8s}{'delta':>9s}{'p':>8s}{'gate%':>8s}{'offr%':>8s}{'atflt%':>8s}")
    for a in order:
        m = M["arms"][a]
        lines.append(f"{a:28s}{m['overall']:8.3f}{m['delta']:+9.3f}{m['wilcoxon_p']:8.3f}"
                     f"{100*m['gate']:8.1f}{100*m['offroad']:8.1f}{100*m['at_fault']:8.1f}")
    lines.append(f"{'baseline':28s}{b.mean():8.3f}{0.0:+9.3f}{1.0:8.3f}"
                 f"{100*M['baseline']['gate']:8.1f}{100*M['baseline']['offroad']:8.1f}"
                 f"{100*M['baseline']['at_fault']:8.1f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
