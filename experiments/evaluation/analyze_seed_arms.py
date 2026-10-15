"""Does the seed INSIDE the importance measurement move the pruned model?

`launch_seed_arms.sh`: nine dual_u40_v2 arms on the same calib_100 clips.

  ref      CoC rollout seed 0, flow-matching noise seed 0 (the shipped estimator, re-run)
  c1..c4   a different sampled CoC per clip, noise held
  n1..n4   a different flow-matching noise draw, CoC held

All on test500, paired with the unpruned `baseline_ada_ps_test` and with each other.
Three older runs of the SAME estimator on the SAME clips are read as the floor:
`dual_ada_u40_v2_test` and `dualgtref_u40_v2_ps_test` (Ada re-measurements) and
`dual_u40_v2_ps_test` (the shipped arm, importance measured on Blackwell).

Readings, each against the scales already measured on this set:
  +0.0524   what the 24% pruning itself costs (median, shipped arm)
  0.363     SD of arm-mean minADE over six disjoint n=100 draws (greedy rule)
  0.083     the same SD under the `se` draw rule

The unit is the clip. A group contrast is the paired difference of per-clip means over
the group's four arms, as in analyze_strat_calib.py. Mean with a bootstrap CI is the
headline; median and Wilcoxon are printed beside it. Arms not yet evaluated are skipped.

Usage:
  python experiments/evaluation/analyze_seed_arms.py [--out outputs/seed_arms]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

BASELINE = "baseline_ada_ps_test"
GROUPS = {"coc": [f"c{i}" for i in range(1, 5)], "noise": [f"n{i}" for i in range(1, 5)]}
ARMS = {"ref": ("dual_seed_ref_test", "slim_dual_seed_ref"),
        **{a: (f"dual_seed_{a}_test", f"slim_dual_seed_{a}")
           for g in GROUPS.values() for a in g}}
OLD = {"dual_ada": ("dual_ada_u40_v2_test", "slim_dual_ada_u40_v2"),
       "dualgtref": ("dualgtref_u40_v2_ps_test", "slim_dual_gtref_u40_v2"),
       "shipped": ("dual_u40_v2_ps_test", "slim_dual_u40_v2")}
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
BOOT, EXPECT, K = 20000, 500, 6
SCALE = {"pruning cost (median)": 0.0524, "draw SD, greedy rule": 0.363,
         "draw SD, se rule": 0.083}


def load_run(exp):
    d = REPO / "outputs" / exp
    rows = {}
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("config"):
            continue
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def kept(slim):
    p = REPO / "outputs" / slim / "slim_meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["vlm"]
    return {ax: [set(layer[ax]) for layer in m] for ax in ("q", "mlp")}


def overlap(a, b, ax):
    return sum(len(x & y) for x, y in zip(a[ax], b[ax])) / sum(len(x) for x in a[ax])


def paired(d, rng):
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    mlo, mhi = np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])
    nz = d[d != 0]
    return {"mean": float(d.mean()), "mean_ci": [float(lo), float(hi)],
            "median": float(np.median(d)), "median_ci": [float(mlo), float(mhi)],
            "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0,
            "frac_changed": float((d != 0).mean()),
            "trim10": float(np.sort(d)[len(d) // 10: -len(d) // 10].mean())}


def fmt(s):
    star = "*" if s["mean_ci"][0] > 0 or s["mean_ci"][1] < 0 else " "
    return (f"{s['mean']:+.4f} [{s['mean_ci'][0]:+.4f}, {s['mean_ci'][1]:+.4f}]{star} "
            f"med {s['median']:+.4f} [{s['median_ci'][0]:+.4f}, {s['median_ci'][1]:+.4f}] "
            f"p={s['p']:.2g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/seed_arms")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    base = load_run(BASELINE)
    ids = sorted(base)
    assert len(ids) == EXPECT, len(ids)
    bucket = np.array([base[c]["bucket"] for c in ids])

    def vec(rows, key="ade_rollout_k"):
        return np.array([min(rows[c][key][:K]) for c in ids])

    b_ade = vec(base)
    runs, skipped = {}, []
    for name, (exp, slim) in {**ARMS, **OLD}.items():
        rows = load_run(exp) if (REPO / "outputs" / exp).exists() else {}
        if not all(c in rows for c in ids):
            skipped.append(f"{name} ({len(rows)}/{EXPECT})")
            continue
        runs[name] = {"ade": vec(rows), "fde": vec(rows, "fde_rollout_k"),
                      "degen": float(np.mean([rows[c]["coc_degenerate"] for c in ids])),
                      "text": [rows[c]["gen_coc"] for c in ids], "kept": kept(slim)}
    if skipped:
        say("skipped (incomplete): " + ", ".join(skipped))
    assert "ref" in runs, "reference arm missing"
    ref = runs["ref"]
    metrics = {"n_clips": EXPECT, "k": K, "baseline_minade": float(b_ade.mean()),
               "arms": {}, "scales": SCALE}

    say(f"baseline minADE@6 mean {b_ade.mean():.4f} median {np.median(b_ade):.4f}")
    say("\n== arms on test500 (minADE@6)")
    say(f"{'arm':10s}{'mean':>8s}{'median':>8s}{'minFDE':>8s}{'degen':>7s}"
        f"{'Q ovl':>8s}{'MLP ovl':>9s}{'same CoC':>10s}   delta vs unpruned")
    for name, r in runs.items():
        vs_b = paired(r["ade"] - b_ade, rng)
        vs_r = paired(r["ade"] - ref["ade"], rng) if name != "ref" else None
        ov = ({ax: overlap(ref["kept"], r["kept"], ax) for ax in ("q", "mlp")}
              if r["kept"] and ref["kept"] else {"q": float("nan"), "mlp": float("nan")})
        same = float(np.mean([a == b for a, b in zip(r["text"], ref["text"])]))
        metrics["arms"][name] = {"mean": float(r["ade"].mean()),
                                 "median": float(np.median(r["ade"])),
                                 "minfde": float(r["fde"].mean()), "degen": r["degen"],
                                 "overlap_vs_ref": ov, "same_coc_as_ref": same,
                                 "vs_baseline": vs_b, "vs_ref": vs_r}
        say(f"{name:10s}{r['ade'].mean():8.4f}{np.median(r['ade']):8.4f}{r['fde'].mean():8.4f}"
            f"{r['degen']:7.3f}{ov['q']:8.4f}{ov['mlp']:9.4f}{same:10.3f}   "
            f"med {vs_b['median']:+.4f} mean {vs_b['mean']:+.4f}")

    say("\n== each arm minus ref (paired, 500 clips)")
    for name in runs:
        if name != "ref":
            say(f"{name:10s} {fmt(metrics['arms'][name]['vs_ref'])}  "
                f"trim10 {metrics['arms'][name]['vs_ref']['trim10']:+.4f}")

    # group level
    say("\n== groups")
    metrics["groups"] = {}
    gm = {}
    for g, arms in GROUPS.items():
        have = [a for a in arms if a in runs]
        if not have:
            continue
        gm[g] = np.mean([runs[a]["ade"] for a in have], axis=0)
        means = [float(runs[a]["ade"].mean()) for a in have]
        meds = [float(np.median(runs[a]["ade"])) for a in have]
        iu = [(a, b) for i, a in enumerate(have) for b in have[i + 1:]]
        pair = [float(np.abs(np.median(runs[a]["ade"] - runs[b]["ade"]))) for a, b in iu]
        ovq = [overlap(runs[a]["kept"], runs[b]["kept"], "q") for a, b in iu]
        ovm = [overlap(runs[a]["kept"], runs[b]["kept"], "mlp") for a, b in iu]
        s = paired(gm[g] - ref["ade"], rng)
        metrics["groups"][g] = {
            "arms": have, "arm_means": means, "sd_of_means": float(np.std(means, ddof=1)),
            "range_of_means": float(max(means) - min(means)),
            "sd_of_medians": float(np.std(meds, ddof=1)),
            "pairwise_abs_median_delta": pair, "pairwise_q_overlap": float(np.mean(ovq)),
            "pairwise_mlp_overlap": float(np.mean(ovm)), "group_minus_ref": s}
        say(f"{g:6s} arms {have}: means {' '.join(f'{m:.4f}' for m in means)}  "
            f"SD {np.std(means, ddof=1):.4f}  range {max(means) - min(means):.4f}  "
            f"SD of medians {np.std(meds, ddof=1):.4f}")
        say(f"       within-group kept overlap Q {np.mean(ovq):.4f} MLP {np.mean(ovm):.4f}; "
            f"pairwise |median delta| max {max(pair):.4f}")
        say(f"       group - ref: {fmt(s)}")
    if len(gm) == 2:
        s = paired(gm["coc"] - gm["noise"], rng)
        metrics["coc_minus_noise"] = s
        say(f"coc - noise (per-clip group means): {fmt(s)}")
        say("by bucket:")
        metrics["coc_minus_noise_by_bucket"] = {}
        for bk in BUCKETS:
            sel = bucket == bk
            sb = paired((gm["coc"] - gm["noise"])[sel], rng)
            metrics["coc_minus_noise_by_bucket"][bk] = sb
            say(f"  {bk:11s} n={int(sel.sum()):3d} {fmt(sb)}")
    allm = [metrics["arms"][a]["mean"] for a in ARMS if a in runs]
    metrics["sd_all_seed_arms"] = float(np.std(allm, ddof=1)) if len(allm) > 1 else None
    say(f"\nall {len(allm)} seed arms: mean of means {np.mean(allm):.4f}, "
        f"SD {metrics['sd_all_seed_arms']:.4f}, range {max(allm) - min(allm):.4f}")
    say("scales: " + "; ".join(f"{k} {v}" for k, v in SCALE.items()))

    # the tail: is an arm's mean a few clips?
    say("\n== where a mean delta vs ref comes from: share carried by the 5 largest |delta| clips")
    for name in runs:
        if name == "ref":
            continue
        d = runs[name]["ade"] - ref["ade"]
        top = np.argsort(-np.abs(d))[:5]
        sh = float(d[top].sum() / d.sum()) if d.sum() != 0 else float("nan")
        big = np.abs(d) > 0.05
        metrics["arms"][name]["top5_share"] = sh
        metrics["arms"][name]["n_moved_005"] = [int((d[big] > 0).sum()), int((d[big] < 0).sum())]
        say(f"{name:10s} top-5 share {sh:+.2f}; clips moving > 0.05 m: "
            f"{int((d[big] > 0).sum())} worse / {int((d[big] < 0).sum())} better")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({
        "baseline": BASELINE, "arms": ARMS, "old": OLD, "groups": GROUPS, "k": K,
        "boot": BOOT, "calibration": "calib_100", "recipe": "dual_u40_v2"}, indent=2))

    order = [a for a in ["shipped", "dual_ada", "dualgtref", "ref"] + GROUPS["coc"] + GROUPS["noise"]
             if a in runs]
    col = {**{a: MUTED for a in OLD}, "ref": C4, **{a: C3 for a in GROUPS["coc"]},
           **{a: C1 for a in GROUPS["noise"]}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    xs = np.arange(len(order))
    for ax, key, title in ((axes[0], "mean", "mean"), (axes[1], "median", "median")):
        d = [metrics["arms"][a]["vs_baseline"] for a in order]
        y = [s[key] for s in d]
        lo = [s[key] - s[f"{key}_ci"][0] for s in d]
        hi = [s[f"{key}_ci"][1] - s[key] for s in d]
        ax.bar(xs, y, color=[col[a] for a in order], width=0.65)
        ax.errorbar(xs, y, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_xticks(xs, order, rotation=30, ha="right")
        ax.set_title(f"minADE@6 minus unpruned, paired {title} (95% CI)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "arm_cost.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = [a for a in order if a != "ref"]
    xs = np.arange(len(names))
    d = [metrics["arms"][a]["vs_ref"] for a in names]
    ax.bar(xs, [s["mean"] for s in d], color=[col[a] for a in names], width=0.65)
    ax.errorbar(xs, [s["mean"] for s in d],
                yerr=[[s["mean"] - s["mean_ci"][0] for s in d],
                      [s["mean_ci"][1] - s["mean"] for s in d]],
                fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.plot(xs, [s["median"] for s in d], "o", color=INK, ms=4)
    ax.axhline(0, color=MUTED, lw=0.8)
    for v, ls in ((SCALE["pruning cost (median)"], "--"), (-SCALE["pruning cost (median)"], "--")):
        ax.axhline(v, color=MUTED, lw=0.8, ls=ls)
    ax.set_xticks(xs, names, rotation=30, ha="right")
    ax.set_title("arm minus ref: paired mean (bar, 95% CI) and median (dot); dashed = +-pruning cost")
    fig.tight_layout()
    fig.savefig(out / "plots" / "vs_ref.png", dpi=150)
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
