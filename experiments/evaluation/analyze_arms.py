"""Read the criterion arms and decide whether the J-lens replaces the CoC labels.

`dual_u40_v2` and `jtraj_u40_v2` are the same config except for X in
`max(rank I_traj, rank X)`: X = CoC NLL Taylor (needs reference reasoning text) vs the
J-lens score (needs nothing but the model). Budget, allocation, expert tower and KV
groups are held identical, and both draw every score from the same 100 calibration
clips, so a difference between them is the criterion and nothing else.

`baseline_ada` is the unpruned model re-evaluated on the same cards as the arms. The
2026-08-07 baseline ran on Blackwell and determinism only holds within one architecture,
so the shipped numbers cannot be paired against an Ada run without carrying a kernel
difference into every delta.

Gates (pre-registered in plans/2026-08-09_criterion-oneshot.md):
  C1  primary. Pooled in-dist val+test, paired minADE(jtraj - dual): |median| < 0.05 m
      and Wilcoxon p > 0.05 -> the J-lens substitutes for the CoC labels.
  C2  OOD nll_gtcoc, same pairing. The J-lens only measures writes that read out through
      the unembedding, so if the price of dropping labels shows up anywhere it is here.
  C3  both arms keep CoC degeneracy below 0.05, or C1 is void because an arm collapsed.

minADE is heavy-tailed (one broken clip lands at 25 m), so the median and the Wilcoxon
are primary and the mean is shown beside them rather than trusted alone.

`--k K` re-reduces every metric to minADE@K instead of the K the run used. Seeds are
`base + k`, so the first K samples of an 8-sample run are exactly what a K-sample run
would have drawn -- provided the run stored per-sample arrays (`ade_rollout_k`), which
`run_baseline.py` does from 2026-08-11 on. Older rows carry only the minimum and are
rejected rather than silently reported at the wrong K.

Usage:
  python experiments/evaluation/analyze_arms.py
  python experiments/evaluation/analyze_arms.py --sets indist test   # before OOD lands
  python experiments/evaluation/analyze_arms.py --k 6 --out arms_k6  # minADE@6
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el
from analyze_baseline import BUCKETS, describe, load_rows

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

GATE_C1_M = 0.05     # smallest minADE shift this protocol resolves
GATE_C3_DEGEN = 0.05
# arm tag -> (exp-id prefix written by launch_arms.sh, rows-file tag written by run_baseline)
ARMS = {
    "baseline": ("baseline_ada_", "baseline"),
    "traj": ("traj_u40_v2_", "slim_traj_u40_v2"),
    "coc": ("coc_u40_v2_", "slim_coc_u40_v2"),
    "j": ("j_u40_v2_", "slim_j_u40_v2"),
    "dual": ("dual_u40_v2_", "slim_dual_u40_v2"),
    "jtraj": ("jtraj_u40_v2_", "slim_jtraj_u40_v2"),
    # ratio sweep: same criteria, uniform budget raised from 0.3986 to 0.55 / 0.75
    # (24.0% -> 33.1% / 45.0% of the model). exp-ids are <arm>_u<pct>_<set>.
    "dual_u55": ("dual_u55_", "slim_dual_u55_v2"),
    "jtraj_u55": ("jtraj_u55_", "slim_jtraj_u55_v2"),
    "dual_u75": ("dual_u75_", "slim_dual_u75_v2"),
    "jtraj_u75": ("jtraj_u75_", "slim_jtraj_u75_v2"),
}
ARM_ORDER = ["baseline", "traj", "coc", "j", "dual", "jtraj",
             "dual_u55", "jtraj_u55", "dual_u75", "jtraj_u75"]
COLORS = {"baseline": MUTED, "traj": C4, "coc": C3, "j": "#7b5bd6",
          "dual": C1, "jtraj": C2,
          # the sweep keeps each criterion's hue and darkens it as the budget rises
          "dual_u55": "#1f5aa0", "dual_u75": "#123a69",
          "jtraj_u55": "#006200", "jtraj_u75": "#004100"}
# every pruned arm against the unpruned reference, then the contrasts the gates read.
# an explicit list rather than all 15 pairs: each one costs a 10k-resample bootstrap
# per metric per set, and the other pairs answer no pre-registered question.
CONTRASTS = [("traj_vs_baseline", "baseline", "traj"),
             ("coc_vs_baseline", "baseline", "coc"),
             ("j_vs_baseline", "baseline", "j"),
             ("dual_vs_baseline", "baseline", "dual"),
             ("jtraj_vs_baseline", "baseline", "jtraj"),
             ("jtraj_vs_dual", "dual", "jtraj"),      # C1: labels needed?
             ("j_vs_coc", "coc", "j"),                # G2: same question, single criterion
             ("dual_vs_traj", "traj", "dual"),        # G1: does the CoC half buy anything?
             ("jtraj_vs_traj", "traj", "jtraj")]      # G1: same, label-free half


def reduce_to_k(rows, k, arm, which):
    """Rewrite the min-over-samples metrics as min over the first `k` samples."""
    for r in rows:
        for arr, out in (("ade_rollout_k", "minADE_rollout"), ("fde_rollout_k", "minFDE_rollout"),
                         ("ade_tf_k", "minADE_tf"), ("fde_tf_k", "minFDE_tf")):
            if out not in r:
                continue
            if arr not in r:
                raise SystemExit(
                    f"{arm}/{which}: --k {k} needs per-sample arrays, but these rows predate "
                    "them (only the minimum was stored). Re-run run_baseline.py.")
            if len(r[arr]) < k:
                raise SystemExit(f"{arm}/{which}: only {len(r[arr])} samples stored, need {k}")
            r[out] = float(min(r[arr][:k]))
    return rows


def paired(a_rows, b_rows, metric):
    """Per-clip b - a over the clips both arms evaluated. Positive = b is worse."""
    a = {r["clip_id"]: r for r in a_rows}
    b = {r["clip_id"]: r for r in b_rows}
    ids = sorted(set(a) & set(b))
    ids = [i for i in ids if metric in a[i] and metric in b[i]]
    if not ids:
        return None
    d = np.array([b[i][metric] - a[i][metric] for i in ids])
    mean, lo, hi = el.paired_bootstrap_ci(d)
    return {"n": len(ids), "mean": mean, "ci": [lo, hi], "median": float(np.median(d)),
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "frac_worse": float(np.mean(d > 0))}


def pooled(rows_by_set, sets, a, b, metric):
    """Same pairing, concatenated across sets -- C1 reads val and test together."""
    merged = {arm: [r for s in sets if s in rows_by_set for r in rows_by_set[s].get(arm, [])]
              for arm in (a, b)}
    if not merged[a] or not merged[b]:
        return None
    return paired(merged[a], merged[b], metric)


def arm_stats(rows):
    out = {"n": len(rows)}
    for m in ("minADE_rollout", "minFDE_rollout", "nll_self"):
        out[m] = describe([r[m] for r in rows])
    if "minADE_tf" in rows[0]:
        for m in ("minADE_tf", "nll_gtcoc"):
            out[m] = describe([r[m] for r in rows])
    out["coc_degen"] = float(np.mean([r["coc_degenerate"] for r in rows]))
    out["coc_len_median"] = float(np.median([r["coc_len"] for r in rows]))
    out["by_bucket"] = {
        b: describe([r["minADE_rollout"] for r in rows if r["bucket"] == b]) or {}
        for b in BUCKETS}
    return out


def d(x, unit="m"):
    if x is None:
        return "n/a"
    return (f"med {x['median']:+.4f}{unit}  mean {x['mean']:+.4f} "
            f"[{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}]  p={x['p']:.2e}  "
            f"worse {x['frac_worse'] * 100:.0f}%  n={x['n']}")


def plots(res, rows_by_set, sets, out_dir):
    arms = [a for a in ARM_ORDER if any(a in rows_by_set.get(s, {}) for s in sets)]
    colors = COLORS

    n_forest = sum(1 for s in sets for pair, _, _ in CONTRASTS
                   if res.get("paired", {}).get(s, {}).get(pair, {}).get("minADE_rollout"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, max(4.3, 0.30 * n_forest)))
    ax = axes[0]
    w = 0.8 / max(len(arms), 1)
    x = np.arange(len(sets))
    for j, a in enumerate(arms):
        med = [res[s][a]["minADE_rollout"]["median"] if a in res.get(s, {}) else np.nan
               for s in sets]
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, med, w, color=colors[a], label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylabel("median minADE (m)")
    ax.set_title("trajectory error by arm")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]  # the primary reading: jtraj - dual, per clip
    for s, c in zip(sets, (C1, C4, C3)):
        p = res.get("paired", {}).get(s, {}).get("jtraj_vs_dual", {}).get("minADE_rollout")
        if not p:
            continue
        a_r, b_r = rows_by_set[s]["dual"], rows_by_set[s]["jtraj"]
        m = {r["clip_id"]: r for r in a_r}
        n = {r["clip_id"]: r for r in b_r}
        ids = sorted(set(m) & set(n))
        dd = np.array([n[i]["minADE_rollout"] - m[i]["minADE_rollout"] for i in ids])
        ax.hist(np.clip(dd, -1, 1), bins=np.linspace(-1, 1, 61), histtype="step",
                lw=1.6, color=c, label=f"{s} (med {np.median(dd):+.3f})")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvspan(-GATE_C1_M, GATE_C1_M, color=C2, alpha=0.10)
    ax.set_xlabel("per-clip minADE:  jtraj - dual  (m, clipped at +-1)")
    ax.set_ylabel("clips")
    ax.set_title(f"gate C1 band +-{GATE_C1_M} m shaded")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]  # forest plot of every paired delta
    labels, mid, lo, hi = [], [], [], []
    for s in sets:
        for pair, _, _ in CONTRASTS:
            p = res.get("paired", {}).get(s, {}).get(pair, {}).get("minADE_rollout")
            if p:
                labels.append(f"{s}: {pair.replace('_vs_', ' - ')}")
                mid.append(p["mean"])
                lo.append(p["mean"] - p["ci"][0])
                hi.append(p["ci"][1] - p["mean"])
    y = np.arange(len(labels))
    ax.errorbar(mid, y, xerr=[lo, hi], fmt="o", color=C1, ecolor=MUTED, capsize=3, ms=4)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("paired mean dminADE (m), 95% CI")
    ax.set_title("all paired contrasts")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "arms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    x = np.arange(len(BUCKETS))
    for j, a in enumerate(arms):
        s0 = sets[0]
        med = [res[s0][a]["by_bucket"].get(b, {}).get("median", np.nan) for b in BUCKETS]
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, med, w, color=colors[a], label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS, rotation=20)
    ax.set_ylabel("median minADE (m)")
    ax.set_title(f"by scenario bucket ({sets[0]})")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    deg = [[res[s][a]["coc_degen"] if a in res.get(s, {}) else np.nan for s in sets]
           for a in arms]
    x = np.arange(len(sets))
    for j, (a, v) in enumerate(zip(arms, deg)):
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, v, w, color=colors[a], label=a)
    ax.axhline(GATE_C3_DEGEN, color=C4, ls="--", lw=1, label=f"gate C3 {GATE_C3_DEGEN}")
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylabel("CoC degeneracy rate")
    ax.set_title("reasoning channel intact?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "buckets_coc.png", dpi=150)
    plt.close(fig)

    if "ood" not in rows_by_set or "dual" not in rows_by_set["ood"]:
        return
    # OOD is the only set where every arm is handed the SAME curated CoC, so it is the
    # only place the trajectory head and the reasoning channel can be told apart.
    O = rows_by_set["ood"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax = axes[0]
    pairs = [("minADE_rollout", "own rollout CoC"), ("minADE_tf", "same GT CoC")]
    x = np.arange(len(pairs))
    for j, a in enumerate(arms):
        v = [np.median([r[m] for r in O[a]]) for m, _ in pairs]
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, v, w, color=colors[a], label=a)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in pairs])
    ax.set_ylabel("median minADE (m)")
    ax.set_title("does the gap survive identical reasoning?")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for j, a in enumerate(arms):
        g = np.array([r["nll_gtcoc"] for r in O[a]])
        m, lo, hi = el.paired_bootstrap_ci(g)
        ax.errorbar([m], [j], xerr=[[m - lo], [hi - m]], fmt="o", color=colors[a],
                    ecolor=MUTED, capsize=3, ms=6)
    ax.set_yticks(np.arange(len(arms)))
    ax.set_yticklabels(arms)
    ax.invert_yaxis()
    ax.set_xlabel("mean NLL of the curated GT CoC")
    ax.set_title("can the arm predict correct reasoning? (gate C2)")

    ax = axes[2]
    for j, a in enumerate(arms):
        d = np.array([r["minADE_rollout"] - r["minADE_tf"] for r in O[a]])
        m, lo, hi = el.paired_bootstrap_ci(d)
        ax.errorbar([m], [j], xerr=[[m - lo], [hi - m]], fmt="o", color=colors[a],
                    ecolor=MUTED, capsize=3, ms=6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(np.arange(len(arms)))
    ax.set_yticklabels(arms)
    ax.invert_yaxis()
    ax.set_xlabel("minADE(own CoC) - minADE(GT CoC), m")
    ax.set_title("what the arm's own reasoning costs it")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "ood_decomposition.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["indist", "test", "ood"])
    ap.add_argument("--k", type=int, default=None,
                    help="re-reduce minADE/minFDE to the first K samples (needs per-sample arrays)")
    ap.add_argument("--out", default="arms_summary")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows_by_set, res, lines = {}, {"paired": {}}, []
    for s in args.sets:
        rows_by_set[s] = {}
        for arm, (prefix, tag) in ARMS.items():
            rows = load_rows(REPO / "outputs" / f"{prefix}{s}", tag)
            if rows:
                rows_by_set[s][arm] = reduce_to_k(rows, args.k, arm, s) if args.k else rows
        if not rows_by_set[s]:
            del rows_by_set[s]
    sets = [s for s in args.sets if s in rows_by_set]

    for s in sets:
        res[s] = {a: arm_stats(r) for a, r in rows_by_set[s].items()}
        lines.append(f"=== {s} ===")
        lines.append(f"{'arm':10s} {'n':>5} {'minADE med':>11} {'mean':>8} "
                     f"{'NLL(self)':>10} {'CoC degen':>10}")
        for a in ARM_ORDER:
            if a not in res[s]:
                continue
            r = res[s][a]
            lines.append(f"{a:10s} {r['n']:5d} {r['minADE_rollout']['median']:11.4f} "
                         f"{r['minADE_rollout']['mean']:8.4f} "
                         f"{r['nll_self']['mean']:10.4f} {r['coc_degen']:10.4f}")

        res["paired"][s] = {}
        metrics = ["minADE_rollout", "nll_self"]
        if "minADE_tf" in rows_by_set[s][next(iter(rows_by_set[s]))][0]:
            metrics += ["minADE_tf", "nll_gtcoc"]
        for pair, a, b in CONTRASTS:
            if a not in rows_by_set[s] or b not in rows_by_set[s]:
                continue
            res["paired"][s][pair] = {m: paired(rows_by_set[s][a], rows_by_set[s][b], m)
                                      for m in metrics}
            lines.append(f"  [{pair.replace('_vs_', ' - ')}]")
            for m in metrics:
                lines.append(f"    {m:16s} {d(res['paired'][s][pair][m], '')}")
        lines.append("")

    # C1 pools val and test: same protocol, same model, twice the clips
    indist_sets = [s for s in sets if s in ("indist", "test")]
    c1 = pooled(rows_by_set, indist_sets, "dual", "jtraj", "minADE_rollout")
    c2 = (paired(rows_by_set["ood"]["dual"], rows_by_set["ood"]["jtraj"], "nll_gtcoc")
          if "ood" in rows_by_set and {"dual", "jtraj"} <= set(rows_by_set["ood"]) else None)
    degen = {a: max((res[s][a]["coc_degen"] for s in sets if a in res[s]), default=0.0)
             for a in ARM_ORDER if any(a in res[s] for s in sets)}
    c3 = all(degen.get(a, 0.0) < GATE_C3_DEGEN for a in ("dual", "jtraj"))
    # G1/G2, pre-registered in plans/2026-08-12_single-criterion-arms.md, read the same
    # pooled in-dist minADE as C1 -- the single-criterion arms exist to say what each
    # half of max(rank I_traj, rank X) contributes on its own.
    # `pooled` already returns None when an arm is absent from every pooled set, so the
    # membership test only needs to survive --sets ood (no in-dist set at all)
    g1 = {b: pooled(rows_by_set, indist_sets, "traj", b, "minADE_rollout")
          for b in ("dual", "jtraj")} if indist_sets else {}
    g1 = {b: v for b, v in g1.items() if v}
    g2 = pooled(rows_by_set, indist_sets, "coc", "j", "minADE_rollout")
    res["gates"] = {
        "C1": {"pooled_sets": indist_sets, "delta": c1,
               "pass": bool(c1 and abs(c1["median"]) < GATE_C1_M and c1["p"] > 0.05)},
        "C2": {"delta": c2},
        "C3": {"max_degen": degen, "threshold": GATE_C3_DEGEN, "pass": bool(c3)},
        "G1": {"pooled_sets": indist_sets, "delta": g1,
               "pass": {b: bool(v and v["ci"][1] < 0) for b, v in g1.items()}},
        "G2": {"pooled_sets": indist_sets, "delta": g2,
               "pass": bool(g2 and g2["ci"][0] <= 0 <= g2["ci"][1])},
    }
    c1_verdict = ("PASS: the J-lens substitutes for the CoC labels"
                  if res["gates"]["C1"]["pass"] else "FAIL: the criteria differ")
    lines += ["=== gates ===",
              f"C1 (primary) pooled {'+'.join(indist_sets)} minADE jtraj - dual",
              f"     {d(c1)}",
              f"     |median| < {GATE_C1_M} and p > 0.05 -> {c1_verdict}",
              "C2 OOD nll_gtcoc jtraj - dual",
              f"     {d(c2, '')}",
              ("C3 max CoC degeneracy  "
               + "  ".join(f"{a} {v:.4f}" for a, v in degen.items())
               + f"  (dual/jtraj < {GATE_C3_DEGEN}) -> {'PASS' if c3 else 'FAIL'}")]
    if g1:
        lines.append(f"G1 combined vs traj-only, pooled {'+'.join(indist_sets)} minADE")
        for b, v in g1.items():
            lines.append(f"     {b} - traj  {d(v)}")
        lines.append("     CI entirely < 0 -> the reasoning half buys something")
    if g2:
        verdict = "PASS" if res["gates"]["G2"]["pass"] else "FAIL"
        lines += ["G2 j - coc (single criterion, no traj half), pooled minADE",
                  f"     {d(g2)}",
                  f"     CI contains 0 -> J substitutes for CoC labels alone too ({verdict})"]

    if sets:
        plots(res, rows_by_set, sets, out_dir)
    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "one-factor criterion test: CoC NLL Taylor vs J-lens, all else held",
        "arms": {a: {"exp_prefix": p, "rows_tag": t} for a, (p, t) in ARMS.items()},
        "sets": sets, "k": args.k, "gate_c1_m": GATE_C1_M, "gate_c3_degen": GATE_C3_DEGEN,
        "n_per_set": {s: {a: len(r) for a, r in rows_by_set[s].items()} for s in sets},
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
