"""Merge the open-loop shards and report the baseline on both evaluation sets.

Reads whatever `run_baseline.py` wrote (any shard layout — shards are merged by clip
id, so a 4-way run and a 1-way run aggregate identically), and reports:

  per set        minADE / minFDE, mean and median with a bootstrap CI, plus CoC
                 degeneracy and generated-CoC length
  by bucket      decel_stop / turn / accel / cruise, derived from the GT path geometry
  OOD only       the two conditions side by side. minADE_tf uses the curated GT CoC as
                 context, minADE_rollout the model's own reasoning; they run on the same
                 clips, so the difference is paired and gets a Wilcoxon test. This is the
                 measurement that says how much correct reasoning is worth in metres.
  OOD only       breakdown by event cluster and by official split (val is primary,
                 train secondary -- we never train, so both are zero-shot, but a reader
                 may want val alone).

Median is reported alongside the mean because minADE is heavy-tailed: one broken clip
lands at 25 m and drags a mean of a few hundred clips visibly.

With `--compare <tag>` a second model's rows are loaded and every metric is also
reported as a paired per-clip delta, which is how pruned configs will be read against
this baseline.

Usage:
  python experiments/evaluation/analyze_baseline.py
  python experiments/evaluation/analyze_baseline.py --compare slim_dual_uniform
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

import eval_lib as el

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

BUCKETS = ["decel_stop", "turn", "accel", "cruise"]


def load_rows(exp_dir, tag):
    """Merge every shard file for one model; last write per clip wins."""
    rows = {}
    for p in sorted(exp_dir.glob(f"{tag}_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return list(rows.values())


def describe(vals):
    """mean / median / bootstrap CI of the mean, for one metric."""
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return None
    mean, lo, hi = el.paired_bootstrap_ci(a)
    return {"n": len(a), "mean": mean, "ci": [lo, hi],
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def by_group(rows, key, metric):
    out = {}
    for g in sorted({r[key] for r in rows if key in r}):
        vals = [r[metric] for r in rows if r.get(key) == g]
        out[g] = describe(vals)
    return out


def fmt(d, unit="m"):
    if d is None:
        return "n/a"
    return (f"{d['mean']:.3f}{unit} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}] "
            f"med {d['median']:.3f} p90 {d['p90']:.3f} (n={d['n']})")


def has_gt_coc(rows):
    """OOD rows carry the GT-CoC condition; in-distribution rows cannot."""
    return bool(rows) and "minADE_tf" in rows[0]


def analyse(rows):
    """All summary statistics for one set."""
    out = {"n": len(rows)}
    for m in ("minADE_rollout", "minFDE_rollout", "nll_self"):
        out[m] = describe([r[m] for r in rows])
    out["coc"] = {
        "degenerate": float(np.mean([r["coc_degenerate"] for r in rows])),
        "degenerate_strict": float(np.mean([r["coc_degenerate_strict"] for r in rows])),
        "empty": float(np.mean([r["coc_empty"] for r in rows])),
        "len_median": float(np.median([r["coc_len"] for r in rows])),
        "gen_tokens_median": float(np.median([r["gen_len"] for r in rows])),
    }
    out["by_bucket"] = by_group(rows, "bucket", "minADE_rollout")
    out["bucket_counts"] = {b: sum(r["bucket"] == b for r in rows) for b in BUCKETS}

    if has_gt_coc(rows):
        for m in ("minADE_tf", "minFDE_tf", "nll_gtcoc"):
            out[m] = describe([r[m] for r in rows])
        # same clips under both conditions -> paired
        d = np.array([r["minADE_rollout"] - r["minADE_tf"] for r in rows])
        mean, lo, hi = el.paired_bootstrap_ci(d)
        out["rollout_minus_tf"] = {
            "mean": mean, "ci": [lo, hi], "median": float(np.median(d)),
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "frac_tf_better": float(np.mean(d > 0)),
        }
        out["by_cluster"] = by_group(rows, "cluster", "minADE_rollout")
        out["by_cluster_tf"] = by_group(rows, "cluster", "minADE_tf")
        out["by_cluster_nll"] = by_group(rows, "cluster", "nll_gtcoc")
        out["by_split"] = by_group(rows, "split", "minADE_rollout")
        out["by_split_nll"] = by_group(rows, "split", "nll_gtcoc")
    return out


def paired_delta(base_rows, cmp_rows, metric):
    """Per-clip delta cmp - base over the clips both evaluated."""
    b = {r["clip_id"]: r for r in base_rows}
    c = {r["clip_id"]: r for r in cmp_rows}
    ids = sorted(set(b) & set(c))
    d = np.array([c[i][metric] - b[i][metric] for i in ids])
    if len(d) == 0:
        return None
    mean, lo, hi = el.paired_bootstrap_ci(d)
    return {"n": len(ids), "mean": mean, "ci": [lo, hi], "median": float(np.median(d)),
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def plot_sets(res, rows_by_set, out_dir):
    sets = list(rows_by_set)
    ood = next((s for s in sets if "rollout_minus_tf" in res[s]), None)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]  # minADE distribution, log x because of the tail
    for s, c in zip(sets, (C1, C4, C2, C3)):
        a = np.array([r["minADE_rollout"] for r in rows_by_set[s]])
        ax.hist(np.clip(a, 1e-2, None), bins=np.logspace(-2, 1.5, 40), alpha=0.55,
                color=c, label=f"{s} (n={len(a)}, med {np.median(a):.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("minADE (m), own-rollout context")
    ax.set_ylabel("clips")
    ax.set_title("trajectory error distribution")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]  # per bucket
    w = 0.38
    x = np.arange(len(BUCKETS))
    for j, (s, c) in enumerate(zip(sets, (C1, C4, C2, C3))):
        med = [res[s]["by_bucket"].get(b, {}).get("median", np.nan) for b in BUCKETS]
        ax.bar(x + (j - 0.5) * w, med, w, color=c, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS, rotation=20)
    ax.set_ylabel("median minADE (m)")
    ax.set_title("by scenario bucket")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]  # OOD: does correct reasoning help?
    if ood:
        r = rows_by_set[ood]
        tf = np.array([x["minADE_tf"] for x in r])
        ro = np.array([x["minADE_rollout"] for x in r])
        ax.scatter(tf, ro, s=8, alpha=0.35, color=C2)
        lim = [1e-2, max(tf.max(), ro.max()) * 1.1]
        ax.plot(lim, lim, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("minADE | GT CoC context (m)")
        ax.set_ylabel("minADE | own rollout (m)")
        ax.set_title(f"OOD: correct reasoning helps on "
                     f"{res[ood]['rollout_minus_tf']['frac_tf_better'] * 100:.0f}% of clips")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "baseline.png", dpi=150)
    plt.close(fig)

    if not ood:
        return
    R = res[ood]
    clusters = sorted(R["by_cluster"], key=lambda k: -R["by_cluster"][k]["n"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    y = np.arange(len(clusters))
    ax = axes[0]
    ax.barh(y - 0.2, [R["by_cluster_tf"][c]["median"] for c in clusters], 0.4,
            color=C2, label="GT CoC context")
    ax.barh(y + 0.2, [R["by_cluster"][c]["median"] for c in clusters], 0.4,
            color=C4, label="own rollout")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c[:26]} ({R['by_cluster'][c]['n']})" for c in clusters],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median minADE (m)")
    ax.set_title("OOD by event cluster")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.barh(y, [R["by_cluster_nll"][c]["mean"] for c in clusters], 0.6, color=C1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("mean NLL of the curated CoC")
    ax.set_title("can the model predict the reference reasoning?")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "ood_clusters.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--compare", default=None, help="second model tag for paired deltas")
    ap.add_argument("--sets", nargs="+", default=["indist", "ood"])
    ap.add_argument("--exp-prefix", default="baseline_")
    ap.add_argument("--out", default="baseline_summary")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    res, rows_by_set, lines = {}, {}, []
    for which in args.sets:
        exp_dir = REPO / "outputs" / f"{args.exp_prefix}{which}"
        rows = load_rows(exp_dir, args.tag)
        if not rows:
            print(f"[skip] no rows for {which} in {exp_dir}", flush=True)
            continue
        rows_by_set[which] = rows
        res[which] = analyse(rows)
        cfg = json.loads((exp_dir / "config.json").read_text())
        res[which]["gpu"] = cfg.get("gpu")
        res[which]["k"] = cfg.get("k")

        r = res[which]
        lines += [f"=== {which}  n={r['n']}  (k={r['k']} samples, {r['gpu']}) ===",
                  f"  minADE  own rollout   {fmt(r['minADE_rollout'])}",
                  f"  minFDE  own rollout   {fmt(r['minFDE_rollout'])}"]
        if has_gt_coc(rows):
            lines += [f"  minADE  GT CoC ctx    {fmt(r['minADE_tf'])}",
                      f"  NLL     GT CoC        {fmt(r['nll_gtcoc'], '')}"]
            d = r["rollout_minus_tf"]
            lines.append(f"  rollout - GT CoC      mean {d['mean']:+.3f} m "
                         f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] med {d['median']:+.3f} "
                         f"p={d['p']:.2e}  GT better on {d['frac_tf_better'] * 100:.0f}%")
        lines.append(f"  NLL     own CoC       {fmt(r['nll_self'], '')}")
        lines.append(f"  CoC     degen {r['coc']['degenerate']:.4f} "
                     f"(strict {r['coc']['degenerate_strict']:.4f}) "
                     f"len med {r['coc']['len_median']:.0f} chars")
        lines.append("  by bucket (median minADE, own rollout):")
        for b in BUCKETS:
            g = r["by_bucket"].get(b)
            if g:
                lines.append(f"    {b:11s} n={g['n']:4d}  med {g['median']:.3f}  "
                             f"mean {g['mean']:.3f}")
        if has_gt_coc(rows):
            lines.append("  by split:")
            for s, g in r["by_split"].items():
                lines.append(f"    {s:6s} n={g['n']:5d}  med {g['median']:.3f}  "
                             f"NLL {r['by_split_nll'][s]['mean']:.3f}")
            lines.append("  by cluster (median minADE rollout | GT CoC | NLL):")
            for c in sorted(r["by_cluster"], key=lambda k: -r["by_cluster"][k]["n"]):
                lines.append(f"    {c[:38]:38s} n={r['by_cluster'][c]['n']:4d}  "
                             f"{r['by_cluster'][c]['median']:.3f} | "
                             f"{r['by_cluster_tf'][c]['median']:.3f} | "
                             f"{r['by_cluster_nll'][c]['mean']:.3f}")
        lines.append("")

    if args.compare:
        lines.append(f"=== paired delta: {args.compare} - {args.tag} ===")
        res["paired"] = {}
        for which, base_rows in rows_by_set.items():
            cmp_rows = load_rows(REPO / "outputs" / f"{args.exp_prefix}{which}", args.compare)
            if not cmp_rows:
                continue
            metrics = ["minADE_rollout", "nll_self"]
            if has_gt_coc(base_rows):
                metrics += ["minADE_tf", "nll_gtcoc"]
            res["paired"][which] = {m: paired_delta(base_rows, cmp_rows, m)
                                    for m in metrics}
            for m, d in res["paired"][which].items():
                if d:
                    lines.append(f"  {which:7s} {m:16s} mean {d['mean']:+.4f} "
                                 f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] "
                                 f"med {d['median']:+.4f} p={d['p']:.2e} (n={d['n']})")
        lines.append("")

    if rows_by_set:
        plot_sets(res, rows_by_set, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
