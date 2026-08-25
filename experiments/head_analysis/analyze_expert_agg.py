"""Gates G5a-G5e for the expert aggregation arms (run_expert_agg.py).

Every arm is the same set of gradients aggregated differently, so the comparison that
matters is arm vs `sum` (the shipped criterion), paired on clip and seed. `magnitude` is
the reference the whole track is about: it is what currently wins at r10-r25, and G5c asks
whether a better aggregation closes that gap.

  G5b  primary, r=0.25: does any arm beat `sum` on paired dminADE@6 with a CI below 0?
  G5c  does the winner close >=50% of the sum-vs-magnitude gap?
  G5d  negative control: `sumabs` was predicted to be a no-op (kept-overlap 0.925 vs sum)
  G5e  side records: r=0.40, dev ranking, kept-set overlaps, per-bucket

Usage:
  python experiments/head_analysis/analyze_expert_agg.py --exp-id stepagg_v1 \
      --out stepagg_analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def boot_ci(d, n_boot=10000, seed=0, stat=np.mean):
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    b = stat(d[rng.randint(0, len(d), size=(n_boot, len(d)))], axis=1)
    return float(stat(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load_shards(exp_dir):
    """Concatenate the per-shard rows. Shards are strided, so clips never repeat."""
    rows, meta, names = [], None, None
    for p in sorted(exp_dir.glob("metrics_s*of*.json")):
        m = json.loads(p.read_text())
        rows.extend(m["rows"])
        meta, names = m["meta"], m["configs"]
    seen = set()
    uniq = []
    for r in rows:
        if r["clip_id"] not in seen:
            seen.add(r["clip_id"])
            uniq.append(r)
    return uniq, meta, names


def reduce_k(rows, names, key, k):
    """min over the first k samples -> (n_clips,) per config. Seeds are base+i, so the
    first k of a K-sample run are exactly what a k-sample run would have drawn."""
    return {n: np.array([min(r["configs"][n][key][:k]) for r in rows]) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6, help="minADE@k; runs store 8 samples")
    ap.add_argument("--primary-ratio", type=float, default=0.25)
    args = ap.parse_args()

    exp_dir = REPO / "outputs" / args.exp_id
    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((exp_dir / "config.json").read_text())
    rows, meta, names = load_shards(exp_dir)

    ade = reduce_k(rows, names, "ade_k", args.k)
    fde = reduce_k(rows, names, "fde_k", args.k)
    dev = {n: np.array([np.mean(r["configs"][n]["dev_k"]) for r in rows]) for n in names}
    buckets = np.array([r["bucket"] for r in rows])

    ratios = sorted({meta[n]["ratio"] for n in names if meta[n]["kind"] == "arm"})
    arms = [a for a in cfg["arms"]]
    lines = [
        f"Expert aggregation arms -- {args.exp_id}",
        f"  n={len(rows)} clips, minADE@{args.k}, arms={len(arms)}, ratios={ratios}",
        (f"  baseline minADE@{args.k} {ade['baseline'].mean():.4f} "
         f"median {np.median(ade['baseline']):.4f}"),
        f"  integrity: {'; '.join(cfg.get('integrity', []))}",
        "",
    ]
    res = {"n_clips": len(rows), "k": args.k, "ratios": ratios,
           "baseline_minADE": float(ade["baseline"].mean()), "by_ratio": {}}

    for r in ratios:
        tag = f"r{int(round(r * 100))}"
        lines.append(f"=== ratio {r:.2f} =========================================")
        lines.append(f"  {'arm':16s} {'dADE vs base':>14s} {'95% CI':>20s} "
                     f"{'dADE vs sum':>13s} {'95% CI':>20s} {'p':>7s} {'dev':>7s}")
        entry = {}
        for a in arms:
            n = f"{a}_{tag}"
            db = ade[n] - ade["baseline"]
            m_b, lo_b, hi_b = boot_ci(db)
            ds = ade[n] - ade[f"sum_{tag}"]
            m_s, lo_s, hi_s = boot_ci(ds)
            p = float(wilcoxon(ds).pvalue) if np.any(ds != 0) else 1.0
            entry[a] = {
                "vs_baseline": {"mean": m_b, "lo": lo_b, "hi": hi_b,
                                "median": float(np.median(db))},
                "vs_sum": {"mean": m_s, "lo": lo_s, "hi": hi_s,
                           "median": float(np.median(ds)), "wilcoxon_p": p},
                "minADE": float(ade[n].mean()), "minFDE": float(fde[n].mean()),
                "dev": float(dev[n].mean()),
                "by_bucket": {b: float((ade[n] - ade["baseline"])[buckets == b].mean())
                              for b in sorted(set(buckets))},
            }
            lines.append(f"  {a:16s} {m_b:+14.4f} [{lo_b:+.4f},{hi_b:+.4f}] "
                         f"{m_s:+13.4f} [{lo_s:+.4f},{hi_s:+.4f}] {p:7.4f} {dev[n].mean():7.4f}")
        res["by_ratio"][tag] = entry
        lines.append("")

    # ---- gates ----
    tag = f"r{int(round(args.primary_ratio * 100))}"
    e = res["by_ratio"][tag]
    cand = [a for a in arms if a not in ("sum", "magnitude")]
    winners = [a for a in cand if e[a]["vs_sum"]["hi"] < 0]
    lines.append(f"G5b (primary, ratio {args.primary_ratio:.2f}): arms beating `sum` "
                 f"with the whole CI below 0")
    if winners:
        best = min(winners, key=lambda a: e[a]["vs_sum"]["mean"])
        lines.append(f"  {', '.join(winners)}   -> best = {best} "
                     f"({e[best]['vs_sum']['mean']:+.4f} "
                     f"[{e[best]['vs_sum']['lo']:+.4f},{e[best]['vs_sum']['hi']:+.4f}])")
        lines.append("  -> G5b ACCEPT (aggregation is a real lever)")
    else:
        best = min(cand, key=lambda a: e[a]["vs_sum"]["mean"])
        lines.append(f"  none. best point estimate {best} {e[best]['vs_sum']['mean']:+.4f} "
                     f"[{e[best]['vs_sum']['lo']:+.4f},{e[best]['vs_sum']['hi']:+.4f}]")
        lines.append("  -> G5b REJECT (no aggregation beats the shipped one; on this tower "
                     "magnitude stands as the honest criterion)")
    res["G5b"] = {"winners": winners, "best": best, "accept": bool(winners)}

    gap = e["sum"]["vs_baseline"]["mean"] - e["magnitude"]["vs_baseline"]["mean"]
    new_gap = e[best]["vs_baseline"]["mean"] - e["magnitude"]["vs_baseline"]["mean"]
    closed = 1.0 - (new_gap / gap) if abs(gap) > 1e-9 else float("nan")
    lines.append(f"G5c (gap to magnitude): sum-magnitude {gap:+.4f}, "
                 f"{best}-magnitude {new_gap:+.4f}  -> {closed * 100:.0f}% closed "
                 f"{'PASS' if closed >= 0.5 else 'FAIL'}")
    res["G5c"] = {"gap_sum_vs_mag": gap, "gap_best_vs_mag": new_gap, "fraction_closed": closed}

    sa = e["sumabs"]["vs_sum"]
    noop = sa["lo"] < 0 < sa["hi"]
    lines.append(f"G5d (negative control): sumabs vs sum {sa['mean']:+.4f} "
                 f"[{sa['lo']:+.4f},{sa['hi']:+.4f}]  -> "
                 f"{'as predicted, a no-op' if noop else 'NOT a no-op; G1 reading needs review'}")
    res["G5d"] = {"as_predicted": bool(noop), **sa}
    lines.append("")

    ov = cfg.get("kept_overlap_q", {}).get(tag, {})
    if ov:
        lines.append(f"kept-set overlap (Q heads, {tag}), vs sum:")
        lines.append("  " + "  ".join(f"{a}:{ov.get(f'sum|{a}', float('nan')):.3f}"
                                      for a in arms))
        lines.append("")

    # ---- plots ----
    for r in ratios:
        tag_r = f"r{int(round(r * 100))}"
        ee = res["by_ratio"][tag_r]
        order = sorted(arms, key=lambda a: ee[a]["vs_baseline"]["mean"])
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.6))
        y = np.arange(len(order))
        m = [ee[a]["vs_baseline"]["mean"] for a in order]
        lo = [m[i] - ee[a]["vs_baseline"]["lo"] for i, a in enumerate(order)]
        hi = [ee[a]["vs_baseline"]["hi"] - m[i] for i, a in enumerate(order)]
        colors = [C2 if a == "magnitude" else (C4 if a == "sum" else C1) for a in order]
        ax.barh(y, m, color=colors, height=0.62)
        ax.errorbar(m, y, xerr=[lo, hi], fmt="none", ecolor=MUTED, capsize=3, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel(f"paired dminADE@{args.k} vs unpruned (m)")
        ax.set_title(f"expert aggregation arms, ratio {r:.2f}  "
                     f"(orange = shipped, green = magnitude)")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"arms_{tag_r}.png", dpi=150)
        plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
