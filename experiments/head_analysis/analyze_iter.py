"""Judge the staged-recalibration gates for one criterion, at minADE@K.

Reads three test_500 row sets -- the re-measured baseline, the one-shot arm, and the
staged (it3) arm -- reduces every clip to minADE@K from the stored per-sample arrays
(seeds are base+k, so the first K of an 8-sample run are exactly what a K-sample run
would have drawn), pairs by clip id, and judges the gates pre-registered in
plans/2026-08-16_iterative-recalibration.md:

  R1a  direction: paired minADE@K (it3 - oneshot), bootstrap 95% CI upper < 0.
       The median CI is primary (minADE deltas are heavy-tailed; house convention),
       the mean CI is reported beside it.
  R1b  size: median(it3 - oneshot) <= -0.3 x median(oneshot - baseline).
  R2   CoC degeneracy of the it3 arm < 0.05, reported next to the one-shot rate.
  R3   kept-set overlap vs the one-shot cut (recorded by run_iter_prune): > 0.95
       reroutes a null R1 to "re-scoring did not change the selection".

Also checks that the re-measured baseline reproduces the original baseline_ada rows
bitwise (same clips, same seeds, same architecture) -- a free integrity check.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_iter.py --criterion dual
"""

import argparse
import glob
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

GATE_R1B_FRAC = 0.3
GATE_R2_DEGEN = 0.05
GATE_R3_OVERLAP = 0.95


def load_rows(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        rows.extend(json.loads(Path(f).read_text()))
    return {r["clip_id"]: r for r in rows}


def ade_at_k(row, k):
    if "ade_rollout_k" not in row:
        raise SystemExit(f"clip {row['clip_id']}: no per-sample array; "
                         "re-evaluate with the post-2026-08-11 runner")
    return float(np.min(np.asarray(row["ade_rollout_k"], dtype=float)[:k]))


def boot(v, fn, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    stats = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


SET_IDS = {"test": ("{c}_u40_v2_ps_test", "iter_{c}_test"),
           "indist": ("{c}_u40_v2_ps_indist", "iter_{c}_indist"),
           "ood": ("{c}_u40_v2_ps_ood", "iter_{c}_ood")}


def metric_at_k(row, key, k):
    if key not in row:
        raise SystemExit(f"clip {row['clip_id']}: no {key}")
    return float(np.min(np.asarray(row[key], dtype=float)[:k]))


HORIZONS = (16, 32, 64)  # 1.6 s / 3.2 s / 6.4 s at 10 Hz


def hkey(base, h):
    return base if h == 64 else f"{base}_h{h}"


def cross_sets(c, k, sets, out):
    """One-shot vs it3 across sets and horizons: minADE@k and minFDE@k, paired."""
    res, tab = {}, []
    for s in sets:
        o_id, t_id = (x.format(c=c) for x in SET_IDS[s])
        one = load_rows(REPO / "outputs" / o_id / f"slim_{c}_u40_v2_s*.json")
        it3 = load_rows(REPO / "outputs" / t_id / f"slim_{c}_u40_it3_s*.json")
        ids = sorted(set(one) & set(it3))
        cell = {"n": len(ids),
                "degen_one": float(np.mean([one[i]["coc_degenerate"] for i in ids])),
                "degen_it3": float(np.mean([it3[i]["coc_degenerate"] for i in ids]))}
        for h in HORIZONS:
            hs = f"{h / 10:.1f}s"
            cell[hs] = {}
            for m, base in (("ade", "ade_rollout_k"), ("fde", "fde_rollout_k")):
                key = hkey(base, h)
                ov = np.array([metric_at_k(one[i], key, k) for i in ids])
                tv = np.array([metric_at_k(it3[i], key, k) for i in ids])
                d = tv - ov
                lo, hi = boot(d, np.median)
                cell[hs][m] = {
                    "one_mean": float(ov.mean()), "one_med": float(np.median(ov)),
                    "it3_mean": float(tv.mean()), "it3_med": float(np.median(tv)),
                    "d_med": float(np.median(d)), "d_med_ci": [lo, hi],
                    "d_mean": float(np.mean(d)),
                    "wilcoxon_p": float(wilcoxon(d).pvalue)}
        res[s] = cell
        a = cell["6.4s"]["ade"]
        tab.append(f"{s:7s} n={cell['n']:4d}  ADE@{k} 6.4s one {a['one_mean']:.4f} -> "
                   f"it3 {a['it3_mean']:.4f}  d_med {a['d_med']:+.4f} "
                   f"[{a['d_med_ci'][0]:+.4f},{a['d_med_ci'][1]:+.4f}] "
                   f"p={a['wilcoxon_p']:.1e}")
    (out / "sets_summary.json").write_text(json.dumps(res, indent=2))

    fig, axes = plt.subplots(2, len(sets), figsize=(3.4 * len(sets), 6.6), squeeze=False)
    width, xs = 0.35, np.arange(len(HORIZONS))
    hlabels = [f"{h / 10:.1f}s" for h in HORIZONS]
    for row, m in enumerate(("ade", "fde")):
        for col, s in enumerate(sets):
            ax = axes[row][col]
            ov = [res[s][hl][m]["one_mean"] for hl in hlabels]
            tv = [res[s][hl][m]["it3_mean"] for hl in hlabels]
            ax.bar(xs - width / 2, ov, width, color=C1, label="one-shot")
            ax.bar(xs + width / 2, tv, width, color=C4, label="it3 (staged)")
            for x, hl in enumerate(hlabels):
                d = res[s][hl][m]
                star = "*" if d["d_med_ci"][0] > 0 or d["d_med_ci"][1] < 0 else ""
                ax.annotate(f"{d['d_med']:+.3f}{star}", (x, max(ov[x], tv[x])),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=7.5, color=INK)
            ax.set_xticks(xs, hlabels)
            if row == 0:
                ax.set_title(f"{s} (n={res[s]['n']})")
            if col == 0:
                ax.set_ylabel(f"min{m.upper()}@{k} mean (m)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle("one-shot vs staged by set and horizon "
                 "(paired median delta annotated, * = CI excludes 0)", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "plots" / "sets.png", dpi=150)
    plt.close(fig)
    print("\n".join(tab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="dual")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--baseline", default="baseline_ada_ps_test")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="cross-set one-shot vs it3 summary (e.g. test indist ood) "
                         "instead of the gate judgment")
    args = ap.parse_args()
    c, k = args.criterion, args.k
    if args.sets:
        out = REPO / "outputs" / f"iter_gates_{c}"
        (out / "plots").mkdir(parents=True, exist_ok=True)
        cross_sets(c, k, args.sets, out)
        return

    base = load_rows(REPO / "outputs" / args.baseline / "baseline_s*.json")
    one = load_rows(REPO / "outputs" / f"{c}_u40_v2_ps_test" / f"slim_{c}_u40_v2_s*.json")
    if not one:  # arms whose 08-12 rows already carry per-sample arrays
        one = load_rows(REPO / "outputs" / f"{c}_u40_v2_test" / f"slim_{c}_u40_v2_s*.json")
    it3 = load_rows(REPO / "outputs" / f"iter_{c}_test" / f"slim_{c}_u40_it3_s*.json")
    ids = sorted(set(base) & set(one) & set(it3))
    if len(ids) < 400:
        raise SystemExit(f"only {len(ids)} overlapping clips; something is missing")

    b = np.array([ade_at_k(base[i], k) for i in ids])
    o = np.array([ade_at_k(one[i], k) for i in ids])
    t = np.array([ade_at_k(it3[i], k) for i in ids])
    d_one, d_it3, contrast = o - b, t - b, t - o

    med = float(np.median(contrast))
    med_lo, med_hi = boot(contrast, np.median)
    mean = float(np.mean(contrast))
    mean_lo, mean_hi = boot(contrast, np.mean)
    p_w = float(wilcoxon(contrast).pvalue)

    r1a = med_hi < 0
    r1b_thresh = -GATE_R1B_FRAC * float(np.median(d_one))
    r1b = med <= r1b_thresh
    degen_it3 = float(np.mean([it3[i]["coc_degenerate"] for i in ids]))
    degen_one = float(np.mean([one[i]["coc_degenerate"] for i in ids]))
    r2 = degen_it3 < GATE_R2_DEGEN
    ov = json.loads((REPO / "outputs" / f"iter_{c}_u40" / "config.json").read_text())[
        "kept_overlap_vs_oneshot"]
    r3_unchanged = ov["q"] > GATE_R3_OVERLAP and ov["mlp"] > GATE_R3_OVERLAP

    # integrity: the re-measured baseline must reproduce the original rows bitwise
    orig = load_rows(REPO / "outputs" / "baseline_ada_test" / "baseline_s*.json")
    shared = [i for i in ids if i in orig]
    repro = float(np.mean([base[i]["minADE_rollout"] == orig[i]["minADE_rollout"]
                           for i in shared])) if shared else float("nan")

    nll_d = np.array([it3[i]["nll_self"] - one[i]["nll_self"] for i in ids])

    lines = [
        f"staged recalibration gates -- {c}, minADE@{k}, n={len(ids)}",
        (f"baseline {np.mean(b):.4f} (med {np.median(b):.4f}) | "
         f"oneshot {np.mean(o):.4f} ({np.median(o):.4f}) | "
         f"it3 {np.mean(t):.4f} ({np.median(t):.4f})"),
        f"oneshot - baseline: med {np.median(d_one):+.4f}  mean {np.mean(d_one):+.4f}",
        f"it3     - baseline: med {np.median(d_it3):+.4f}  mean {np.mean(d_it3):+.4f}",
        "",
        (f"contrast it3 - oneshot: med {med:+.4f} [{med_lo:+.4f}, {med_hi:+.4f}]  "
         f"mean {mean:+.4f} [{mean_lo:+.4f}, {mean_hi:+.4f}]  wilcoxon p={p_w:.2e}"),
        f"R1a (median CI upper < 0)          -> {'PASS' if r1a else 'FAIL'}",
        f"R1b (med <= {r1b_thresh:+.4f} = 30% recovery) -> {'PASS' if r1b else 'FAIL'}",
        (f"R2  (it3 degeneracy {degen_it3:.3f} < 0.05; oneshot {degen_one:.3f}) "
         f"-> {'PASS' if r2 else 'FAIL'}"),
        f"R3  overlap q {ov['q']:.3f} mlp {ov['mlp']:.3f} -> "
        + ("selection unchanged: a null R1 is uninformative" if r3_unchanged
           else "selection changed: R1 reads at face value"),
        f"nll_self it3 - oneshot: med {np.median(nll_d):+.4f}",
        (f"baseline reproduction vs original rows: {repro:.1%} bitwise "
         f"({len(shared)} shared clips)"),
    ]
    out = REPO / "outputs" / f"iter_gates_{c}"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(9.6, 4.0), width_ratios=[1, 1.2])
    for x, (label, v, colour) in enumerate((("one-shot", d_one, C1),
                                            ("it3 (staged)", d_it3, C4))):
        m = float(np.median(v))
        lo, hi = boot(v, np.median)
        ax.bar(x, m, 0.55, color=colour, label=label)
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                    capsize=3, lw=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([0, 1], ["one-shot", "it3 (staged)"])
    ax.set_ylabel(f"paired dminADE@{k} vs baseline, median (m)")
    ax.set_title("cost of the same 24.0% budget")
    v = np.clip(contrast, -0.3, 0.3)
    axh.hist(v, bins=48, range=(-0.3, 0.3), color=C4, alpha=0.75)
    axh.axvline(0, color=MUTED, lw=0.9, ls="--")
    axh.axvspan(med_lo, med_hi, color=INK, alpha=0.12)
    axh.axvline(med, color=INK, lw=1.4)
    axh.set_xlabel(f"per-clip minADE@{k}: it3 - oneshot (m, clipped to +-0.3)")
    axh.set_title(f"median {med:+.4f} [{med_lo:+.4f}, {med_hi:+.4f}], "
                  f"wilcoxon p={p_w:.1e}")
    fig.tight_layout()
    fig.savefig(out / "plots" / "gates.png", dpi=150)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps({
        "criterion": c, "k": k, "n": len(ids),
        "contrast_median": med, "contrast_median_ci": [med_lo, med_hi],
        "contrast_mean": mean, "contrast_mean_ci": [mean_lo, mean_hi],
        "wilcoxon_p": p_w, "oneshot_cost_median": float(np.median(d_one)),
        "it3_cost_median": float(np.median(d_it3)),
        "r1a": bool(r1a), "r1b": bool(r1b), "r1b_threshold": r1b_thresh,
        "r2": bool(r2), "degen_it3": degen_it3, "degen_oneshot": degen_one,
        "r3_overlap": ov, "baseline_repro": repro,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
