"""Attribution analysis for the criterion x allocation grid (run_grid.py).

The question is not "which config is best" but "which factor did the work". With budget held
equal across all eight cells, the grid decomposes into two main effects:

  criterion effect  = mean over allocations of (traj - dual)
  allocation effect = spread over allocations within a criterion
  traj x agree      -- if it holds up, the layerwise budget is the mechanism, not the criterion
  traj x depthprior -- if it also holds up, no CoC information is needed at all, only a
                       depth tilt (this is the CoC-free control)

Endpoints: CoC degeneracy rate (reasoning channel, free-running), teacher-forced CoC NLL,
minADE/minFDE (action channel). All comparisons are clip-paired against baseline, since every
config saw the same clips, the same reference CoC and the same denoising seeds.

Usage: python analyze_grid.py --exp-id grid_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import eval_lib as el

REPO = Path(__file__).resolve().parents[2]

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

CRITS = ["traj", "dual"]
ALLOCS = ["uniform", "late", "agree", "depthprior"]
KEYS = ("degenerate", "degenerate_strict", "nll", "ade", "fde")


def paired(per_clip, name, base, key):
    a = np.asarray(per_clip[name][key], dtype=float)
    b = np.asarray(per_clip[base][key], dtype=float)
    n = min(len(a), len(b))
    return a[:n] - b[:n]


def stat_row(per_clip, name, base, key):
    """Paired stats vs baseline.

    minADE deltas are heavy-tailed -- a few clips where the baseline itself is poor
    (baseline max 6.1 m) dominate the mean and can even flip its sign, so the median and
    the win/loss count are the honest summary and Wilcoxon is the matching test.
    """
    d = paired(per_clip, name, base, key)
    mean, lo, hi = el.paired_bootstrap_ci(d)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"delta": mean, "median": float(np.median(d)), "ci": [lo, hi], "p": p,
            "worse": int((d > 0).sum()), "better": int((d < 0).sum()),
            "abs": float(np.mean(np.asarray(per_clip[name][key], dtype=float)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    args = ap.parse_args()

    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    met = json.loads((out / "metrics.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    pc, meta = met["per_clip"], met["meta"]
    n = met["n_clips"]

    cells = [f"{c}_{a}" for c in CRITS for a in ALLOCS if f"{c}_{a}" in pc]
    res = {name: {k: stat_row(pc, name, "baseline", k)
                  for k in KEYS} for name in cells}
    base_abs = {k: float(np.mean(np.asarray(pc["baseline"][k], dtype=float)))
                for k in KEYS}

    # main effects on the reasoning channel, budget held equal
    degen = {name: res[name]["degenerate_strict"]["abs"] for name in cells}
    crit_effect = float(np.mean([degen[f"traj_{a}"] - degen[f"dual_{a}"] for a in ALLOCS
                                 if f"traj_{a}" in degen and f"dual_{a}" in degen]))
    alloc_effect = float(np.mean([max(degen[f"{c}_{a}"] for a in ALLOCS)
                                  - min(degen[f"{c}_{a}"] for a in ALLOCS) for c in CRITS]))

    metrics = {"n_clips": n, "baseline_abs": base_abs, "cells": res,
               "criterion_effect_degen": crit_effect, "allocation_effect_degen": alloc_effect,
               "removed": {name: meta[name]["removed_qo_mlp"] for name in cells}}
    (out / "grid_analysis.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        (f"Criterion x allocation attribution -- {args.exp_id}, n={n} clips, "
         f"K={cfg['k_samples']}, split={cfg['eval_split']}"),
        (f"  budget held equal: {min(metrics['removed'].values()):.3f}-"
         f"{max(metrics['removed'].values()):.3f} of VLM q/o+MLP params removed"),
        f"  held fixed: {cfg['held_fixed']}",
        "",
        (f"baseline: CoC degen {base_abs['degenerate']:.3f}  NLL {base_abs['nll']:.3f}  "
         f"minADE {base_abs['ade']:.3f}"),
        "",
        "CoC degeneracy rate (free-running, strict) -- absolute, and paired delta vs baseline",
        f"  {'allocation':12s}" + "".join(f"{c:>26s}" for c in CRITS),
    ]
    for a in ALLOCS:
        row = f"  {a:12s}"
        for c in CRITS:
            name = f"{c}_{a}"
            if name not in res:
                row += f"{'--':>26s}"
                continue
            r = res[name]["degenerate_strict"]
            cell = "{:.3f} (d{:+.3f}, p={:.3f})".format(r["abs"], r["delta"], r["p"])
            row += f"{cell:>26s}"
        lines.append(row)
    lines += [
        "",
        f"  criterion main effect  (traj - dual, mean over allocations): {crit_effect:+.3f}",
        f"  allocation main effect (max - min within a criterion)      : {alloc_effect:+.3f}",
        "",
        "  loose-threshold rates (directly comparable to the closed-loop degeneracy numbers):",
        "    " + "  ".join(f"{name}={res[name]['degenerate']['abs']:.3f}" for name in cells),
        "",
        "All endpoints (paired delta vs baseline, 95% CI, Wilcoxon p)",
        (f"  {'config':16s}{'removed':>9s}{'CoC degen(strict)':>33s}{'CoC NLL':>33s}"
         f"{'minADE (median, worse/better)':>36s}"),
    ]
    for name in cells:
        r = res[name]
        lines.append(f"  {name:16s}{metrics['removed'][name]:>9.3f}"
                     f"{fmt(r['degenerate_strict']):>33s}{fmt(r['nll']):>33s}"
                     f"{fmt_med(r['ade']):>36s}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    x = np.arange(len(ALLOCS))
    w = 0.38
    panels = [(axes[0], "degenerate_strict", "abs", "CoC degeneracy rate (free-running)"),
              (axes[1], "nll", "abs", "CoC NLL (teacher-forced)"),
              # the ADE mean is outlier-driven, so the action channel is shown as a median delta
              (axes[2], "ade", "median", r"minADE median $\Delta$ vs baseline (m)")]
    for ax, key, stat, label in panels:
        for i, (c, col) in enumerate(zip(CRITS, [C3, C1])):
            vals = [res[f"{c}_{a}"][key][stat] if f"{c}_{a}" in res else np.nan for a in ALLOCS]
            ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=c)
        ref = base_abs[key] if stat == "abs" else 0.0
        ax.axhline(ref, color=MUTED, lw=1.2, ls="--", label="baseline")
        ax.set_xticks(x); ax.set_xticklabels(ALLOCS)
        ax.set_xlabel("layerwise allocation"); ax.set_title(label)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "plots" / "grid_attribution.png", dpi=150)
    print(f"\nwrote {out}")


def fmt(r):
    return f"{r['delta']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.3f}"


def fmt_med(r):
    return f"{r['median']:+.3f} ({r['worse']}/{r['better']}) p={r['p']:.3f}"


if __name__ == "__main__":
    main()
