"""Is the damage from pruning and from quantization additive?

Storage multiplies by construction -- that is arithmetic. The question is whether the
quality costs add. Every arm here is evaluated on the same clips with the same
clip-derived seeds, so the interaction can be formed per clip rather than by comparing
group means:

    interaction_i = d(both)_i - d(prune)_i - d(quant)_i

where d(x)_i = minADE(x, clip i) - minADE(baseline, clip i). Bootstrapping that vector
over clips gives a CI: containing zero means the two compressions are independent, and
positive means the pruned model is more fragile under quantization than the unpruned one.

Additivity is a descriptive null, not a law -- minADE deltas have no reason to compose
linearly. It is the reference point that makes "interaction" definable.

Usage:
  python experiments/evaluation/analyze_composition.py --set test
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
from analyze_baseline import load_rows

FULL_GB = 11_078_526_194 * 2 / 1e9
SLIM = "slim_dual_u40_v2"
# label -> (exp-id stem, rows tag)
BASE = ("baseline_ada_{s}", "baseline")
PRUNE = ("dual_u40_v2_{s}", SLIM)
CELLS = [("w8", ("w8_all_{s}", "uniform_w8_all"),
          ("dual_u40_w8_{s}", f"{SLIM}_w8_all_dual_u40"), "w8_all_dual_u40"),
         ("w4", ("w4_all_{s}", "uniform_w4_all"),
          ("dual_u40_w4_{s}", f"{SLIM}_w4_all_dual_u40"), "w4_all_dual_u40")]


def series(exp, tag, which):
    rows = load_rows(REPO / "outputs" / exp.format(s=which), tag)
    return {r["clip_id"]: r["minADE_rollout"] for r in rows}, rows


def ci(v):
    m, lo, hi = el.paired_bootstrap_ci(np.asarray(v, dtype=float))
    return m, lo, hi, ("" if lo <= 0 <= hi else "*")


def size_gb(spec_tag, slim_dir):
    params = json.loads((REPO / "outputs" / slim_dir / "config.json").read_text()
                        )["params"]["slim"] if slim_dir else 11_078_526_194
    if not spec_tag:
        return params * 2 / 1e9
    m = json.loads((REPO / "outputs" / "quant_specs" / f"{spec_tag}.json").read_text())
    return m["projected_pool_gb"] + (params - m["pool_params"]) * 2 / 1e9


BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
# label -> (exp stem, rows tag, spec tag or None, slim dir or None, colour, marker)
PLOT_ARMS = [
    ("baseline", "baseline_ada_{s}", "baseline", None, None, MUTED, "o"),
    ("w8 (+expert)", "w8_all_{s}", "uniform_w8_all", "uniform_w8_all", None, C1, "s"),
    ("w4 (+expert)", "w4_all_{s}", "uniform_w4_all", "uniform_w4_all", None, C1, "D"),
    ("prune 24%", "dual_u40_v2_{s}", SLIM, None, SLIM, C4, "^"),
    ("prune + w8", "dual_u40_w8_{s}", f"{SLIM}_w8_all_dual_u40", "w8_all_dual_u40", SLIM, C2, "P"),
    ("prune + w4", "dual_u40_w4_{s}", f"{SLIM}_w4_all_dual_u40", "w4_all_dual_u40", SLIM, C2, "X"),
]


def tradeoff_plot(which, out, path):
    """Projected checkpoint size against the quality it costs, one point per arm."""
    base, _ = series(*BASE, which)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for label, exp, tag, spec, slim, colour, marker in PLOT_ARMS:
        cur, _ = series(exp, tag, which)
        ids = sorted(set(base) & set(cur))
        if not ids:
            continue
        d = np.array([cur[i] - base[i] for i in ids])
        m, lo, hi = el.paired_bootstrap_ci(d)
        gb = size_gb(spec, slim)
        ax.errorbar([gb], [m], yerr=[[m - lo], [hi - m]], fmt=marker, color=colour,
                    ecolor=MUTED, capsize=3, ms=8, lw=1)
        ax.annotate(f"  {label}", (gb, m), fontsize=9, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("projected checkpoint size (GB, lower is better ->)")
    ax.set_ylabel("paired dminADE@8 vs baseline (m)")
    ax.set_title(f"what each compression buys, and what it costs ({which} 500)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="test")
    args = ap.parse_args()
    w = args.which

    base, _ = series(*BASE, w)
    prune, _ = series(*PRUNE, w)
    if not base or not prune:
        raise SystemExit("baseline or prune-only rows missing")

    out = {"set": w, "n_baseline": len(base), "cells": {}}
    print(f"set={w}  baseline n={len(base)}  mean minADE "
          f"{np.mean(list(base.values())):.4f}\n")
    for name, q, both, spec in CELLS:
        quant, _ = series(*q, w)
        comb, comb_rows = series(*both, w)
        ids = sorted(set(base) & set(prune) & set(quant) & set(comb))
        if not ids:
            print(f"{name}: no overlapping clips yet"); continue
        dp = np.array([prune[i] - base[i] for i in ids])
        dq = np.array([quant[i] - base[i] for i in ids])
        db = np.array([comb[i] - base[i] for i in ids])
        inter = db - dp - dq
        mp, mq, mb = (ci(dp), ci(dq), ci(db))
        mi = ci(inter)
        gb = size_gb(spec, SLIM)
        deg = float(np.mean([r["coc_degenerate"] for r in comb_rows]))
        print(f"=== {name}  (n={len(ids)}, {gb:.2f} GB, {FULL_GB / gb:.2f}x) ===")
        print(f"  prune only        {mp[0]:+.4f} [{mp[1]:+.4f}, {mp[2]:+.4f}]{mp[3]}")
        print(f"  quant only        {mq[0]:+.4f} [{mq[1]:+.4f}, {mq[2]:+.4f}]{mq[3]}")
        print(f"  both (measured)   {mb[0]:+.4f} [{mb[1]:+.4f}, {mb[2]:+.4f}]{mb[3]}")
        print(f"  additive predict  {mp[0] + mq[0]:+.4f}")
        print(f"  INTERACTION       {mi[0]:+.4f} [{mi[1]:+.4f}, {mi[2]:+.4f}]{mi[3]}"
              f"   p={wilcoxon(inter).pvalue:.2e}")
        print(f"  verdict           {'independent' if not mi[3] else ('super-additive' if mi[0] > 0 else 'sub-additive')}")
        print(f"  CoC degeneracy    {deg * 100:.1f}%  (gate X3: < 5%)\n")
        out["cells"][name] = {
            "n": len(ids), "gb": gb, "ratio": FULL_GB / gb,
            "prune": mp[:3], "quant": mq[:3], "both": mb[:3],
            "additive_prediction": mp[0] + mq[0],
            "interaction": {"mean": mi[0], "ci": [mi[1], mi[2]],
                            "p": float(wilcoxon(inter).pvalue),
                            "independent": not mi[3]},
            "coc_degenerate": deg,
        }
    d = REPO / "outputs" / f"composition_{w}"
    (d / "plots").mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    lines = [f"prune x quant composition, set={w}, baseline n={len(base)}", ""]
    for name, c in out["cells"].items():
        i = c["interaction"]
        verdict = "independent" if i["independent"] else "INTERACTING"
        lines += [
            f"[{name}]  {c['gb']:.2f} GB ({c['ratio']:.2f}x)  n={c['n']}",
            (f"  prune {c['prune'][0]:+.4f}  quant {c['quant'][0]:+.4f}  "
             f"both {c['both'][0]:+.4f}  additive {c['additive_prediction']:+.4f}"),
            (f"  interaction {i['mean']:+.4f} [{i['ci'][0]:+.4f}, {i['ci'][1]:+.4f}] "
             f"p={i['p']:.2e} -> {verdict}"),
            f"  CoC degeneracy {c['coc_degenerate'] * 100:.1f}%", ""]
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    tradeoff_plot(w, out, d / "plots" / "tradeoff.png")
    print("saved ->", d)


if __name__ == "__main__":
    main()
