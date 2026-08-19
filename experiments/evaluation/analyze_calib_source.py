"""Does the calibration distribution change what pruning should keep?

One factor: `dual_u40_v2` and `dual_u40_ood` are the same criterion, budget, allocation
and model revision. Only the 100 clips the Taylor scores were measured on differ --
official train (in-distribution) versus the OOD long-tail pool. Both are judged on sets
disjoint from either calibration set:

  test_500   official test, in-distribution
  ood_val    the 262 OOD clips held out of calib_ood_100

The in-dist arm has never seen OOD clips and the OOD arm has never seen official-train
clips, so neither is advantaged by the judging sets.

Usage:
  python experiments/evaluation/analyze_calib_source.py
"""

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

OUT = REPO / "outputs"
# set -> (exp dir, rows tag) per arm; the two reference arms were measured over the whole
# OOD set, so ood_val is taken as the subset rather than re-run
SETS = {
    "test_500": {
        "baseline": ("baseline_ada_test", "baseline"),
        "in-dist calib": ("dual_u40_v2_test", "slim_dual_u40_v2"),
        "OOD calib": ("dual_u40_ood_test", "slim_dual_u40_ood"),
        "pooled 200": ("dual_u40_mix_test", "slim_dual_u40_mix"),
    },
    "ood_val_262": {
        "baseline": ("baseline_ada_ood", "baseline"),
        "in-dist calib": ("dual_u40_v2_ood", "slim_dual_u40_v2"),
        "OOD calib": ("dual_u40_ood_oodval", "slim_dual_u40_ood"),
        "pooled 200": ("dual_u40_mix_oodval", "slim_dual_u40_mix"),
    },
}
METRICS = ["minADE_rollout", "minFDE_rollout", "nll_self"]


def rows_for(exp, tag, keep=None):
    r = load_rows(OUT / exp, tag)
    if keep is not None:
        r = [x for x in r if x["clip_id"] in keep]
    return {x["clip_id"]: x for x in r}


def main():
    import pandas as pd
    val_ids = set(pd.read_parquet(OUT / "eval_sets" / "ood_val.parquet")["clip_id"])
    out = {}
    for set_name, arms in SETS.items():
        keep = val_ids if set_name.startswith("ood_val") else None
        data = {a: rows_for(*spec, keep) for a, spec in arms.items()}
        missing = [a for a, d in data.items() if not d]
        if missing:
            print(f"[{set_name}] not ready, skipping those arms: {missing}")
            data = {a: d for a, d in data.items() if d}
        if "baseline" not in data or len(data) < 2:
            continue
        ids = sorted(set.intersection(*(set(d) for d in data.values())))
        print(f"\n=== {set_name}  (n={len(ids)} clips shared by all arms) ===")
        print(f"{'arm':16s} " + " ".join(f"{m:>16s}" for m in METRICS))
        for a, d in data.items():
            print(f"{a:16s} " + " ".join(
                f"{np.mean([d[i][m] for i in ids]):16.4f}" for m in METRICS))
        out[set_name] = {"n": len(ids)}
        base = data["baseline"]
        for a in ("in-dist calib", "OOD calib"):
            for m in METRICS:
                dd = np.array([data[a][i][m] - base[i][m] for i in ids])
                mu, lo, hi = el.paired_bootstrap_ci(dd)
                out[set_name][f"{a} vs baseline|{m}"] = [mu, lo, hi]
        ref = "in-dist calib"
        for arm in [a for a in data if a not in ("baseline", ref)]:
            print(f"\n  one-factor contrast ({arm} - {ref}):")
            contrast(data, ids, arm, ref, out[set_name])


    d = OUT / "calib_source"
    (d / "plots").mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    plots(out, d / "plots" / "calib_source.png")
    print("\nsaved ->", d)


BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C4 = "#2a78d6", "#008300", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
# every number here is a paired minADE delta on official test 500, so the bars are
# directly comparable; sources are the reports each was measured in
OTHER_FACTORS = [
    ("calibration -> OOD", 0.1905, C4),
    ("calibration +OOD (half)", 0.0838, C4),
    ("24% pruning itself", 0.0955, MUTED),
    ("+4-bit quantization", 0.0816, MUTED),
    ("criterion (jtraj-dual)", 0.0707, MUTED),
]


def plots(out, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    arms = ["in-dist calib", "pooled 200", "OOD calib"]
    ax = axes[0]
    for j, (s, colour, lbl) in enumerate(
            [("test_500", C1, "test 500"), ("ood_val_262", C2, "OOD-val 262")]):
        if s not in out:
            continue
        xs, ys, los, his = [], [], [], []
        for i, a in enumerate(arms):
            k = f"{a} vs baseline|minADE_rollout"
            if k not in out[s]:
                continue
            mu, lo, hi = out[s][k]
            xs.append(i); ys.append(mu); los.append(mu - lo); his.append(hi - mu)
        ax.errorbar(np.array(xs) + (j - 0.5) * 0.08, ys, yerr=[los, his], fmt="o-",
                    color=colour, ecolor=MUTED, capsize=3, ms=7, lw=1.4, label=lbl)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["in-dist\n100", "pooled\n200", "OOD\n100"])
    ax.set_xlabel("clips the Taylor scores were measured on")
    ax.set_ylabel("paired dminADE@8 vs unpruned (m)")
    ax.set_title("more OOD in the calibration set is monotonically worse")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    labels = [x[0] for x in OTHER_FACTORS]
    vals = [x[1] for x in OTHER_FACTORS]
    cols = [x[2] for x in OTHER_FACTORS]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:+.4f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("paired dminADE@8 on test 500 (m)")
    ax.set_title("what each design choice costs, same axis")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def contrast(data, ids, arm, ref, store):
    """Paired arm - ref on the clips both evaluated, plus what this set can resolve."""
    for m in METRICS:
        dd = np.array([data[arm][i][m] - data[ref][i][m] for i in ids])
        mu, lo, hi = el.paired_bootstrap_ci(dd)
        star = "" if lo <= 0 <= hi else " *"
        p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        print(f"    {m:16s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}  p={p:.2e}")
        store[f"{arm} - {ref}|{m}"] = [mu, lo, hi, float(p)]
        if m == "minADE_rollout":
            sd = dd.std(ddof=1)
            print(f"    {'':16s} sigma={sd:.3f} -> this set resolves "
                  f"{2.80 * sd / np.sqrt(len(dd)):.4f} at 80% power")


if __name__ == "__main__":
    main()
