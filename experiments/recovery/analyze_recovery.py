"""Recovery evaluation: minADE best-of-6 tables, paired deltas, pre-registered gates.

Open-loop metric is minADE@6, re-reduced from stored per-sample arrays
(`ade_rollout_k`; seeds are base+k so the first 6 samples of a K=8 run are exactly a
K=6 run). Arms whose rows lack the arrays cannot be re-reduced and are refused rather
than silently reported at a different K.

Sets and sources:
  val_500  baseline_ada_ps_indist / dual_u40_v2_ps_indist / dual_u55_indist / <rec>_indist
  test_500 baseline_ada_ps_test   / dual_u40_v2_ps_test   / dual_u55_test   / <rec>_test
  ood_val  baseline_ada_ps_oodval / dual_u40_v2_ps_ood (sliced to the 262) /
           dual_u55_oodval        / <rec>_oodval

Gates (plans/2026-08-19_recovery-training.md):
  G1 recovered test_500 minADE@6 mean <= 1.6 AND paired delta vs zero-shot u55 CI < 0
  G2 recovered test_500 CoC degen <= 0.05
  G3 (with --peer) pooled d(recovered - peer) over the three sets: CI above 0 means the
     pruning criterion's damage survives recovery; CI containing 0 with |mean| < 0.05
     means recovery erases it (plans/2026-08-23_coc-u55-recovery.md).

Usage:
  python experiments/recovery/analyze_recovery.py [--rec slim_recover_dual_u55] \
      [--out outputs/recovery_eval]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

import matplotlib

matplotlib.use("Agg")
import eval_lib as el
import matplotlib.pyplot as plt

K_REPORT = 6

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
ARM_ORDER = ("baseline", "dual_u40", "zeroshot", "recovered")
ARM_COLOR = {"baseline": MUTED, "dual_u40": C4, "zeroshot": C3, "recovered": C2,
             "peer": C1}


def load_rows(exp_dir):
    rows = []
    for f in sorted(glob.glob(str(REPO / "outputs" / exp_dir / "*_s*of*.json"))):
        rows += json.loads(Path(f).read_text())
    return {r["clip_id"]: r for r in rows}


def at_k(rows, ids, key="ade_rollout_k"):
    missing = [i for i in ids if key not in rows[i]]
    assert not missing, f"{len(missing)} rows lack {key} -- cannot re-reduce to @{K_REPORT}"
    return np.array([min(rows[i][key][:K_REPORT]) for i in ids])


def arm_stats(rows, ids):
    a = at_k(rows, ids)
    degen = np.array([bool(rows[i]["coc_degenerate"]) for i in ids])
    return {"n": len(ids), "minADE6_mean": float(a.mean()),
            "minADE6_median": float(np.median(a)), "degen": float(degen.mean())}


def paired(rows_a, rows_b, ids):
    """delta = a - b on common clips, bootstrap CI + Wilcoxon."""
    d = at_k(rows_a, ids) - at_k(rows_b, ids)
    _, lo, hi = el.paired_bootstrap_ci(d)
    from scipy.stats import wilcoxon
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "ci": [float(lo), float(hi)], "wilcoxon_p": p,
            "sig": not (lo <= 0 <= hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=str, default="slim_recover_dual_u55",
                    help="recovered model tag; rows live in <tag>_{indist,test,oodval}")
    ap.add_argument("--zs", type=str, default="dual_u55",
                    help="zero-shot arm stem; rows in <stem>_{indist,test,oodval}")
    ap.add_argument("--peer", type=str, default=None,
                    help="second recovered arm to compare against; rows in "
                         "<tag>_{indist,test,oodval}")
    ap.add_argument("--out", type=str, default="outputs/recovery_eval")
    args = ap.parse_args()
    arm_order = ARM_ORDER + (("peer",) if args.peer else ())

    oodval_ids = set(pd.read_parquet(
        REPO / "outputs" / "eval_sets" / "ood_val.parquet").clip_id)
    sets = {
        "val_500": {"baseline": "baseline_ada_ps_indist", "dual_u40": "dual_u40_v2_ps_indist",
                    "zeroshot": f"{args.zs}_indist", "recovered": f"{args.rec}_indist"},
        "test_500": {"baseline": "baseline_ada_ps_test", "dual_u40": "dual_u40_v2_ps_test",
                     "zeroshot": f"{args.zs}_test", "recovered": f"{args.rec}_test"},
        "ood_val": {"baseline": "baseline_ada_ps_oodval", "dual_u40": "dual_u40_v2_ps_ood",
                    "zeroshot": f"{args.zs}_oodval", "recovered": f"{args.rec}_oodval"},
    }
    if args.peer:
        for set_name, suf in (("val_500", "indist"), ("test_500", "test"),
                              ("ood_val", "oodval")):
            sets[set_name]["peer"] = f"{args.peer}_{suf}"

    out = REPO / Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report, lines, pooled_src = {}, [], {}
    for set_name, arms in sets.items():
        loaded = {a: load_rows(d) for a, d in arms.items()}
        present = {a: r for a, r in loaded.items() if r}
        ids = set.intersection(*[set(r) for r in present.values()])
        if set_name == "ood_val":
            ids &= oodval_ids
        ids = sorted(ids)
        pooled_src[set_name] = (present, ids)
        entry = {"n_common": len(ids), "arms": {}, "paired": {}}
        lines.append(f"\n== {set_name} (common clips {len(ids)}; "
                     f"minADE@{K_REPORT}, rollout) ==")
        lines.append(f"{'arm':14s} {'mean':>8} {'median':>8} {'degen':>7}")
        for a in arm_order:
            if a not in present:
                lines.append(f"{a:14s} {'--':>8} (rows missing)")
                continue
            s = arm_stats(present[a], ids)
            entry["arms"][a] = s
            lines.append(f"{a:14s} {s['minADE6_mean']:8.4f} {s['minADE6_median']:8.4f} "
                         f"{s['degen']:7.3f}")
        for a, b in (("recovered", "baseline"), ("recovered", "zeroshot"),
                     ("recovered", "dual_u40"), ("zeroshot", "baseline"),
                     ("recovered", "peer")):
            if a in present and b in present:
                entry["paired"][f"{a}-{b}"] = pr = paired(present[a], present[b], ids)
                lines.append(f"  d({a}-{b}): {pr['mean']:+.4f} "
                             f"[{pr['ci'][0]:+.4f},{pr['ci'][1]:+.4f}] "
                             f"p={pr['wilcoxon_p']:.4f}{' *' if pr['sig'] else ''}")
        report[set_name] = entry

    # Per-set deltas of the size we care about (~0.03) sit inside each set's CI, so the
    # criterion verdict is pooled over all three sets -- the same pooling that resolved
    # u55 v1 vs v2 (n=1,262, delta -0.0341).
    report["pooled"] = {}
    lines.append("\n== pooled (val_500 + test_500 + ood_val) ==")
    for a, b in (("recovered", "zeroshot"), ("recovered", "dual_u40"),
                 ("recovered", "peer")):
        ds = [at_k(pr[a], ii) - at_k(pr[b], ii)
              for pr, ii in pooled_src.values() if a in pr and b in pr]
        if not ds:
            continue
        d = np.concatenate(ds)
        _, lo, hi = el.paired_bootstrap_ci(d)
        from scipy.stats import wilcoxon
        pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        sig = not (lo <= 0 <= hi)
        report["pooled"][f"{a}-{b}"] = {"n": int(d.size), "mean": float(d.mean()),
                                        "median": float(np.median(d)),
                                        "ci": [float(lo), float(hi)],
                                        "wilcoxon_p": pv, "sig": sig}
        lines.append(f"  d({a}-{b}) n={d.size}: {d.mean():+.4f} "
                     f"[{lo:+.4f},{hi:+.4f}] p={pv:.4f}{' *' if sig else ''}")

    gates = {}
    t = report.get("test_500", {})
    if "recovered" in t.get("arms", {}):
        rec = t["arms"]["recovered"]
        dz = t["paired"].get("recovered-zeroshot", {})
        gates["G1_minADE6_le_1.6"] = rec["minADE6_mean"] <= 1.6
        gates["G1_improves_zeroshot"] = bool(dz.get("sig")) and dz.get("mean", 0) < 0
        gates["G2_degen_le_0.05"] = rec["degen"] <= 0.05
        g3 = report["pooled"].get("recovered-peer")
        if g3:
            # H1: the criterion's damage survives recovery. H0: recovery erases it.
            gates["G3_criterion_persists"] = bool(g3["sig"] and g3["mean"] > 0)
            gates["G3_recovery_erases"] = bool(not g3["sig"] and abs(g3["mean"]) < 0.05)
        lines.append("\n== gates (test_500) ==")
        for g, ok in gates.items():
            lines.append(f"  {g}: {'PASS' if ok else 'FAIL'}")
    report["gates"] = gates

    labels = {"baseline": "baseline 11.1B", "dual_u40": "dual_u40 zs 8.42B",
              "zeroshot": f"{args.zs} zs", "recovered": f"{args.rec.replace('slim_recover_', '')} recovered"}
    if args.peer:
        labels["peer"] = f"{args.peer.replace('slim_recover_', '')} recovered"
    plot_at6(report, out / "plots", labels, arm_order)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved ->", out)


def plot_at6(report, plots_dir, labels, arm_order=ARM_ORDER):
    """Grouped bars: minADE@6 and degen per set, the arms side by side."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sets = [s for s in ("val_500", "test_500", "ood_val") if s in report]
    for key, ylab, fname in (("minADE6_mean", "minADE@6 (rollout)", "openloop_at6.png"),
                             ("degen", "CoC degeneracy rate", "openloop_degen.png")):
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        w = 0.8 / len(arm_order)
        for i, a in enumerate(arm_order):
            xs, ys = [], []
            for j, s in enumerate(sets):
                if a in report[s]["arms"]:
                    xs.append(j + i * w)
                    ys.append(report[s]["arms"][a][key])
            ax.bar(xs, ys, width=w, color=ARM_COLOR[a], label=labels[a])
        ax.set_xticks(np.arange(len(sets)) + w * (len(arm_order) - 1) / 2)
        ax.set_xticklabels(sets)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8.5, ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
