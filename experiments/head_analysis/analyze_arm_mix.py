"""Gate A2-mix of plans/2026-09-21_importance-causal-validation.md.

A2-bands found that the trajectory-only mask applied in layers 0-21 alone costs the action far
more than the same arm's full mask: its late-layer pruning repairs what its trunk pruning
breaks. This crosses one arm's trunk with another arm's (or a random) late mask on the same
held-out clips and seeds, to ask which late units do the repairing.

  (a) `traj` trunk + `coc` late does NOT repair: FM loss still >= +20% over dense
  (b) `traj` trunk + random late repairs partly: between the full `traj` mask and its trunk alone
  (c) `coc` trunk + `traj` late lowers the FM damage of the `coc` trunk mask

Usage:
  python experiments/head_analysis/analyze_arm_mix.py --mix armmix_v1_s0 ... --arms armheld_v1_s0 ... --out armmix_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))

from analyze_token_ablation import merge  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
READ = (("fm", "FM loss"), ("nll", "CoC NLL"), ("ade", "minADE"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="the A2-heldout shards: full and band masks, same clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    pm, ids_m, _ = merge(args.mix)
    pa, ids_a, _ = merge(args.arms)
    order = [ids_a.index(c) for c in ids_m]  # align the arms run to the mix run's clip order
    x = {k: {} for k, _ in READ}
    for k, _ in READ:
        for c, r in pa.items():
            x[k][c.removeprefix("arm_")] = np.array(r[k], float)[order]
        for c, r in pm.items():
            name = c.removeprefix("arm_")
            v = np.array(r[k], float)
            if name in x[k]:  # measured in both runs: the reproducibility check
                x[k][name + " (rerun)"] = v
            else:
                x[k][name] = v
    dense_drift = {k: float(np.max(np.abs(x[k]["dense"] - x[k]["dense (rerun)"]))) for k, _ in READ}
    rerun = {k: float(np.max(np.abs(x[k]["traj_trunk"] - x[k]["traj_trunk (rerun)"]))) for k, _ in READ}

    def ci(d):
        b = rng.choice(d, (args.n_boot, len(d))).mean(1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    def greater(k, hi, lo):
        return (float(wilcoxon(x[k][hi], x[k][lo], alternative="greater").pvalue),
                float(np.median(x[k][hi] - x[k][lo])))

    show = ["traj", "traj_trunk", "traj_trunk__dual_late", "traj_trunk__coc_late", "traj_trunk__rand_late",
            "coc", "coc_trunk", "coc_trunk__traj_late", "coc_trunk__rand_late", "dual", "dual_trunk"]
    out = {"n_clips": len(ids_m), "reproducibility_max_abs_diff": {"dense": dense_drift, "traj_trunk": rerun}, "configs": {}}
    lines = [f"trunk of one arm x late layers of another -- {len(ids_m)} held-out clips, dense text",
             f"reproducibility across the two runs, max |diff| per clip: dense {dense_drift}, traj_trunk {rerun}", "",
             f"  {'mask':24s} " + " ".join(f"{name + ' d':>10s} {'[95% CI]':>19s} {'rel':>8s}" for _, name in READ)]
    for c in show:
        row, out["configs"][c] = f"  {c:24s}", {}
        for k, _ in READ:
            d = x[k][c] - x[k]["dense"]
            lo, hi = ci(d)
            out["configs"][c][k] = {"delta": float(d.mean()), "ci": [lo, hi], "rel": float(d.mean() / x[k]["dense"].mean())}
            row += f" {d.mean():+10.4f} [{lo:+.4f},{hi:+.4f}] {d.mean() / x[k]['dense'].mean():+8.1%}"
        lines.append(row)
    lines.append("")

    rel = {c: out["configs"][c]["fm"]["rel"] for c in show}
    p_a, m_a = greater("fm", "traj_trunk__coc_late", "traj")
    out["a"] = {"fm_rel": rel["traj_trunk__coc_late"], "still_ge_20pct": bool(rel["traj_trunk__coc_late"] >= 0.20),
                "vs_full_traj": {"p": p_a, "median_diff": m_a}}
    out["a"]["pass"] = out["a"]["still_ge_20pct"]
    p_b1, m_b1 = greater("fm", "traj_trunk__rand_late", "traj")
    p_b2, m_b2 = greater("fm", "traj_trunk", "traj_trunk__rand_late")
    out["b"] = {"fm_rel": rel["traj_trunk__rand_late"], "above_full_traj": {"p": p_b1, "median_diff": m_b1},
                "below_trunk_only": {"p": p_b2, "median_diff": m_b2}, "pass": bool(p_b1 < 0.01 and p_b2 < 0.01)}
    p_c, m_c = greater("fm", "coc_trunk", "coc_trunk__traj_late")
    out["c"] = {"fm_rel": rel["coc_trunk__traj_late"], "coc_trunk_gt_mixed": {"p": p_c, "median_diff": m_c},
                "pass": bool(p_c < 0.01)}
    lines += [f"(a) traj trunk + coc late: FM loss {rel['traj_trunk__coc_late']:+.1%} over dense (no repair predicted, >= +20%); "
              f"against the full traj mask ({rel['traj']:+.1%}): median {m_a:+.4f}, p = {p_a:.2e} -> "
              f"{'PASS' if out['a']['pass'] else 'FAIL'}",
              f"(b) traj trunk + random late: FM loss {rel['traj_trunk__rand_late']:+.1%}; above the full traj mask p = {p_b1:.2e}, "
              f"below the trunk-only mask ({rel['traj_trunk']:+.1%}) p = {p_b2:.2e} -> {'PASS' if out['b']['pass'] else 'FAIL'}",
              f"(c) coc trunk + traj late: FM loss {rel['coc_trunk__traj_late']:+.1%} against coc trunk only "
              f"({rel['coc_trunk']:+.1%}): median {m_c:+.4f}, p = {p_c:.2e} -> {'PASS' if out['c']['pass'] else 'FAIL'}", ""]
    for k, hi, lo in (("fm", "traj_trunk", "traj_trunk__dual_late"), ("fm", "traj_trunk__dual_late", "traj"),
                      ("fm", "traj_trunk__coc_late", "traj_trunk__rand_late"), ("fm", "coc_trunk", "coc_trunk__rand_late"),
                      ("ade", "traj_trunk", "traj_trunk__coc_late"), ("ade", "traj_trunk", "traj_trunk__dual_late"),
                      ("ade", "traj_trunk", "traj_trunk__rand_late"), ("nll", "traj_trunk__rand_late", "traj_trunk__coc_late")):
        p, med = greater(k, hi, lo)
        out.setdefault("descriptive", {})[f"{k}:{hi}>{lo}"] = {"p": p, "median_diff": med}
        lines.append(f"  descriptive {k:3s} {hi:24s} > {lo:24s}: median {med:+.4f}, p = {p:.2e}")

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md (A2-mix)", "mix": args.mix, "arms": args.arms,
        "clip_ids": ids_m, "n_boot": args.n_boot}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
