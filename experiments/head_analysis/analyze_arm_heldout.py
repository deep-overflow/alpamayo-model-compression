"""Gate A2-heldout of plans/2026-09-21_importance-causal-validation.md.

A2 read the three shipped arms on calib_100, where every criterion is in-sample for its own
loss. This reads the same three masks on held-out clips (run_token_ablation.py --arm-masks):
the dense model's text, the same noise, so FM loss, CoC NLL and minADE are paired per clip.

  prediction  FM loss highest under `coc`, NLL highest under `traj`, `dual` lowest on both
              (paired one-sided Wilcoxon, p < 0.01 each)

Usage:
  python experiments/head_analysis/analyze_arm_heldout.py --shards armheld_v1_s0 armheld_v1_s1 \
      armheld_v1_s2 armheld_v1_s3 --out armheld_v1
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
ARMS = ("dual", "traj", "coc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    per, ids, buckets = merge(args.shards)
    cfg0 = json.loads((REPO / "outputs" / args.shards[0] / "config.json").read_text())
    rng = np.random.default_rng(0)
    x = {k: {a: np.array(per["dense" if a == "dense" else f"arm_{a}"][k], float) for a in ("dense",) + ARMS}
         for k in ("fm", "nll", "ade")}

    def ci(d):
        b = rng.choice(d, (args.n_boot, len(d))).mean(1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    out = {"n_clips": len(ids), "manifest": cfg0["manifest"], "means": {}, "tests": {}}
    lines = [f"shipped arms on held-out clips -- {len(ids)} clips of {cfg0['manifest']}, dense text, K={cfg0['k_samples']}",
             f"  {'':6s} {'FM loss':>9s} {'d [95% CI]':>26s} {'CoC NLL':>9s} {'d [95% CI]':>26s} {'minADE':>8s} {'d [95% CI]':>26s}"]
    for a in ("dense",) + ARMS:
        row = f"  {a:6s}"
        out["means"][a] = {}
        for k in ("fm", "nll", "ade"):
            d = x[k][a] - x[k]["dense"]
            lo, hi = ci(d) if a != "dense" else (0.0, 0.0)
            out["means"][a][k] = {"mean": float(x[k][a].mean()), "delta": float(d.mean()), "ci": [lo, hi],
                                  "median_delta": float(np.median(d))}
            row += f" {x[k][a].mean():9.4f} {d.mean():+9.4f} [{lo:+.4f},{hi:+.4f}]"
        lines.append(row)
    lines.append("")
    for label, k, hi, lo, gated in (("FM", "fm", "coc", "dual", True), ("FM", "fm", "coc", "traj", True),
                                    ("NLL", "nll", "traj", "dual", True), ("NLL", "nll", "traj", "coc", True),
                                    ("FM", "fm", "traj", "dual", True), ("NLL", "nll", "coc", "dual", True),
                                    ("minADE", "ade", "coc", "dual", False), ("minADE", "ade", "traj", "dual", False),
                                    ("minADE", "ade", "coc", "traj", False)):
        p = float(wilcoxon(x[k][hi], x[k][lo], alternative="greater").pvalue)
        out["tests"][f"{label}:{hi}>{lo}"] = {"p": p, "median_diff": float(np.median(x[k][hi] - x[k][lo])),
                                             "gated": gated, "pass": bool(p < 0.01)}
        lines.append(f"  {label:6s} {hi:4s} > {lo:4s}: median paired diff {np.median(x[k][hi] - x[k][lo]):+.4f}, "
                     f"one-sided Wilcoxon p = {p:.2e}" + (f" -> {'PASS' if p < 0.01 else 'FAIL'}" if gated else "  (descriptive)"))
    out["pass"] = bool(all(t["pass"] for t in out["tests"].values() if t["gated"]))
    gap_in = None
    a2 = REPO / "outputs" / "gradanat_pruned_v1" / "metrics.json"
    if a2.exists():
        m = json.loads(a2.read_text())["A2"]["nll_mean"]
        gap_in = m["traj"] - m["coc"]
    gap_out = out["means"]["traj"]["nll"]["mean"] - out["means"]["coc"]["nll"]["mean"]
    out["nll_gap_traj_minus_coc"] = {"held_out": gap_out, "in_sample": gap_in}
    lines += [f"  A2-heldout -> {'PASS' if out['pass'] else 'FAIL'}",
              f"  NLL gap traj - coc: held out {gap_out:+.4f}" + (f", in sample (calib_100) {gap_in:+.4f}" if gap_in is not None else "")]

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md (A2-heldout)", "shards": args.shards,
        "clip_ids": ids, "buckets": buckets, "arm_masks": cfg0["arm_masks"], "n_boot": args.n_boot}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
