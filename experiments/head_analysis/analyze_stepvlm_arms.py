"""Gates V2 / V3: does the step-fixed VLM criterion beat the shipped one at matched budget?

Both arms are the same config (`traj_u40_v2` or `dual_u40_v2`), the same uniform ratio, the
same axes, and remove exactly the same parameter count. Only the importance file differs:
the control is `run_importance`'s |sum_s dL_s/dg|, the arm is the step-normalised version of
the identical gradients. Both are measured on Ada, because Ada-vs-Blackwell drift alone
moves this selection 2-3% and the shipped Blackwell rows would confound the comparison.

Paired on clip and seed, so sampling noise cancels in the difference. minADE@6 per the fixed
protocol (runs store k=8; seeds are base+i, so the first 6 are what a 6-sample run drew).

Usage:
  python experiments/head_analysis/analyze_stepvlm_arms.py \
      --control trajctl_val --arm trajznorm_val --out stepvlm_arms_val
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]


def boot_ci(d, n_boot=10000, seed=0, stat=np.mean):
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    b = stat(d[rng.randint(0, len(d), size=(n_boot, len(d)))], axis=1)
    return float(stat(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load_rows(exp_id):
    d = REPO / "outputs" / exp_id
    rows = []
    for p in sorted(d.glob("*_s*of*.json")):
        rows.extend(json.loads(p.read_text()))
    return {r["clip_id"]: r for r in rows}


def reduce_at(rows, ids, field, k):
    return np.array([min(rows[c][field][:k]) for c in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    ctl, arm = load_rows(args.control), load_rows(args.arm)
    ids = sorted(set(ctl) & set(arm))
    if not ids:
        raise SystemExit("no shared clips between the two runs")
    lines = [
        f"VLM step-fix arms -- {args.label or args.out}",
        (f"  control {args.control} (n={len(ctl)})  arm {args.arm} (n={len(arm)})  "
         f"paired on {len(ids)} clips, minADE@{args.k}"),
        "",
    ]
    res = {"control": args.control, "arm": args.arm, "n_paired": len(ids), "k": args.k}

    # seeds are derived from the clip id, so a paired difference is the same noise draw
    mism = [c for c in ids if ctl[c]["seed"] != arm[c]["seed"]]
    if mism:
        raise SystemExit(f"{len(mism)} clips have different seeds; pairing is invalid")
    lines.append(f"  seed check: all {len(ids)} clips share a seed between arms  OK")

    for field, name in (("ade_rollout_k", "minADE"), ("fde_rollout_k", "minFDE")):
        a = reduce_at(ctl, ids, field, args.k)
        b = reduce_at(arm, ids, field, args.k)
        d = b - a
        m, lo, hi = boot_ci(d)
        med, mlo, mhi = boot_ci(d, stat=np.median)
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        res[name] = {"control": float(a.mean()), "arm": float(b.mean()),
                     "control_median": float(np.median(a)), "arm_median": float(np.median(b)),
                     "delta_mean": m, "lo": lo, "hi": hi,
                     "delta_median": med, "median_lo": mlo, "median_hi": mhi,
                     "wilcoxon_p": p}
        lines.append(f"  {name}@{args.k}: control {a.mean():.4f} (med {np.median(a):.4f})  "
                     f"arm {b.mean():.4f} (med {np.median(b):.4f})")
        lines.append(f"           paired delta (arm - control) {m:+.4f} [{lo:+.4f},{hi:+.4f}]"
                     f"   median {med:+.4f} [{mlo:+.4f},{mhi:+.4f}]  p={p:.4f}")

    for key in ("coc_degenerate", "coc_empty"):
        if key in ctl[ids[0]]:
            a = np.array([float(ctl[c][key]) for c in ids])
            b = np.array([float(arm[c][key]) for c in ids])
            res[key] = {"control": float(a.mean()), "arm": float(b.mean())}
            lines.append(f"  {key}: control {a.mean():.4f}  arm {b.mean():.4f}")

    buckets = np.array([ctl[c]["bucket"] for c in ids])
    a = reduce_at(ctl, ids, "ade_rollout_k", args.k)
    b = reduce_at(arm, ids, "ade_rollout_k", args.k)
    res["by_bucket"] = {}
    lines.append("  per-bucket paired delta:")
    for bk in sorted(set(buckets)):
        sel = buckets == bk
        m, lo, hi = boot_ci((b - a)[sel])
        res["by_bucket"][bk] = {"n": int(sel.sum()), "mean": m, "lo": lo, "hi": hi}
        lines.append(f"    {bk:11s} n={int(sel.sum()):3d}  {m:+.4f} [{lo:+.4f},{hi:+.4f}]")

    d = b - a
    m, lo, hi = boot_ci(d)
    verdict = "ACCEPT" if hi < 0 else ("REGRESSION" if lo > 0 else "INCONCLUSIVE")
    note = {"ACCEPT": "the step-fixed criterion is better at matched budget",
            "REGRESSION": "the step fix is worse -- the expert result does not carry over",
            "INCONCLUSIVE": "no separation; the fix changes the picks but not the outcome"}
    lines.append("")
    lines.append(f"  -> {verdict}: {note[verdict]}")
    res["verdict"] = verdict

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text("\n".join(lines))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
