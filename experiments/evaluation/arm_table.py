"""Mean minADE@6 / minFDE@6 for a set of arms, paired against one reference.

Both metrics are over the full 6.4 s / 64-waypoint horizon; the `@6` counts SAMPLES, so
they take the min over the FIRST 6 of the 8 sampled rollouts. run_baseline stores the
per-sample arrays (`ade_rollout_k` / `fde_rollout_k`) precisely so any K' <= k is a
prefix; its own `minADE_rollout` field is the min over all 8 and is off-protocol.

The inference matches the statistic. A paired MEAN difference is tested with the
bootstrap CI over per-clip differences; Wilcoxon is a test on the median and is reported
alongside only so the two readings can be compared, not as the mean's p-value. These
deltas are heavy-tailed -- one broken config lands at 25 m -- so a mean CI that excludes
zero while the median sits at zero means "a few clips moved a long way", not "the arm is
uniformly worse".

    python experiments/evaluation/arm_table.py --ref tyr_u40_r --arms tyrK tyr_rd_a ...
"""

import argparse
import json
from pathlib import Path

import numpy as np

O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
SETS = [("test", "test500"), ("indist", "val500"), ("oodval", "OOD-val")]
# The frozen protocol is minADE@6 / minFDE@6, and the @6 counts SAMPLES, not seconds:
# run_baseline stores the per-sample arrays precisely so "minADE@K' for any K' <= k is a
# prefix of these". The stored `minADE_rollout` is the min over all 8, i.e. @8, and using
# it puts every absolute number off-protocol -- dual on val500 reads 0.7766 instead of the
# 0.8904 the reports carry. Recompute from the arrays instead.
K = 6
METRICS = [("ade_rollout_k", "minADE@6"), ("fde_rollout_k", "minFDE@6")]


def load(stem, suffix):
    d = O / f"{stem}_{suffix}"
    if not d.exists():
        return None
    rows = {}
    for f in sorted(d.glob("*_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r
    return rows or None


def boot(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def wilcoxon_p(d):
    from scipy.stats import wilcoxon

    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    return float(wilcoxon(d).pvalue) if len(d) >= 5 else float("nan")


def collect(arms, ref):
    out = {}
    for suffix, label in SETS:
        R = load(ref, suffix)
        if R is None:
            continue
        rows = {}
        for arm in arms:
            A = load(arm, suffix)
            if A is None:
                continue
            common = sorted(set(A) & set(R))
            e = {"n": len(common)}
            for key, mname in METRICS:
                a = np.array([min(A[c][key][:K]) for c in common], float)
                r = np.array([min(R[c][key][:K]) for c in common], float)
                d = a - r
                e[mname] = {
                    "mean": float(a.mean()), "ref_mean": float(r.mean()),
                    "d_mean": float(d.mean()),
                    "lo": boot(d)[1], "hi": boot(d)[2],
                    "d_median": float(np.median(d)),
                    "p_wilcoxon": wilcoxon_p(d) if arm != ref else float("nan"),
                }
            e["coc_degen"] = float(100 * np.mean([A[c]["coc_degenerate"] for c in common]))
            e["coc_same"] = float(100 * np.mean(
                [A[c].get("gen_coc") == R[c].get("gen_coc") for c in common]))
            rows[arm] = e
        out[label] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = [args.ref] + [a for a in args.arms if a != args.ref]
    data = collect(arms, args.ref)

    for label, rows in data.items():
        n = next(iter(rows.values()))["n"]
        print(f"\n=== {label}  (n={n}, 참조 {args.ref}) ===")
        head = (f"{'arm':22s}{'minADE@6':>10s}{'Δ':>9s}{'95% CI':>20s}"
                f"{'minFDE@6':>10s}{'Δ':>9s}{'95% CI':>20s}{'CoC퇴화':>9s}")
        print(head)
        print("-" * len(head))
        for arm in arms:
            if arm not in rows:
                continue
            e = rows[arm]
            line = f"{arm[:21]:22s}"
            for _, mname in METRICS:
                m = e[mname]
                if arm == args.ref:
                    line += f"{m['mean']:10.4f}{'--':>9s}{'--':>20s}"
                else:
                    ci = f"[{m['lo']:+.4f}, {m['hi']:+.4f}]"
                    line += f"{m['mean']:10.4f}{m['d_mean']:+9.4f}{ci:>20s}"
            line += f"{e['coc_degen']:8.1f}%"
            print(line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"ref": args.ref, "sets": data},
                                       indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
