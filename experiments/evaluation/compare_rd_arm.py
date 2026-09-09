"""Compare a calibration-variant arm against its calib_100 twin, clip by clip.

The two arms differ in one factor -- which 100 clips the criterion (or, for Tyr, the
reconstruction Hessian) was measured on -- so the comparison is paired on clip_id.  Seeds
come from the clip id (`sample_cache.clip_seed`), not the loop index, so the same clip
gets the same 8 rollout samples in both runs and the pairing is exact.  Runs measured on
cvlab20 pair with cvlab21 ones because both boxes were shown bit-identical on the
baseline AND the slim path (cvlab20-server/cvlab20-usage.md section 4).

minADE here is `minADE_rollout`: min over the 8 sampled rollouts of the ADE over the full
6.4 s / 64-waypoint horizon, which is exactly what run_baseline's own summary line reports.

    python experiments/evaluation/compare_rd_arm.py \
        --arm outputs/tyr_rd_a --ref outputs/tyr_u40_r --base outputs/baseline_ada_ps
"""

import argparse
import json
from pathlib import Path

import numpy as np

# run_baseline names the OOD run's dir with a different suffix than its --set
SETS = [("test", "test", "test500"), ("indist", "indist", "val500"),
        ("oodval", "oodval", "OOD-val")]


def load(run_dir):
    """-> {clip_id: row}. The shard file is named for the model, so glob it."""
    d = Path(run_dir)
    if not d.exists():
        return None
    files = sorted(d.glob("*_s*of*.json"))
    if not files:
        return None
    rows = {}
    for f in files:
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r
    return rows


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
    return float(wilcoxon(d).pvalue) if len(d) >= 5 else None


# @6 counts SAMPLES, not seconds. `minADE_rollout` is the min over all 8 and reads
# off-protocol -- dual on val500 is 0.7766 at K=8 against the 0.8904 the reports carry.
K = 6


def minade(r):
    return min(r["ade_rollout_k"][:K])


def stats(rows, key=None):
    v = np.asarray([minade(r) for r in rows.values()], float)
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "coc_degen": float(100 * np.mean([r["coc_degenerate"] for r in rows.values()]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="dir prefix, e.g. outputs/tyr_rd_a")
    ap.add_argument("--ref", required=True, help="the calib_100 twin, e.g. outputs/tyr_u40_r")
    ap.add_argument("--base", default=None, help="optional unpruned baseline prefix")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report, missing = [], []
    for suffix, _set, label in SETS:
        a = load(f"{args.arm}_{suffix}")
        r = load(f"{args.ref}_{suffix}")
        if a is None:
            missing.append(label)
            continue
        row = {"set": label, "arm": stats(a)}
        if r is not None:
            common = sorted(set(a) & set(r))
            row["ref"] = stats({k: r[k] for k in common})
            row["arm_common"] = stats({k: a[k] for k in common})
            d = [minade(a[k]) - minade(r[k]) for k in common]
            dm, lo, hi = boot(d)
            row["paired"] = {
                "n": len(common), "d_mean": dm, "lo": lo, "hi": hi,
                "d_median": float(np.median(d)), "p": wilcoxon_p(d),
                "better": int(np.sum(np.asarray(d) < 0)),
                "worse": int(np.sum(np.asarray(d) > 0)),
                "identical": int(np.sum(np.asarray(d) == 0)),
                "coc_same": float(100 * np.mean(
                    [a[k].get("gen_coc") == r[k].get("gen_coc") for k in common])),
            }
        if args.base:
            b = load(f"{args.base}_{suffix}")
            if b is not None:
                common = sorted(set(a) & set(b))
                db = [minade(a[k]) - minade(b[k]) for k in common]
                row["vs_base"] = {"n": len(common), "d_mean": float(np.mean(db)),
                                  "base_mean": stats({k: b[k] for k in common})["mean"]}
        report.append(row)

    for row in report:
        s = row["set"]
        a = row["arm"]
        print(f"\n=== {s}  (n={a['n']}) ===")
        print(f"  arm          mean {a['mean']:.4f}  median {a['median']:.4f}  "
              f"CoC 퇴화 {a['coc_degen']:.1f}%")
        if "ref" in row:
            r, p = row["ref"], row["paired"]
            print(f"  ref(calib_100) mean {r['mean']:.4f}  median {r['median']:.4f}  "
                  f"CoC 퇴화 {r['coc_degen']:.1f}%")
            sig = "유의" if p["p"] is not None and p["p"] < 0.05 else "n.s."
            print(f"  arm - ref    mean {p['d_mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]  "
                  f"median {p['d_median']:+.4f}  p={p['p']:.4f} ({sig})")
            print(f"               개선 {p['better']} / 악화 {p['worse']} / "
                  f"동일 {p['identical']}  |  CoC 텍스트 일치 {p['coc_same']:.1f}%")
        if "vs_base" in row:
            v = row["vs_base"]
            print(f"  vs 무압축     baseline {v['base_mean']:.4f}  "
                  f"압축 비용 {v['d_mean']:+.4f}")
    if missing:
        print(f"\n아직 없는 세트: {', '.join(missing)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
