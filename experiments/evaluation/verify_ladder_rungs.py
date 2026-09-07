"""Gate the derived ladder rungs before anything is built from them.

`make_block_importance.py` writes a prefix's importance as the fp64 mean of that prefix's
per-clip rows. The file existing proves none of what the design needs, so check it:

  1. NESTING -- c500's clips are the first 500 of c1000's, and c1000's the first 1000 of
     c2000's. Without it the draw moves with n and the ladder measures nothing.
  2. SANITY -- same keys, no NaN, magnitudes in family across rungs.
  3. CONVERGENCE -- do the within-layer rankings settle as n grows.
  4. HEADROOM -- does the kept set still move enough between rungs to be worth evaluating.

Usage:
  python experiments/evaluation/verify_ladder_rungs.py \
      --runs importance_st4000_c500 importance_st4000_c1000 importance_st4000_c2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
KEEP = 0.3985632694          # the u40_v2 uniform ratio, from run_grid.allocations()
AXES = (("q", "traj_vlm_q", "coc_vlm_q"), ("mlp", "traj_vlm_mlp", "coc_vlm_mlp"))


def dual_kept(z, lyr, tk, ck):
    """max(rank I_traj, rank I_CoC) within one layer -- the dual recipe make_slim uses."""
    t, c = np.asarray(z[tk], float)[lyr], np.asarray(z[ck], float)[lyr]
    rt = np.argsort(np.argsort(t)) / (len(t) - 1)
    rc = np.argsort(np.argsort(c)) / (len(c) - 1)
    return set(np.argsort(-np.maximum(rt, rc))[:round(KEEP * len(t))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="importance run dirs, ascending in n; the last is the reference")
    ap.add_argument("--out", default="outputs/ladder_rungs")
    args = ap.parse_args()

    dirs = [REPO / "outputs" / r for r in args.runs]
    cfgs = [json.loads((d / "config.json").read_text()) for d in dirs]
    z = [dict(np.load(d / "importance.npz")) for d in dirs]
    ref, ref_name = z[-1], args.runs[-1]
    m = {"runs": args.runs}

    print("=== 1. 중첩 ===")
    ids = [c["clip_ids"] for c in cfgs]
    nested = True
    for i in range(len(ids) - 1):
        ok = ids[i] == ids[i + 1][:len(ids[i])]
        nested &= ok
        print(f"  {args.runs[i]} (n={len(ids[i])}) 가 {args.runs[i + 1]}의 앞부분: {ok}")
    m["nested"] = bool(nested)

    print("\n=== 2. 건전성 ===")
    keys_ok = all(sorted(x) == sorted(ref) for x in z)
    nan = sum(int(np.isnan(np.asarray(v, float)).sum()) for x in z for v in x.values())
    print(f"  키 동일 {keys_ok}   NaN 총합 {nan}")
    m["keys_identical"], m["nan_total"] = bool(keys_ok), nan
    for _, tk, ck in AXES:
        for k in (tk, ck):
            print(f"  {k:14s} " + "  ".join(
                f"{n.split('_')[-1]}: {np.asarray(x[k], float).mean():.3e}"
                for n, x in zip(args.runs, z)))

    print(f"\n=== 3. 순위 수렴 (층별 spearman, {ref_name} 대비) ===")
    m["rank_rho"] = {}
    for _, tk, ck in AXES:
        for k in (tk, ck):
            row = f"  {k:14s}"
            for n, x in zip(args.runs[:-1], z[:-1]):
                a, b = np.asarray(x[k], float), np.asarray(ref[k], float)
                per = [spearmanr(a[i], b[i]).statistic for i in range(a.shape[0])]
                per = [p for p in per if np.isfinite(p)]
                m["rank_rho"][f"{n}|{k}"] = float(np.mean(per))
                row += f"  {n.split('_')[-1]}: {np.mean(per):+.4f}"
            print(row)

    print(f"\n=== 4. 유지집합 여지 (dual 레시피, {ref_name} 대비) ===")
    m["kept_overlap"] = {}
    for axis, tk, ck in AXES:
        n_layer = np.asarray(ref[tk]).shape[0]
        ref_sets = [dual_kept(ref, lyr, tk, ck) for lyr in range(n_layer)]
        for n, x in zip(args.runs[:-1], z[:-1]):
            ov = tot = 0
            for lyr in range(n_layer):
                s = dual_kept(x, lyr, tk, ck)
                ov += len(s & ref_sets[lyr])
                tot += len(s)
            m["kept_overlap"][f"{n}|{axis}"] = ov / tot
            print(f"  {n.split('_')[-1]:6s} {axis:4s} 중첩 {ov / tot:.4f}")

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print("\n->", out)
    if not nested or nan:
        raise SystemExit("REFUSING: 중첩이 깨졌거나 NaN이 있다 -- 빌드하지 말 것")


if __name__ == "__main__":
    main()
