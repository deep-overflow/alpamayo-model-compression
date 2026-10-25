"""Paired open-loop readout of a Minitron arm against dense and MoOP on stored rows.

minADE@6 / minFDE@6 = min over the first six of the eight stored samples (the paper's
protocol), paired per clip with `baseline_ada_ps_*` (dense) and `dual_u40_v2_ps_*` (MoOP,
24.0%), bootstrap 95% CI of the mean difference (20,000 draws), CoC degeneracy.

Usage:
  .venv/bin/python minitron/code/compare_openloop.py --arm minitron/mtw24_zs [--sets test indist oodval]
"""

import argparse
import glob
import json

import numpy as np

OUT = "/mnt/nvme1n1/ad_vla/outputs/chan"
# set -> (arm suffix, dense dirs, MoOP dirs). "test" is test500 (the stored rows cover only
# those clips, so a 2,000-row arm is cut to them by the pairing); "test2000" adds the 1,500
# clips of eval_sets/test_2000_ext1500, run for the reference arms into minitron/test2000x_*
EXT = "minitron/test2000x_"
SETS = {"test": ("test", ("baseline_ada_ps_test",), ("dual_u40_v2_ps_test",)),
        "test2000": ("test", ("baseline_ada_ps_test", f"{EXT}dense"),
                     ("dual_u40_v2_ps_test", f"{EXT}moop")),
        "indist": ("indist", ("baseline_ada_ps_indist",), ("dual_u40_v2_ps_indist",)),
        "oodval": ("oodval", ("baseline_ada_ps_oodval",), ("dual_u40_v2_ps_ood",))}


def rows(*dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(f"{OUT}/{d}/*.json"):
            if f.endswith(("config.json", "metrics.json")):
                continue
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for r in data:
                    if "ade_rollout_k" in r:
                        out[r["clip_id"]] = r
    return out


def boot_ci(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="output dir prefix, e.g. minitron/mtw24_zs")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--ref", default=None,
                    help="another arm prefix to pair against too, e.g. minitron/mtw24_zs")
    args = ap.parse_args()
    for s in args.sets:
        suffix, base_d, moop_d = SETS[s]
        arm, base, moop = rows(f"{args.arm}_{suffix}"), rows(*base_d), rows(*moop_d)
        ext = [f"{EXT}{args.ref.split('/')[-1]}"] if args.ref and s == "test2000" else []
        ref = rows(f"{args.ref}_{suffix}", *ext) if args.ref else None
        ids = sorted(set(arm) & set(base) & set(moop) & (set(ref) if ref else set(arm)))
        if not ids:
            print(f"== {s}: no common clips")
            continue

        def at6(rr, key="ade_rollout_k", ids=ids):
            return np.array([min(rr[c][key][:6]) for c in ids])

        a, b, m = at6(arm), at6(base), at6(moop)
        refs = [("dense", b), ("MoOP zs", m)] + ([("ref", at6(ref))] if ref else [])
        deg = np.mean([bool(arm[c].get("coc_degenerate", False)) for c in ids])
        deg_m = np.mean([bool(moop[c].get("coc_degenerate", False)) for c in ids])
        print(f"== {s}: n={len(ids)} ({args.arm} rows {len(arm)})")
        for name, x in refs + [("arm", a)]:
            print(f"  {name:8s} minADE@6 mean {x.mean():.4f} median {np.median(x):.4f}")
        for name, x in refs:
            d = a - x
            lo, hi = boot_ci(d)
            print(f"  arm - {name:8s} mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"median {np.median(d):+.4f}")
        fde = [at6(r, "fde_rollout_k").mean() for r in (base, moop, arm)]
        print(f"  minFDE@6 mean: dense {fde[0]:.3f} MoOP {fde[1]:.3f} arm {fde[2]:.3f}"
              + (f" ref {at6(ref, 'fde_rollout_k').mean():.3f}" if ref else ""))
        print(f"  CoC degenerate: arm {deg:.1%}  MoOP {deg_m:.1%}"
              + (f"  ref {np.mean([bool(ref[c].get('coc_degenerate', False)) for c in ids]):.1%}"
                 if ref else ""))


if __name__ == "__main__":
    main()
