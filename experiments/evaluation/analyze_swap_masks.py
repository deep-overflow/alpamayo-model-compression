"""Which mask difference carries the damage? Reads `launch_swap_masks.sh` runs.

Every variant is evaluated on the test500 cruise subset with the arms' own seeds, so the
stored per-clip results of the source arms (n1, n4, ref, dual_ada) and of the unpruned
model pair with them clip by clip. For a (bad, good) pair the reading is the share of the
bad-minus-good gap a splice closes or opens:

  <bad>_L0<good>   closes it       -> the layer-0 head swap is necessary
  <good>_L0<bad>   opens it        -> and sufficient
  <bad>_Q<good> / <bad>_M<good>    -> which axis carries the rest

G0: `g0n1` is the n1 checkpoint re-run on the subset and must reproduce its stored
test500 rows bitwise.

Usage:
  python experiments/evaluation/analyze_swap_masks.py [--manifest test_cruise120]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
STORED = {"unpruned": "baseline_ada_ps_test", "n1": "dual_seed_n1_test",
          "n4": "dual_seed_n4_test", "ref": "dual_seed_ref_test",
          "ada": "dual_ada_u40_v2_test"}
PAIRS = [("n1", "n4"), ("ref", "ada")]
BOOT, K = 20000, 6


def load_run(exp):
    rows = {}
    for p in sorted((REPO / "outputs" / exp).glob("*.json")):
        if not p.name.startswith("config"):
            for r in json.loads(p.read_text()):
                rows[r["clip_id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="test_cruise120")
    ap.add_argument("--out", default="outputs/swap_masks")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = args.manifest.replace("test_", "")
    ids = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet").clip_id.tolist()
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    def vec(rows):
        return np.array([min(rows[c]["ade_rollout_k"][:K]) for c in ids])

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        p = wilcoxon(nz).pvalue if len(nz) > 10 else 1.0
        return float(d.mean()), float(lo), float(hi), float(np.median(d)), float(p)

    R = {k: vec(load_run(e)) for k, e in STORED.items()}
    raw = {k: load_run(e) for k, e in STORED.items()}
    names = [f"{a}_{op}{b}" for bad, good in PAIRS
             for a, op, b in ((bad, "L0", good), (good, "L0", bad), (bad, "Q", good), (bad, "M", good))]
    for n in names + ["g0n1"]:
        rows = load_run(f"swap_{n}_{tag}")
        if all(c in rows for c in ids):
            R[n], raw[n] = vec(rows), rows
        else:
            say(f"missing {n} ({len(rows)}/{len(ids)})")
    metrics = {"n_clips": len(ids), "manifest": args.manifest, "arms": {}, "pairs": {}}
    if "g0n1" in R:
        same = sum(raw["g0n1"][c]["ade_rollout_k"] == raw["n1"][c]["ade_rollout_k"]
                   and raw["g0n1"][c]["gen_coc"] == raw["n1"][c]["gen_coc"] for c in ids)
        metrics["g0_bit_identical"] = [same, len(ids)]
        say(f"G0: n1 re-run reproduces its stored rows on {same}/{len(ids)} clips")

    say(f"\n== {len(ids)} cruise clips of test500, minADE@6")
    say(f"{'model':12s}{'mean':>8s}{'median':>8s}{'degen':>7s}   delta vs unpruned (mean [95% CI], median)")
    for n, v in R.items():
        if n == "g0n1":
            continue
        m, lo, hi, med, _ = ci(v - R["unpruned"])
        deg = float(np.mean([raw[n][c]["coc_degenerate"] for c in ids]))
        metrics["arms"][n] = {"mean": float(v.mean()), "median": float(np.median(v)),
                              "degen": deg, "vs_unpruned": [m, lo, hi, med]}
        say(f"{n:12s}{v.mean():8.4f}{np.median(v):8.4f}{deg:7.3f}   "
            f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]  med {med:+.4f}")

    for bad, good in PAIRS:
        gap = R[bad] - R[good]
        g = ci(gap)
        say(f"\n== {bad} (bad) vs {good} (good): gap {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] "
            f"median {g[3]:+.4f} p={g[4]:.2g}")
        pr = {"gap": g, "variants": {}}
        for n, ref_side, what in (
                (f"{bad}_L0{good}", bad, "bad with good's layer-0 heads"),
                (f"{good}_L0{bad}", good, "good with bad's layer-0 heads"),
                (f"{bad}_Q{good}", bad, "bad's MLP + good's Q heads"),
                (f"{bad}_M{good}", bad, "bad's Q heads + good's MLP")):
            if n not in R:
                continue
            vs_bad, vs_good = ci(R[n] - R[bad]), ci(R[n] - R[good])
            # share of the gap this variant still carries, 0 = as good as good, 1 = as bad as bad
            pos = float((R[n] - R[good]).mean() / gap.mean())
            pr["variants"][n] = {"vs_bad": vs_bad, "vs_good": vs_good, "position": pos}
            say(f"{n:11s} {what:32s} mean {R[n].mean():.4f}  "
                f"vs {bad} {vs_bad[0]:+.4f} [{vs_bad[1]:+.4f}, {vs_bad[2]:+.4f}] p={vs_bad[4]:.2g}  "
                f"vs {good} {vs_good[0]:+.4f} [{vs_good[1]:+.4f}, {vs_good[2]:+.4f}] p={vs_good[4]:.2g}  "
                f"position {pos:+.2f}")
        metrics["pairs"][f"{bad}_vs_{good}"] = pr
    say("\nposition: 0 = lands on the good arm, 1 = lands on the bad arm (share of the gap carried)")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({"stored": STORED, "pairs": PAIRS,
                                                  "manifest": args.manifest, "k": K}, indent=2))


if __name__ == "__main__":
    main()
