"""Draw calibration sets by rule from the action-labelled candidates -- no greedy matching.

plans/2026-09-11_action-stratified-calib.md. Three rules, all random within whatever they
condition on, so seeds give exchangeable replicates (the greedy series calib_100 -> nt_a ->
... was a preference order, not replicates):

  rd   uniform random over the candidates (no strata, no matching) -- the control.
       calib_rd100_a/b already exist from the same pool; this adds the third.
  se   stratified by bucket5, quota = test500 composition 56/13/15/9/7
  su   stratified by bucket5, quota = 20 per bucket

Sets are drawn in the order given and each removes its clips from the candidates, so all
new sets are disjoint from each other; they are asserted disjoint from every existing
calib_* manifest, the OOD pool and every evaluation set. t0 is CALIB_T0 for every clip,
because the bucket is a property of (clip, t0).

Writes to outputs/eval_sets/: calib_<rule>100_<letter>.parquet per set, the union
calib_strat<N>.parquet for build_cache.py, config/quality/summary_calib_strat<N>.*.

Usage:
  python experiments/evaluation/make_calib_strat.py --labels outputs/strat_calib/labels.parquet \
      --rules rd:c se:abc su:abc --seed 20260911
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from label_actions import AV, CALIB_T0, ES, EVAL_SETS, ORDER
from make_eval_sets import ATTR, derive, encode, weighted_l1

REPO = Path(__file__).resolve().parents[2]
N = 100
QUOTA = {
    "rd": None,
    "se": {"cruise": 56, "decel_stop": 13, "accel": 15, "turn_left": 9, "turn_right": 7},
    "su": {"cruise": 20, "decel_stop": 20, "accel": 20, "turn_left": 20, "turn_right": 20},
}
KEEP = ["clip_id", "t0_us", "chunk", "block", "bucket", "country", "platform_class",
        "radar_config", "month", "hour_of_day", "time_of_day", "season"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/strat_calib/labels.parquet")
    ap.add_argument("--rules", nargs="+", default=["rd:c", "se:abc", "su:abc"],
                    help="rule:letters, drawn in this order")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--exp-id", default="eval_sets")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id

    lab = pd.read_parquet(REPO / args.labels)
    assert (lab["t0_us"] == CALIB_T0).all(), "labels are not at CALIB_T0"
    for q in QUOTA.values():
        assert q is None or sum(q.values()) == N

    # everything already measured or evaluated is held out -- computed before anything is
    # written so the sets drawn here are not in their own exclusion list
    ood = set(pd.read_parquet(AV / "reasoning" / "ood_reasoning.parquet").index.astype(str))
    prior = {}
    for p in sorted(glob.glob(str(ES / "calib_*.parquet"))) + \
            [str(ES / f"{s}.parquet") for s in EVAL_SETS]:
        d = pd.read_parquet(p)
        col = "clip_id" if "clip_id" in d.columns else d.columns[0]
        prior[Path(p).stem] = set(d[col].astype(str))
    held = ood.union(*prior.values())
    cand = lab[~lab["clip_id"].isin(held)].reset_index(drop=True)
    print(f"candidates {len(lab)}  -held out {len(lab) - len(cand)}  -> {len(cand)}", flush=True)
    print("candidate composition: " + "  ".join(
        f"{b} {int(n)}" for b, n in cand["bucket"].value_counts().reindex(ORDER).items()))

    # the 6-attribute target is the full official train split, as make_eval_sets uses
    ci = pd.read_parquet(AV / "clip_index.parquet")
    dc = pd.read_parquet(AV / "metadata" / "data_collection.parquet")
    full = derive(ci[ci.split == "train"], dc)

    remaining = cand.copy()
    sets, quality = [], []
    for ri, spec in enumerate(args.rules):
        rule, letters = spec.split(":")
        quota = QUOTA[rule]
        for li, letter in enumerate(letters):
            rng = np.random.default_rng([args.seed, ri, li])
            if quota is None:
                idx = rng.choice(len(remaining), N, replace=False)
            else:
                idx = []
                for b, q in quota.items():
                    pool_b = np.flatnonzero(remaining["bucket"].to_numpy() == b)
                    assert len(pool_b) >= q, f"{rule}_{letter}: bucket {b} has {len(pool_b)} < {q}"
                    idx.extend(rng.choice(pool_b, q, replace=False).tolist())
                idx = np.array(idx)
            sel = remaining.iloc[idx].copy()
            sel["block"] = letter
            name = f"calib_{rule}{N}_{letter}"
            comp = sel["bucket"].value_counts().reindex(ORDER).fillna(0).astype(int)
            if quota is not None:
                assert comp.to_dict() == quota, f"{name}: realised {comp.to_dict()} != quota"
            codes, targets = encode(full, sel)
            counts = {a: np.bincount(codes[a], minlength=len(targets[a])).astype(float)
                      for a in ATTR}
            wl1 = weighted_l1(counts, N, targets)
            print(f"  {name}: seed [{args.seed},{ri},{li}]  6-attr weighted L1 {wl1:.4f}  "
                  f"buckets {comp.to_dict()}", flush=True)
            quality.append({"set": name, "rule": rule, "block": letter, "weighted_l1": wl1,
                            "countries": int(sel["country"].nunique()), **comp.to_dict()})
            sets.append((name, sel[KEEP]))
            remaining = remaining[~remaining["clip_id"].isin(sel["clip_id"])]

    union = pd.concat([s for _, s in sets], ignore_index=True)
    assert len(union) == len(set(union["clip_id"])) == N * len(sets), "new sets overlap"
    overlap = {k: len(set(union["clip_id"]) & v) for k, v in prior.items()}
    overlap["ood_reasoning"] = len(set(union["clip_id"]) & ood)
    assert not any(overlap.values()), f"new sets touch a held-out set: {overlap}"
    assert (union["t0_us"] == CALIB_T0).all()

    for name, s in sets:
        s.to_parquet(out_dir / f"{name}.parquet", index=False)
    union_name = f"calib_strat{len(union)}"
    union.to_parquet(out_dir / f"{union_name}.parquet", index=False)
    pd.DataFrame(quality).to_csv(out_dir / f"quality_{union_name}.csv", index=False)
    (out_dir / f"config_{union_name}.json").write_text(json.dumps({
        "purpose": "action-stratified calibration draws, 3 rules, no greedy matching",
        "plan": "plans/2026-09-11_action-stratified-calib.md",
        "labels": args.labels, "rules": args.rules, "quota": QUOTA, "n": N,
        "seed_rule": "np.random.default_rng([seed, rule_index, letter_index])",
        "seed": args.seed, "t0_us": CALIB_T0,
        "candidates": {"labelled": len(lab), "after_hold_out": len(cand)},
        "manifests": [n for n, _ in sets], "union": union_name,
        "held_out_sources": {k: len(v) for k, v in prior.items()},
        "overlap_with_held_out": overlap,
    }, indent=2))
    lines = [f"{len(sets)} x {N} calibration sets by rule, t0={CALIB_T0} us, seed {args.seed}",
             f"candidates {len(cand)} labelled official-train clips outside every existing set"]
    for q in quality:
        lines.append(f"  {q['set']}: L1 {q['weighted_l1']:.4f}  " +
                     "  ".join(f"{b} {q[b]}" for b in ORDER))
    lines.append("disjoint from each other, every calib_* manifest, the OOD pool and every "
                 "evaluation set (asserted)")
    (out_dir / f"summary_{union_name}.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {union_name} + {len(sets)} sets to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
