"""Do cvlab20 and cvlab21 agree clip for clip? (run AFTER re-measuring a baseline arm)

This project's determinism rule is that two runs match bitwise only within one GPU
architecture -- the same clip and seed gave 0.286 on Ada and 0.291 on Blackwell, and 3-4%
of clips produced different CoC text across architectures. cvlab20 is RTX 5880 Ada, the
same card as cvlab21's evaluation GPUs, so results *should* transfer. Same card model is
not the same software stack, though: a different driver or CUDA build can reorder
reductions, so this is measured rather than assumed before any arm is mixed across boxes.

  # on cvlab20, re-measure the arm cvlab21 already has
  . ~/project/chan/env.sh
  bash experiments/head_analysis/run_retry_host.sh 60 \
      experiments/evaluation/run_baseline.py --set test --model baseline \
      --exp-id baseline_cvlab20_test --gpu 0

  # bring the rows back, then here
  python parity_check.py --a outputs/baseline_ada_ps_test --b outputs/baseline_cvlab20_test

Verdict: identical minADE on every clip is a pass. Anything else means the two boxes'
numbers may not be pooled, and each comparison has to stay inside one box.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

K = 6


def load(d):
    rows = []
    for f in sorted(glob.glob(str(Path(d) / "*_s*of*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    return {r["clip_id"]: r for r in rows if "ade_rollout_k" in r}


def at_k(r, key):
    return float(np.min(np.asarray(r[key], dtype=float)[:K]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference run (cvlab21)")
    ap.add_argument("--b", required=True, help="the run to check (cvlab20)")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("no clips in common")
    print(f"{len(a)} vs {len(b)} rows, {len(ids)} in common")

    da = np.array([at_k(a[i], "ade_rollout_k") for i in ids])
    db = np.array([at_k(b[i], "ade_rollout_k") for i in ids])
    diff = np.abs(da - db)
    same_coc = np.mean([a[i].get("gen_coc") == b[i].get("gen_coc") for i in ids])

    print(f"minADE@{K}   a {da.mean():.6f}   b {db.mean():.6f}   "
          f"paired mean diff {(db - da).mean():+.6f}")
    print(f"clips differing at all: {(diff > 0).sum()}/{len(ids)}   "
          f"max |diff| {diff.max():.3e}")
    print(f"identical CoC text: {same_coc:.1%}")

    if diff.max() == 0.0 and same_coc == 1.0:
        print("PASS -- bitwise identical; the two boxes' runs can be pooled")
    elif diff.max() < 1e-3:
        print("MARGINAL -- not bitwise but far below the effects being measured; "
              "pool only with the difference stated")
    else:
        print("FAIL -- keep every comparison inside one box, as with Ada vs Blackwell")


if __name__ == "__main__":
    main()
