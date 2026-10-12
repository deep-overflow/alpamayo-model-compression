"""Layer-0 Q-head variants: does the head choice in layer 0 explain collapses in general?

`make_swap_masks.py` found that a single layer-0 head carries n1's collapse and all of
ref's +0.083. Two follow-ups, both as spliced slim_meta.json:

  nt<x>_L0ship      calibration block x's kept set with layer 0's Q heads taken from the
                    shipped arm: are the bad-draw collapses (nt_c/d/e) the same thing?
  ship_L0drop<h>    the shipped kept set with layer-0 head h dropped and head 20 restored
                    in its place (20 is the head n1 kept instead of 30, harmless on its own
                    account in every good arm): one variant per kept layer-0 head, i.e. a
                    single-head sensitivity map of layer 0 under the shipped mask.

Also writes test_cruise<N>.parquet as a prefix of the 120-clip subset.

Usage:
  python experiments/evaluation/make_l0_masks.py [--n-clips 60]
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SPARE = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=60)
    args = ap.parse_args()
    out = REPO / "outputs"

    def meta(d):
        return json.loads((out / d / "slim_meta.json").read_text())

    ship = meta("slim_dual_u40_v2")
    variants = {}
    for x in "abcde":
        m = copy.deepcopy(meta(f"slim_dual_nt_{x}"))
        print(f"nt_{x} layer 0 drops {sorted(set(range(32)) - set(m['vlm'][0]['q']))}")
        m["vlm"][0]["q"] = list(ship["vlm"][0]["q"])
        variants[f"nt{x}_L0ship"] = m
    kept0 = sorted(ship["vlm"][0]["q"])
    assert SPARE not in kept0
    print(f"shipped layer 0 keeps {kept0}")
    for h in kept0:
        m = copy.deepcopy(ship)
        m["vlm"][0]["q"] = sorted([q for q in kept0 if q != h] + [SPARE])
        variants[f"ship_L0drop{h}"] = m
    for name, m in variants.items():
        assert all(len(l["q"]) == 19 and len(l["mlp"]) == 7390 for l in m["vlm"]), name
        m["config"] = f"swap_{name}"
        d = out / f"slim_swap_{name}"
        d.mkdir(exist_ok=True)
        (d / "slim_meta.json").write_text(json.dumps(m))
    print(f"{len(variants)} variants written")

    big = pd.read_parquet(out / "eval_sets" / "test_cruise120.parquet")
    path = out / "eval_sets" / f"test_cruise{args.n_clips}.parquet"
    big.head(args.n_clips).to_parquet(path)
    print(f"{args.n_clips} clips -> {path}")


if __name__ == "__main__":
    main()
