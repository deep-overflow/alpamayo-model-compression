"""Hand-built kept sets that isolate which mask difference carries a seed arm's damage.

`slim_meta.json` is a complete recipe (slim_lib.load_slim rebuilds the model from it), so
a variant is just two shipped metas spliced. Two cases from `analyze_seed_arms.py`:

  n1 vs n4    same clips, same CoC, FM noise seed 1 vs 4: n1 collapses on cruise (+1.21 m),
              n4 does not. n1 is the only one of 12 arms that drops layer-0 head 30.
  ref vs ada  the same estimator run twice: 2 Q heads (layers 0 and 9) + 316 MLP channels
              apart, +0.083 on test500.

For each pair (bad, good):
  <bad>_L0<good>   bad with layer 0's Q heads taken from good     necessity of the L0 swap
  <good>_L0<bad>   good with layer 0's Q heads taken from bad     sufficiency
  <bad>_Q<good>    bad's MLP channels + good's Q heads, all layers   } which axis
  <bad>_M<good>    bad's Q heads + good's MLP channels, all layers   }

Also writes the evaluation subset: the first N cruise clips of test_500 in manifest order
(a rule that does not look at any arm's result), as outputs/eval_sets/test_cruise<N>.parquet.

Usage:
  python experiments/evaluation/make_swap_masks.py [--n-clips 120]
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = {"n1": "slim_dual_seed_n1", "n4": "slim_dual_seed_n4",
       "ref": "slim_dual_seed_ref", "ada": "slim_dual_ada_u40_v2"}
PAIRS = [("n1", "n4"), ("ref", "ada")]            # (bad, good)


def splice(base, donor, layers=None, axes=("q",)):
    """base's meta with `axes` of `layers` (None = all) replaced by donor's."""
    m = copy.deepcopy(base)
    for li in range(len(m["vlm"])) if layers is None else layers:
        for ax in axes:
            m["vlm"][li][ax] = list(donor["vlm"][li][ax])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=120)
    args = ap.parse_args()
    out = REPO / "outputs"
    meta = {k: json.loads((out / d / "slim_meta.json").read_text()) for k, d in SRC.items()}

    variants = {}
    for bad, good in PAIRS:
        variants[f"{bad}_L0{good}"] = splice(meta[bad], meta[good], layers=[0])
        variants[f"{good}_L0{bad}"] = splice(meta[good], meta[bad], layers=[0])
        variants[f"{bad}_Q{good}"] = splice(meta[bad], meta[good], axes=("q",))
        variants[f"{bad}_M{good}"] = splice(meta[bad], meta[good], axes=("mlp",))
    for name, m in variants.items():
        # every variant must keep the budget: 19 Q heads and 7390 channels in every layer
        assert all(len(l["q"]) == 19 and len(l["mlp"]) == 7390 for l in m["vlm"]), name
        m["config"] = f"swap_{name}"
        d = out / f"slim_swap_{name}"
        d.mkdir(exist_ok=True)
        (d / "slim_meta.json").write_text(json.dumps(m))
        dq = {k: sum(len(set(a["q"]) ^ set(b["q"])) // 2
                     for a, b in zip(m["vlm"], meta[k]["vlm"])) for k in SRC}
        dm = {k: sum(len(set(a["mlp"]) ^ set(b["mlp"])) // 2
                     for a, b in zip(m["vlm"], meta[k]["vlm"])) for k in SRC}
        print(f"{name:10s} Q heads differing from {dq}  MLP {dm}")
    for bad, good in PAIRS:
        a, b = meta[bad]["vlm"][0]["q"], meta[good]["vlm"][0]["q"]
        print(f"layer 0: {bad} keeps {sorted(set(a) - set(b))} where {good} keeps "
              f"{sorted(set(b) - set(a))}")

    base = {}
    for p in sorted((out / "baseline_ada_ps_test").glob("baseline_s*of*.json")):
        for r in json.loads(p.read_text()):
            base[r["clip_id"]] = r["bucket"]
    df = pd.read_parquet(out / "eval_sets" / "test_500.parquet")
    sub = df[df.clip_id.map(base) == "cruise"].head(args.n_clips)
    assert len(sub) == args.n_clips
    path = out / "eval_sets" / f"test_cruise{args.n_clips}.parquet"
    sub.to_parquet(path)
    print(f"{len(sub)} cruise clips -> {path}")


if __name__ == "__main__":
    main()
