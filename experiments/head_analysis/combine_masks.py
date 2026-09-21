"""One arm's keep mask below a cut layer, another's (or a random one) from it on.

plans/2026-09-21_importance-causal-validation.md, A2-mix: the trajectory-only mask applied in
layers 0-21 alone costs the action more than the same arm's full mask, so its late-layer
pruning repairs what its trunk pruning breaks. Crossing the trunk of one arm with the late
layers of another asks which late units do the repairing.

Usage:
  python experiments/head_analysis/combine_masks.py --trunk outputs/masks_u40_v2/traj.npz \
      --late outputs/masks_u40_v2/coc.npz --out outputs/masks_u40_v2/traj_trunk__coc_late.npz
  ... --late random:0   keeps, per late layer, as many random units as the trunk arm keeps there
"""

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", required=True, help="npz with q_mask/mlp_mask, used in layers < --cut")
    ap.add_argument("--late", required=True, help="npz used in layers >= --cut, or random:SEED")
    ap.add_argument("--cut", type=int, default=22)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = np.load(REPO / args.trunk)
    out = {}
    for k in ("q_mask", "mlp_mask"):
        m = a[k].astype(np.float32).copy()  # (36, U)
        if args.late.startswith("random:"):
            rng = np.random.default_rng([int(args.late.split(":")[1]), m.shape[1]])
            for li in range(args.cut, len(m)):
                keep = rng.choice(m.shape[1], size=int(a[k][li].sum()), replace=False)
                m[li] = 0.0
                m[li, keep] = 1.0
        else:
            m[args.cut:] = np.load(REPO / args.late)[k][args.cut:]
        out[k] = m
    path = REPO / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"{args.trunk} below layer {args.cut} + {args.late} from it on: q keep {out['q_mask'].mean():.4f}, "
          f"mlp keep {out['mlp_mask'].mean():.4f} -> {path}")


if __name__ == "__main__":
    main()
