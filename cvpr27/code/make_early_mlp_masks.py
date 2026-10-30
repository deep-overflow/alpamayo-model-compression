"""Keep-masks for the deeper MLP cut in VLM layers 1-4, built on top of l0keep.

Plan: cvpr27/plans/2026-10-29_early-mlp-cut.md (sections 2 and 3). Every config starts from B0 =
outputs/masks_u40_l0keep/dual.npz (the selection of slim_dual_u40_l0keep_v2) and changes only the
MLP of layers 1-4; Q heads, the other layers' MLP, the expert and KV stay B0's.

  b0          B0 as is
  e10 / e05   layers 1-4 keep the top 1229 / 614 channels by the dual score (= arms A10 / A05)
  r10         layers 1-4 keep 1229 channels drawn at random from B0's kept set (seed 0)
  t64         layers 1-4 drop the top 64 channels by the dual score from B0
  r64         layers 1-4 drop 64 channels drawn at random from B0's kept set (seed 0)
  e05_l<N>    only layer N keeps its top 614, the other three stay B0

The dual score is make_slim's: max(rank_norm I_traj, rank_norm I_CoC) per layer from importance_v2.
B0's kept set in layers 1-4 is exactly its top 7359 by that score (checked here), so every e/t
config is nested in B0.

Writes outputs/<exp-id>/m_<name>/masks.npz with both key conventions: q_mask / mlp_mask for
run_token_ablation.py --arm-masks, qhead / mlp for make_slim.py --config maskfile_<exp-id>/m_<name>.
Plus outputs/<exp-id>/masks_config.json with the per-config counts and removed parameters.

Usage:
  .venv/bin/python cvpr27/code/make_early_mlp_masks.py [--exp-id cvpr27/early_mlp]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
from run_cocsafe import rank_norm

LAYERS = (1, 2, 3, 4)
FULL_PARAMS = 11_078_526_194
P_Q, P_CH = 2 * 4096 * 128, 3 * 4096  # params per VLM Q head (q_proj rows + o_proj cols), per MLP channel
B0_REMOVED = 2_657_447_936            # slim_dual_u40_l0keep_v2 params.removed


def removed_params(q, m):
    """q: (36, 32), m: (36, 12288) keep masks -> VLM params removed."""
    return int((q.size - q.sum()) * P_Q + (m.size - m.sum()) * P_CH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cvpr27/early_mlp")
    ap.add_argument("--base", default="outputs/masks_u40_l0keep/dual.npz")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = np.load(REPO / args.base)
    q0 = b["q_mask"].astype(np.float32)    # (36, 32)
    m0 = b["mlp_mask"].astype(np.float32)  # (36, 12288)
    imp = np.load(REPO / "outputs" / args.importance / "importance.npz")
    score = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))  # (36, 12288)
    assert removed_params(q0, m0) == B0_REMOVED, removed_params(q0, m0)
    order = {li: np.argsort(score[li])[::-1] for li in LAYERS}  # descending dual score
    for li in LAYERS:
        n = int(m0[li].sum())
        assert set(order[li][:n]) == set(np.flatnonzero(m0[li])), f"layer {li}: B0 is not its top {n}"
    rng = np.random.default_rng(args.seed)

    def keep_top(m, li, k):
        m[li] = 0.0
        m[li, order[li][:k]] = 1.0

    configs = {"b0": m0.copy()}
    for name, k in (("e10", 1229), ("e05", 614)):
        m = m0.copy()
        for li in LAYERS:
            keep_top(m, li, k)
        configs[name] = m
    m = m0.copy()
    for li in LAYERS:
        kept = np.flatnonzero(m0[li])
        m[li] = 0.0
        m[li, rng.choice(kept, 1229, replace=False)] = 1.0
    configs["r10"] = m
    m = m0.copy()
    for li in LAYERS:
        m[li, order[li][:64]] = 0.0
    configs["t64"] = m
    m = m0.copy()
    for li in LAYERS:
        m[li, rng.choice(np.flatnonzero(m0[li]), 64, replace=False)] = 0.0
    configs["r64"] = m
    for li in LAYERS:
        m = m0.copy()
        keep_top(m, li, 614)
        configs[f"e05_l{li}"] = m

    out = REPO / "outputs" / args.exp_id
    meta = {"plan": "cvpr27/plans/2026-10-29_early-mlp-cut.md", "base": args.base,
            "importance": args.importance, "seed": args.seed, "layers": list(LAYERS), "configs": {}}
    for name, m in configs.items():
        # nothing outside layers 1-4's MLP may differ from B0
        other = [li for li in range(36) if li not in LAYERS]
        assert np.array_equal(m[other], m0[other]), name
        nested = all(set(np.flatnonzero(m[li])) <= set(np.flatnonzero(m0[li])) for li in LAYERS)
        assert nested, name
        rem = removed_params(q0, m)
        d = out / f"m_{name}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "masks.npz", q_mask=q0, mlp_mask=m, qhead=q0, mlp=m)
        meta["configs"][name] = {
            "path": str((d / "masks.npz").relative_to(REPO)),
            "mlp_kept_layers_1_4": [int(m[li].sum()) for li in LAYERS],
            "removed": rem, "removed_pct": round(rem / FULL_PARAMS * 100, 3),
            "extra_vs_b0": rem - B0_REMOVED, "extra_over_b0_removed": round((rem - B0_REMOVED) / B0_REMOVED, 4),
        }
        print(f"{name:8s} kept L1-4 {meta['configs'][name]['mlp_kept_layers_1_4']} removed {rem:,} "
              f"({rem / FULL_PARAMS * 100:.2f}%), extra {rem - B0_REMOVED:+,}")
    (out / "masks_config.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
