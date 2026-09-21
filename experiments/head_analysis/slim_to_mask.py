"""slim_meta.json -> the q_mask / mlp_mask npz that the --mask runners read.

run_importance.py, run_jlens.py and run_gradient_anatomy.py take a pruned VLM as a 0/1 keep
mask on the full model. A slim build that only slices (no OSSCAR rewrite, expert and KV
left whole) IS that mask, and this writes it. It refuses a build the mask would not
reproduce: the parameters the mask removes must be all the parameters the build removed.

Usage:
  python experiments/head_analysis/slim_to_mask.py --slim outputs/slim_dual_u40_v2 \
      --out outputs/masks_u40_v2/dual.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
N_HEADS, HEAD_DIM, INTERMEDIATE, HIDDEN = 32, 128, 12288, 4096
REWRITTEN = ("tyr_u40", "dualr_u40", "dualgr_u40", "dualrc_u40")  # make_slim: o_proj/down_proj rewritten


def kept_masks(meta):
    """slim_meta -> q (L, 32), mlp (L, 12288) float32 0/1 keep masks."""
    n = len(meta["vlm"])
    q, mlp = np.zeros((n, N_HEADS), np.float32), np.zeros((n, INTERMEDIATE), np.float32)
    for li, e in enumerate(meta["vlm"]):
        q[li, e["q"]] = 1.0
        mlp[li, e["mlp"]] = 1.0
    return q, mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", required=True, help="slim checkpoint dir, repo-relative")
    ap.add_argument("--out", required=True, help="npz to write, repo-relative")
    args = ap.parse_args()

    meta = json.loads((REPO / args.slim / "slim_meta.json").read_text())
    cfg = str(meta.get("config", ""))
    if cfg.startswith(REWRITTEN):
        raise SystemExit(f"{cfg} rewrites o_proj/down_proj: a mask on the full model is not this build")
    if meta["kvonly_layers"]:
        raise SystemExit(f"{cfg} has kv-only layers {meta['kvonly_layers']}: not expressible as a unit mask")
    q, mlp = kept_masks(meta)
    # q_proj rows + o_proj columns per head; gate/up rows + down columns per channel
    removed = int(((1 - q).sum() * HEAD_DIM * HIDDEN * 2) + ((1 - mlp).sum() * HIDDEN * 3))
    if removed != meta["params"]["removed"]:
        raise SystemExit(f"the VLM mask removes {removed:,} parameters but the build removed "
                         f"{meta['params']['removed']:,}: something outside the VLM units was cut")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, q_mask=q, mlp_mask=mlp)
    print(f"{cfg}: q keep {q.mean():.4f} ({int(q.sum(1).min())}-{int(q.sum(1).max())} heads per layer), "
          f"mlp keep {mlp.mean():.4f}, {removed:,} parameters removed -> {out}")


if __name__ == "__main__":
    main()
