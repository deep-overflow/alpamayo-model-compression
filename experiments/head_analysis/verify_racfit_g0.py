"""Gate G0: does run_racfit's `VT` mixture reproduce the shipped Tyr reconstruction?

plans/2026-08-25_cot-reconstruction.md. `VT` weights the vision and prompt-text
streams by their natural token counts and drops the decode stream, which is
exactly run_tyr_supernet's H ("full fused prompt prefill, no labels") up to a
global scale -- and the OSSCAR solve is scale invariant, so the level-0 solution
must agree. Layer 0 is the reference because it is solved before any
error-accumulation write-back, so the two pipelines see identical inputs there.

The supernet's per-level .pth files were deleted (outputs/tyr_supernet_u40 is 16 KB
now), so the reference is the surviving checkpoint slim_tyr_uniform_u40_recon,
whose slim_state.pt holds the same weights after slicing.

  1. produce the reference solution
     bash experiments/head_analysis/run_retry_host.sh 10 \
         experiments/head_analysis/run_racfit.py --gpu 0 --exp-id racfit_g0 \
         --num-clips 100 --k-seeds 1 --layer-end 1 --folds 1 --mixes VT \
         --keeps-q 19 --keeps-m 7390 --no-dual --dump-u40-mix VT
  2. python experiments/head_analysis/verify_racfit_g0.py --exp-id racfit_g0
"""

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
KEYS = {"o": "vlm.model.language_model.layers.{i}.self_attn.o_proj.weight",
        "m": "vlm.model.language_model.layers.{i}.mlp.down_proj.weight"}
HEAD_DIM = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="racfit_g0")
    ap.add_argument("--mix", default="VT")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ref", default="outputs/slim_tyr_uniform_u40_recon")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ref_dir = REPO / args.ref
    meta = json.loads((ref_dir / "slim_meta.json").read_text())
    sd = torch.load(ref_dir / "slim_state.pt", map_location="cpu", weights_only=True,
                    mmap=True)
    kept_ref = {"o": meta["vlm"][args.layer]["q"], "m": meta["vlm"][args.layer]["mlp"]}

    ok = True
    for tag in ("o", "m"):
        sol = torch.load(REPO / "outputs" / args.exp_id / "sol" /
                         f"L{args.layer:02d}_{tag}_{args.mix}.pt",
                         map_location="cpu", weights_only=True).float()  # (out, in) dense
        gs = HEAD_DIM if tag == "o" else 1
        n_groups = sol.shape[1] // gs
        alive = sol.abs().sum(0).reshape(n_groups, gs).sum(1) != 0
        kept_ours = torch.nonzero(alive).flatten().tolist()
        same = kept_ours == kept_ref[tag]
        inter = len(set(kept_ours) & set(kept_ref[tag])) / max(len(kept_ref[tag]), 1)

        cols = torch.tensor([g * gs + d for g in kept_ours for d in range(gs)])
        ours = sol[:, cols]                                    # (out, kept*gs)
        theirs = sd[KEYS[tag].format(i=args.layer)].float()     # (out, kept*gs)
        if ours.shape != theirs.shape:
            print(f"{tag}: SHAPE MISMATCH ours {tuple(ours.shape)} "
                  f"ref {tuple(theirs.shape)}")
            ok = False
            continue
        rel = float((ours - theirs).norm() / theirs.norm())
        passed = same and rel < args.tol
        ok &= passed
        print(f"{tag}: kept sets {'identical' if same else f'differ (overlap {inter:.3f})'}"
              f"   rel Frobenius diff {rel:.3e}   -> {'PASS' if passed else 'FAIL'}")

    print(f"G0 VERDICT: {'PASS' if ok else 'FAIL'} (tol {args.tol})")


if __name__ == "__main__":
    main()
