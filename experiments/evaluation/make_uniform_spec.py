"""Write a uniform-bit quantization spec over any subset of the pool.

`alloc_lib` emits a uniform spec as a by-product of a guided allocation, but only over
the parts that allocation used (VLM text + lm_head + ViT). The expert tower is excluded
there because a CoC-only criterion cannot score it -- an argument about the criterion,
not about quantizability. A uniform budget has no criterion, so it can cover the expert,
which matters: the expert is 2.279 B of the 2.921 B that otherwise stays bf16, and
moving it to 4 bits saves more storage than 24% structured pruning does.

Usage:
  python experiments/evaluation/make_uniform_spec.py --bits 4 --parts vlm_text lm_head vit expert
  python experiments/evaluation/make_uniform_spec.py --bits 8 --tag uniform_w8_all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import quant_lib as ql
import slim_lib as sl
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # noqa: F401  installs the gated-repo hub patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True, choices=list(ql.BITS))
    ap.add_argument("--parts", nargs="+",
                    default=["vlm_text", "lm_head", "vit", "expert"])
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--model", default="baseline",
                    help="'baseline' or a slim ckpt dir -- shapes come from the model the "
                         "spec will be applied to, so a pruned model needs its own spec")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="outputs/quant_specs")
    args = ap.parse_args()

    if args.model == "baseline":
        model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B",
                                            revision=sl.MODEL_REV, dtype=torch.bfloat16)
    else:
        model = sl.load_slim(REPO / args.model, device="cpu")
    shapes = ql.pool_shapes(model, parts=tuple(args.parts))
    names = sorted(shapes)
    spec = {n: np.full(shapes[n][0], args.bits, dtype=np.int64) for n in names}
    eff, tot_bits, tot_par = ql.effective_bits(shapes, {n: torch.from_numpy(spec[n])
                                                        for n in names}, args.group)

    tag = args.tag or f"uniform_w{args.bits}_" + "_".join(sorted(args.parts))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}.npz", **spec, _names=np.array(names, dtype=object))
    (out / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "bits": args.bits, "parts": args.parts, "group": args.group,
        "model": args.model, "n_tensors": len(names), "pool_params": tot_par,
        "effective_bits": eff, "pool_bf16_gb": tot_par * 2 / 1e9,
        "projected_pool_gb": tot_bits / 8 / 1e9,
    }, indent=2))
    print(f"{tag}: {len(names)} tensors, {tot_par / 1e9:.3f}B params, "
          f"{eff:.3f} eff bits, {tot_par * 2 / 1e9:.2f} -> {tot_bits / 8 / 1e9:.2f} GB "
          f"(projected)")


if __name__ == "__main__":
    main()
