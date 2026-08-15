"""Bake a fake-quantized slim model into a real checkpoint directory.

Open-loop evaluation applies the bit spec at load time (`run_baseline.py --quant`), but
the alpasim driver has no such hook: its vendored `load_slim` reads `slim_meta.json` and
then *strict-loads* `slim_state.pt`. So a composed arm only reaches closed loop if the
quantized weights are materialized on disk.

Fake quant keeps dtype and shape, so the result is an ordinary slim checkpoint whose
weights happen to sit on the quantization grid -- the driver needs no change, and the
saved state_dict is the same 16.8 GB as any other slim checkpoint (the saving is
projected from the bit allocation, not realized in the file).

Usage:
  python experiments/evaluation/make_quant_ckpt.py \
      --slim outputs/slim_dual_u40_v2 --quant outputs/quant_specs/w4_all_dual_u40.npz \
      --out outputs/slim_dual_u40_w4 --gpu 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import quant_lib as ql
import slim_lib as sl
from expert_per_clip import reserve_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", required=True, help="source slim checkpoint dir")
    ap.add_argument("--quant", required=True, help="per-row bit spec .npz for THAT model")
    ap.add_argument("--parts", nargs="+",
                    default=["vlm_text", "lm_head", "vit", "expert"])
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    args = ap.parse_args()

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = sl.load_slim(REPO / args.slim, device="cuda")

    z = np.load(args.quant, allow_pickle=True)
    names = [str(n) for n in z["_names"]]
    spec = {n: torch.from_numpy(z[n]) for n in names}
    parts = tuple(args.parts)
    pool = ql.pool_shapes(model, parts=parts)
    bad = [(n, len(spec[n]), pool[n][0]) for n in names
           if n not in pool or len(spec[n]) != pool[n][0]]
    if bad:
        raise SystemExit(f"spec does not fit this model: {bad[:3]}")

    t0 = time.time()
    report = ql.apply_quant(model, spec, group=args.group, parts=parts)
    eff, tot_bits, tot_par = ql.effective_bits({n: pool[n] for n in names}, spec, args.group)
    print(f"quantized {len(report)} tensors in {time.time() - t0:.0f}s: "
          f"{eff:.3f} eff bits, pool {tot_par * 2 / 1e9:.2f} -> {tot_bits / 8 / 1e9:.2f} GB "
          f"(projected)", flush=True)
    sl.check_slim(model)

    meta = json.loads((REPO / args.slim / "slim_meta.json").read_text())
    out_dir = REPO / args.out
    t0 = time.time()
    sl.save_slim(model, meta, out_dir)
    print(f"saved in {time.time() - t0:.0f}s -> {out_dir}", flush=True)

    params = sl.n_params(model)
    (out_dir / "config.json").write_text(json.dumps({
        "kind": "slim + baked fake quantization",
        "slim_from": args.slim, "quant_spec": args.quant, "quant_parts": list(parts),
        "quant_group": args.group, "effective_bits": eff,
        "projected_pool_gb": tot_bits / 8 / 1e9,
        "projected_total_gb": tot_bits / 8 / 1e9 + (params - tot_par) * 2 / 1e9,
        "params": params, "model_revision": sl.MODEL_REV,
        "worst_rel_mse": max(r["rel_mse"] for r in report.values()),
        "mean_rel_mse": float(np.mean([r["rel_mse"] for r in report.values()])),
        "note": "weights are bf16 on a quantization grid; the file is full size and the "
                "storage saving is projected, not realized",
    }, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
