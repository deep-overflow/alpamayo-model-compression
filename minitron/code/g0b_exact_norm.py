"""G0b follow-up: is the G0b gap the reference's bf16 rounding, or a slicing error?

G0b (tolerance 1e-2, fixed before the run) FAILED on 2026-10-24: CoC argmax agreement
1.0000, but relative error 1.58e-2 (logits) / 1.77e-2 (K/V) / 3.3e-3 (expert velocity),
against 0.39 / 0.43 / 0.23 for the cut itself. That reference rescaled all 73 RMSNorm
weight vectors by sqrt(r) in bf16, rounding every weight by up to 2^-8 relative. Here the
reference keeps the original weights (removed channels zeroed) and instead divides the
RMSNorm variance by the kept count -- no weight is rounded, so the only difference left
to the physical slice is matmul accumulation order (length-4096 dot products with zeros
vs length-3328).

Reading, fixed before the run: if every relative error falls below the 1e-2 G0b tolerance,
the surgery is right and the G0b failure came from the reference; otherwise look for a
slicing error before anything is built on slice_hidden.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 1 \\
      minitron/code/g0b_exact_norm.py --gpu 5
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import types

import numpy as np
import torch

# g0_hidden_surgery puts the repo's experiment directories on sys.path, so the shared
# modules are taken from its namespace rather than imported here
from g0_hidden_surgery import (
    H_KEEP,
    KV_LAYERS,
    REPO,
    H,
    clips,
    compare,
    helper,
    load_dense,
    probe,
    reserve_gpu,
    set_determinism,
    sl,
)


def kv_by_layer(a, b):
    """Relative K/V error per probed cache layer (max over clips): does it grow with depth?"""
    return [max(float((x["kv"][i] - y["kv"][i]).norm() / y["kv"][i].norm()) for x, y in zip(a, b))
            for i in range(len(KV_LAYERS))]


def kept_mean_rmsnorm(self, hidden_states):
    """Qwen3VLTextRMSNorm.forward with the mean taken over the kept channels only."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).sum(-1, keepdim=True) / self.n_kept
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


@torch.no_grad()
def exact_reference(model, removed, n_kept):
    lm = model.vlm.model.language_model
    idx = torch.as_tensor(removed, dtype=torch.long, device=lm.embed_tokens.weight.device)

    def norm_(norm):
        norm.weight[idx] = 0
        norm.n_kept = n_kept
        norm.forward = types.MethodType(kept_mean_rmsnorm, norm)

    lm.embed_tokens.weight[:, idx] = 0
    for layer in lm.layers:
        norm_(layer.input_layernorm)
        norm_(layer.post_attention_layernorm)
        layer.self_attn.o_proj.weight[idx] = 0
        layer.mlp.down_proj.weight[idx] = 0
    norm_(lm.norm)
    vis = model.vlm.model.visual
    for merger in (vis.merger, *vis.deepstack_merger_list):
        merger.linear_fc2.weight[idx] = 0
        merger.linear_fc2.bias[idx] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="g0b_exact_norm")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-precision-reduction", action="store_true",
                    help="disable cuBLAS bf16 reduced-precision (split-K) reductions, so a "
                         "change of the dot-product length cannot change the rounding")
    args = ap.parse_args()
    if args.full_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    out_dir = REPO / "outputs" / "minitron" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    # same draw as g0_hidden_surgery.py, so the kept set matches G0b exactly
    rng = np.random.default_rng(args.seed)
    kept_h = np.sort(rng.choice(H, H_KEEP, replace=False))
    removed_h = np.setdiff1d(np.arange(H), kept_h)
    items = clips()

    # noise floor: the dense model against itself on the K/V-gather attention path
    # (bind_identity) -- mathematically identical, numerically a different computation
    model = load_dense()
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    dense = probe(model, processor, tok, items)
    sl.bind_identity(model)
    floor = compare(probe(model, processor, tok, items), dense)
    floor["kv_by_layer"] = kv_by_layer(probe(model, processor, tok, items), dense)
    del model
    torch.cuda.empty_cache()

    model = load_dense()
    exact_reference(model, removed_h, H_KEEP)
    ref = probe(model, processor, tok, items)
    del model
    torch.cuda.empty_cache()
    model = load_dense()
    sl.slice_hidden(model, kept_h)
    sliced = probe(model, processor, tok, items)
    res = compare(sliced, ref)
    res["kv_by_layer"] = kv_by_layer(sliced, ref)
    res["below_g0b_tolerance"] = (res["rel_logits"] < 1e-2 and res["rel_kv"] < 1e-2
                                  and res["rel_v"] < 1e-2)
    res["noise_floor"] = floor
    res["gpu"] = gpu_name
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))

    def fmt(r):
        layers = " ".join(f"L{li}:{e:.1e}" for li, e in zip(KV_LAYERS, r["kv_by_layer"]))
        return (f"argmax {r['argmax_agree']:.4f} rel logits {r['rel_logits']:.2e} "
                f"kv {r['rel_kv']:.2e} ({layers}) v {r['rel_v']:.2e}")

    lines = [(f"G0b exact-norm reference ({gpu_name}): {fmt(res)} "
              f"| below 1e-2: {res['below_g0b_tolerance']}"),
             f"noise floor (dense vs dense on the gather path): {fmt(floor)}"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
