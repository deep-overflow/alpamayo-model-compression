"""Cache-Jacobian importance: how much does removing a VLM unit move the KV cache the
expert is sensitive to?  (plans/2026-08-30_cache-jlens-criterion.md)

The first-order Taylor score of a cache-reconstruction loss is zero at the unpruned
point, so this measures the second-order quantity directly:

  I_cache(u) = sum_{l,g} S[l,g] * ( ||dK_{l,g}/dg_u||^2 + ||dV_{l,g}/dg_u||^2 )

with S the per-(layer, KV group) sensitivity from the cache-use map (Stage C swap damage /
shift; layer 0 is 0). Per-unit backwards are impossible (36 x 12320 units), so it uses
random probes: seed the cache with r_{l,g} ~ N(0, S[l,g] / (T*D)) per element and backprop
<sum r, cache> once through prune_lib.UnitGates -- the SAME path run_importance uses for
I_traj with the expert's cache gradient as the seed (prune_lib.vlm_backward_from_cache).
E[(sum_l d<r_l, cache_l>/dg_u)^2] = sum_l S_l ||d cache_l / dg_u||^2 because the probes are
independent and zero-mean, so squaring each probe's gate gradient and averaging over
probes and clips is an unbiased estimate. Label-free: no rollout, no GT trajectory -- the
prompt-only prefill cache is what the expert reads (99.5% of positions).

Writes outputs/<exp-id>/importance.npz with cache_vlm_q (36, 32), cache_vlm_mlp (36, 12288),
their even/odd-clip halves (split-half stability), per-probe variance, and metrics.json.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 60 \
      experiments/head_analysis/run_cache_jlens.py --gpu 6 --exp-id cachejlens_v1
  # smoke: --num-clips 2 --probes 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import analysis_lib as lib  # noqa: E402
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from slim_lib import MODEL_REV  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cachejlens_v1")
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--sensitivity", default="cacheuse_v1/maps_swap.npz",
                    help="npz with `sensitivity` (36, 8); 'uniform' weights every group 1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.vlm.enable_input_require_grads()  # the cache tensors need a graph (prune_lib)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    tc = model.vlm.config.text_config
    L, H, I = tc.num_hidden_layers, tc.num_attention_heads, tc.intermediate_size
    G = tc.num_key_value_heads

    if args.sensitivity == "uniform":
        S = np.ones((L, G))
    else:
        S = np.nan_to_num(np.load(REPO / "outputs" / args.sensitivity)["sensitivity"])  # (36, 8)
    S = S / S.sum()  # overall scale is irrelevant to the ranking; keep numbers O(1)
    S_t = torch.tensor(S, device="cuda", dtype=torch.float32)

    acc = {k: np.zeros(shape) for k, shape in (("q", (L, H)), ("mlp", (L, I)))}
    acc_half = {p: {k: np.zeros(shape) for k, shape in (("q", (L, H)), ("mlp", (L, I)))}
                for p in (0, 1)}
    sq = {k: np.zeros(shape) for k, shape in (("q", (L, H)), ("mlp", (L, I)))}  # for variance
    n_probe_total = 0
    records = []
    t_all = time.time()
    gen = torch.Generator(device="cuda")
    for ci, (clip_id, t0_us) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        ids = inputs["input_ids"]
        gates = pl.UnitGates(model.vlm.model.language_model.layers, H, tc.head_dim, I,
                             "cuda", torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, cache, _ = pl.vlm_forward_with_grad(model, ids, inputs["tokenized_data"],
                                                    use_cache=True)
        cache_t = pl.retain_cache_grads(cache, L)
        T = cache.get_seq_length()
        gen.manual_seed(sc.clip_seed(args.seed, clip_id))
        for pi in range(args.probes):
            seeds = []
            for li, (k, v) in enumerate(cache_t):
                # (1, G, T, D): N(0, S[l,g] / (T*D)) per element -> sum over elements of
                # r^2 * J^2 has expectation S[l,g] * mean_{t,d} J^2
                std = (S_t[li] / (T * k.shape[-1])).sqrt().view(1, G, 1, 1)
                rk = torch.randn(k.shape, device="cuda", generator=gen,
                                 dtype=torch.float32) * std
                rv = torch.randn(v.shape, device="cuda", generator=gen,
                                 dtype=torch.float32) * std
                seeds.append((rk, rv))
            pl.vlm_backward_from_cache(cache_t, seeds, retain=pi < args.probes - 1)
            # a gate in layer l only reaches the caches of layers > l (K/V are projections
            # of the layer INPUT), so the last layer's gates get no gradient at all: their
            # cache importance is exactly zero and the CoC half alone ranks them
            gq = np.stack([(g.grad.float() if g.grad is not None else torch.zeros_like(g))
                           .cpu().numpy() for g in gates.q_gates])  # (L, H)
            gm = np.stack([(g.grad.float() if g.grad is not None else torch.zeros_like(g))
                           .cpu().numpy() for g in gates.mlp_gates])  # (L, I)
            for k, g2 in (("q", gq ** 2), ("mlp", gm ** 2)):
                acc[k] += g2
                acc_half[ci % 2][k] += g2
                sq[k] += g2 ** 2
            n_probe_total += 1
            gates.zero_grads()
            for k, v in cache_t:
                k.grad = None
                v.grad = None
        gates.remove()
        del cache, cache_t
        torch.cuda.empty_cache()
        records.append({"clip_id": clip_id, "prefill": T, "seconds": time.time() - t0})
        print(f"[{ci + 1}/{len(calib)}] {clip_id[:8]} T={T} probes={args.probes} "
              f"({time.time() - t0:.1f}s) q-mass top layer {int(acc['q'].sum(1).argmax())}",
              flush=True)
        if (ci + 1) % 10 == 0 or ci + 1 == len(calib):
            n = n_probe_total
            np.savez(out_dir / "importance.npz",
                     cache_vlm_q=acc["q"] / n, cache_vlm_mlp=acc["mlp"] / n,
                     cache_vlm_q_even=acc_half[0]["q"] / max(n / 2, 1),
                     cache_vlm_mlp_even=acc_half[0]["mlp"] / max(n / 2, 1),
                     cache_vlm_q_odd=acc_half[1]["q"] / max(n / 2, 1),
                     cache_vlm_mlp_odd=acc_half[1]["mlp"] / max(n / 2, 1),
                     cache_vlm_q_var=sq["q"] / n - (acc["q"] / n) ** 2,
                     cache_vlm_mlp_var=sq["mlp"] / n - (acc["mlp"] / n) ** 2,
                     sensitivity=S, n_probes=n)
            (out_dir / "metrics.json").write_text(json.dumps({
                "model_revision": MODEL_REV,
                "plan": "plans/2026-08-30_cache-jlens-criterion.md",
                "n_clips": ci + 1, "probes_per_clip": args.probes, "n_probes": n,
                "sensitivity": args.sensitivity, "seed": args.seed,
                "gpu": torch.cuda.get_device_name(device),
                "criterion": ("E_probe (d<r,cache>/dg)^2 with r ~ N(0, S[l,g]/(T*D)) "
                              "= sum_l S_l ||dcache_l/dg||^2"),
                "records": records, "seconds": time.time() - t_all}, indent=1))
    print("done ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
