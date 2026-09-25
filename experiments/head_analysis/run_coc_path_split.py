"""Path split of I_CoC at generated-CoC positions: own-token path vs cross-position path.

plans/2026-09-25_coc-position-disagreement.md, test 1. At a generated-CoC position p a unit's
output reaches the CE loss two ways: down its own residual stream to the logits at p (the
own-token path, which nothing else in the model reads) and through the K/V written at p into
the later CoC queries (the cross-position path, the only kind of path the FM loss has). Two CE
backwards per clip through TypedUnitGates on the dense model's own rollout:

  full     the shipped CE backward (typed by the position the unit acts on)
  direct   the same with the K and V of every CoC position detached in every layer, so the
           CoC-position gate gradient carries the own-token path only
  cross    = full - direct (the backward is linear in the seed, so the split is exact up to
           bf16 rounding of a repeated forward)

Same clips, rollout seeds and typed gates as run_gradient_anatomy.py, so the CoC row of `full`
must reproduce that run's ce arrays (G0). Writes per-clip typed Q-head grads (5, L, H) for full
and direct, and for MLP channels the CoC row and the pooled sum (L, I).

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_coc_path_split.py --gpu 4 --exp-id coc_pathsplit_v1_s0 --shard 0 --n-shards 4
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

import analysis_lib as lib
import prune_lib as pl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")  # run_gradient_anatomy.TYPES
COC = TYPES.index("coc")


def token_types(model, seq_tf, prompt_len):
    sp = lib.compute_spans(model, seq_tf)
    idx = torch.full((seq_tf.shape[1],), TYPES.index("prompt_text"), dtype=torch.long)
    idx[sp["vision"]] = TYPES.index("vision")
    idx[sp["hist"]] = TYPES.index("hist")
    idx[sp["sink"]] = TYPES.index("sink")
    idx[prompt_len:] = COC
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--no-mlp", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sc.calib_samples(REPO, args.manifest)[: args.num_clips][args.shard :: args.n_shards]
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    model.vlm.enable_input_require_grads()
    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "plan": "plans/2026-09-25_coc-position-disagreement.md (test 1)", "manifest": args.manifest, "cache": args.cache,
        "num_clips": len(clips), "shard": [args.shard, args.n_shards], "seed": args.seed,
        "seed_from": "sample_cache.clip_seed(seed, clip_id), as run_gradient_anatomy", "types": TYPES,
        "direct": "K and V of every generated-CoC position detached in every layer during the CE backward",
        "gpu": torch.cuda.get_device_name(device)}, indent=2))

    per = {"q_full": [], "q_direct": [], "mlp_coc_full": [], "mlp_coc_direct": [], "mlp_pool_full": [], "mlp_pool_direct": []}
    rows, done = [], []

    def ce_backward(gates, inputs, seq_tf, coc_start, coc_end, detach_coc):
        handles = []
        if detach_coc:
            def hook(module, inp, out):
                o = out.clone()
                o[:, coc_start:coc_end] = out[:, coc_start:coc_end].detach()
                return o
            for layer in layers:
                handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
                handles.append(layer.self_attn.v_proj.register_forward_hook(hook))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _, _ = pl.vlm_forward_with_grad(model, seq_tf, inputs["tokenized_data"], use_cache=False)
            nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
        nll.backward()
        for h in handles:
            h.remove()
        q = gates.q_signed().astype(np.float32)  # (5, L, H)
        mlp = None if args.no_mlp else gates.mlp_signed().astype(np.float32)  # (5, L, I)
        gates.zero_grads()
        del hidden
        return float(nll), q, mlp

    for ci, (clip_id, clip_t0) in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        seed = sc.clip_seed(args.seed, clip_id)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end]
        del roll
        # the gates go on AFTER the rollout (their hooks index a fixed (T,) type vector, which a
        # growing generation cannot use) and come off after the clip, as in run_gradient_anatomy
        gates = pl.TypedUnitGates(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size,
                                  len(TYPES), "cuda", mlp=not args.no_mlp)
        gates.set_types(token_types(model, seq_tf, prompt_len).to("cuda"))
        nll_f, q_f, m_f = ce_backward(gates, inputs, seq_tf, coc_start, coc_end, False)
        nll_d, q_d, m_d = ce_backward(gates, inputs, seq_tf, coc_start, coc_end, True)
        gates.remove()
        del gates
        per["q_full"].append(q_f)
        per["q_direct"].append(q_d)
        if m_f is not None:
            per["mlp_coc_full"].append(m_f[COC])
            per["mlp_coc_direct"].append(m_d[COC])
            per["mlp_pool_full"].append(m_f.sum(0))
            per["mlp_pool_direct"].append(m_d.sum(0))
        share_q = float(np.abs(q_d[COC]).sum() / max(np.abs(q_f[COC]).sum(), 1e-30))
        add_q = float((q_d[COC] * np.sign(q_f[COC])).sum() / max(np.abs(q_f[COC]).sum(), 1e-30))
        rows.append({"clip_id": clip_id, "coc_len": int(coc_end - coc_start), "nll_full": nll_f, "nll_direct_forward": nll_d,
                     "q_coc_direct_abs_share": share_q, "q_coc_direct_additive_share": add_q})
        done.append(clip_id)
        print(f"[{ci + 1}/{len(clips)}] {clip_id} coc={coc_end - coc_start} nll={nll_f:.4f} (repeat {nll_d:.4f}) | "
              f"Q CoC-row grad: own-token |.| share {share_q:.3f}, additive share {add_q:+.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            np.savez(out_dir / "pathsplit_perclip.npz", **{k: np.stack(v) for k, v in per.items() if v})
            (out_dir / "metrics.json").write_text(json.dumps({"n_clips": len(done), "clip_ids": done, "per_clip": rows}, indent=1))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
