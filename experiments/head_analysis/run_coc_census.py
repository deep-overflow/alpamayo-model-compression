"""Where do generated-CoC queries put their attention, per head and layer?

plans/2026-09-25_coc-position-disagreement.md, test 2. Twin of run_vision_census.py with the
query rows moved to the generated-CoC positions and the keys grouped by token type:

  sink | vision (all cameras) | ego-history | prompt text | earlier CoC tokens | self

Same clips (calib_100 from the sample cache), same per-clip rollout seeds
(sample_cache.clip_seed) as run_gradient_anatomy.py, so the heads that the two scores rank
differently at CoC positions (analyze_coc_position_heads.py) are looked up on the same text.
Eager attention, one layer's probabilities at a time; only the (H, n_coc, T) query rows are
summed, nothing else is kept. Writes census.npz with mass_by_clip (N, L, H, 6) and metrics.json.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_coc_census.py --gpu 4 --exp-id coc_census_v1_s0 --shard 0 --n-shards 2
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
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu

REPO = Path(__file__).resolve().parents[2]
GROUPS = ("sink", "vision", "hist", "prompt", "coc_prev", "self")
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")  # run_gradient_anatomy.TYPES


def token_types(model, seq_tf, prompt_len):
    """(T,) long: index into TYPES for every position of the teacher-forced sequence
    (run_gradient_anatomy.token_types)."""
    sp = lib.compute_spans(model, seq_tf)
    idx = torch.full((seq_tf.shape[1],), TYPES.index("prompt_text"), dtype=torch.long)
    idx[sp["vision"]] = TYPES.index("vision")
    idx[sp["hist"]] = TYPES.index("hist")
    idx[sp["sink"]] = TYPES.index("sink")
    idx[prompt_len:] = TYPES.index("coc")
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
    ap.add_argument("--reserve-gb", type=float, default=34.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sc.calib_samples(REPO, args.manifest)[: args.num_clips][args.shard :: args.n_shards]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_expert_attn_impl(model, "sdpa")
    layers = model.vlm.model.language_model.layers
    n_l, n_h = len(layers), model.vlm.config.text_config.num_attention_heads

    state = {}

    def make_hook(li):
        def hook(module, a, kw, output):
            attn = output[1]  # (1, H, T, T) attention probabilities (eager)
            assert attn is not None, "eager attention required"
            with torch.autocast("cuda", enabled=False):
                rows = attn[0].index_select(1, state["coc_idx"]).float()  # (H, Nc, T)
                m = torch.einsum("hqt,gt->hg", rows, state["onehot"])  # (H, 5): sink, vision, hist, prompt, coc(all)
                self_mass = rows[:, torch.arange(rows.shape[1], device=rows.device), state["coc_idx"]].sum(1)  # (H,)
                out = torch.cat([m[:, :4], (m[:, 4] - self_mass).unsqueeze(1), self_mass.unsqueeze(1)], 1)  # (H, 6)
            state["mass"][li] = (out / state["coc_idx"].numel()).cpu().numpy()
        return hook

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "plan": "plans/2026-09-25_coc-position-disagreement.md (test 2)",
        "manifest": args.manifest, "cache": args.cache, "num_clips": len(clips), "shard": [args.shard, args.n_shards],
        "seed": args.seed, "seed_from": "sample_cache.clip_seed(seed, clip_id), as run_gradient_anatomy",
        "groups": GROUPS, "queries": "generated-CoC tokens (teacher-forced own rollout)",
        "gpu": torch.cuda.get_device_name(device)}, indent=2))

    per_clip, done, n_coc, lens = [], [], [], []
    for ci, (clip_id, clip_t0) in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        seed = sc.clip_seed(args.seed, clip_id)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        lib.set_vlm_attn_impl(model, "sdpa")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1].clone()  # (1, T) prompt + generated CoC
        del roll
        T = seq_tf.shape[1]
        type_idx = token_types(model, seq_tf, prompt_len)
        onehot = torch.stack([type_idx == TYPES.index(t) for t in ("sink", "vision", "hist", "prompt_text", "coc")]).float()  # (5, T)
        state["coc_idx"] = torch.arange(prompt_len, T, device="cuda")
        state["onehot"] = onehot.to("cuda")
        state["mass"] = np.zeros((n_l, n_h, len(GROUPS)))
        lib.set_vlm_attn_impl(model, "eager")
        handles = [layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=False,
            )
        for h in handles:
            h.remove()
        per_clip.append(state["mass"].copy())
        done.append(clip_id)
        n_coc.append(int(T - prompt_len))
        lens.append(int(T))
        lay = state["mass"].mean(1)  # (L, 6)
        print(f"[{ci + 1}/{len(clips)}] {clip_id} T={T} coc={n_coc[-1]} row-sum {state['mass'].sum(-1).mean():.4f} | "
              f"vision mass L8 {lay[8, 1]:.3f} L20 {lay[20, 1]:.3f} L30 {lay[30, 1]:.3f} | coc_prev+self L30 {lay[30, 4:].sum():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del state["onehot"]
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            arr = np.stack(per_clip)  # (N, L, H, 6)
            np.savez(out_dir / "census.npz", mass_by_clip=arr, mass=arr.mean(0))
            (out_dir / "metrics.json").write_text(json.dumps({
                "n_clips": len(done), "clip_ids": done, "n_coc": n_coc, "T": lens, "groups": GROUPS,
                "layer_mean_mass": arr.mean((0, 2)).tolist()}, indent=2))
    lay = np.stack(per_clip).mean((0, 2))  # (L, 6)
    lines = [f"CoC-query attention census -- {len(done)} clips", "", "layer  " + "  ".join(f"{g:>9s}" for g in GROUPS)]
    lines += [f"{li:5d}  " + "  ".join(f"{v:9.3f}" for v in lay[li]) for li in range(n_l)]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
