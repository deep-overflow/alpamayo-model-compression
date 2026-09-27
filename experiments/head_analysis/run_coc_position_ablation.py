"""Double dissociation at generated-CoC positions (plans/2026-09-25_coc-position-content.md, B).

Switches OFF the outputs of chosen Q heads at the generated-CoC positions only (TypedUnitGates
with the CoC-type gate set to zero; every other position keeps the head) and reads, on the dense
model's own rollout, the CoC NLL, the GT-anchored FM loss and minADE over K denoisings, paired
with the unmasked model on the same clip, text and noise (run_token_ablation.eval_config).
Head sets from make_coc_position_masks.py: T = FM-favoured at CoC, C = CE-favoured at CoC,
R0-2 = random, each for all layers and for layers 0-21.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_coc_position_ablation.py --gpu 4 --exp-id coc_posabl_v1_s0 --shard 0 --n-shards 4
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
import eval_lib as el
import prune_lib as pl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_token_ablation import MODEL_REV, eval_config

REPO = Path(__file__).resolve().parents[2]
TYPES = ("vision", "hist", "prompt_text", "coc", "sink")
COC = TYPES.index("coc")
SAMPLED = ("dense", "T_all", "C_all", "R0_all", "T_trunk", "C_trunk", "R0_trunk")  # minADE at K for these


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
    ap.add_argument("--sets", default="coc_posabl_sets_v1")
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--manifest", default="indist_500", help="held-out clips, as run_token_ablation")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sc.calib_samples(REPO, args.manifest)[: args.num_clips][args.shard :: args.n_shards]
    sets = np.load(REPO / "outputs" / args.sets / "masks.npz")
    cfgs = [("dense", None)] + [(name, sets[name].astype(bool)) for name in sets.files]  # (L, H) 1 = off at CoC

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    print(f"{len(cfgs)} configs ({sum(n in SAMPLED for n, _ in cfgs)} sampled at K={args.k}) x {len(clips)} clips", flush=True)
    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV, "plan": "plans/2026-09-25_coc-position-content.md (B)",
        "sets": args.sets, "manifest": args.manifest, "cache": args.cache, "num_clips": len(clips), "shard": [args.shard, args.n_shards],
        "k_samples": args.k, "seed": args.seed, "fm_steps": args.fm_steps,
        "protocol": "heads switched off at generated-CoC positions only (typed gate = 0 at the CoC type); reference text from the unmasked rollout",
        "configs": [n for n, _ in cfgs], "sampled": list(SAMPLED), "gpu": torch.cuda.get_device_name(device)}, indent=2))

    results = {n: {"nll": [], "fm": [], "ade": [], "fde": []} for n, _ in cfgs}
    buckets, clip_ids, coc_lens = [], [], []
    for ci, (clip_id, clip_t0) in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)
        base = sc.clip_seed(args.seed, clip_id)
        seeds = [base + k for k in range(args.k)]
        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll
        # gates go on after the rollout (their hooks index a fixed type vector) and come off after the clip
        gates = pl.TypedUnitGates(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size, len(TYPES), "cuda", mlp=False)
        gates.set_types(token_types(model, seq_tf, prompt_len).to("cuda"))
        for name, off in cfgs:
            with torch.no_grad():
                for l, g in enumerate(gates.q_gates):
                    g.fill_(1.0)
                    if off is not None:
                        g[COC, torch.as_tensor(off[l], device=g.device)] = 0.0
            nll, fm, ade, fde = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, x1, base, args.fm_steps,
                                            seeds if name in SAMPLED else [])
            r = results[name]
            r["nll"].append(nll)
            r["fm"].append(fm)
            r["ade"].append(ade)
            r["fde"].append(fde)
        gates.remove()
        del gates
        buckets.append(el.bucket(gt_xy))
        clip_ids.append(clip_id)
        coc_lens.append(int(coc_end - coc_start))
        d, t_, c_ = results["dense"], results["T_all"], results["C_all"]
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} coc={coc_lens[-1]} dense nll={d['nll'][-1]:.4f} fm={d['fm'][-1]:.4f} "
              f"minADE={d['ade'][-1]:.3f} | T_all dNLL {t_['nll'][-1] - d['nll'][-1]:+.4f} dFM {t_['fm'][-1] - d['fm'][-1]:+.4f} | "
              f"C_all dNLL {c_['nll'][-1] - d['nll'][-1]:+.4f} dFM {c_['fm'][-1] - d['fm'][-1]:+.4f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            (out_dir / "metrics.json").write_text(json.dumps({
                "n_clips": ci + 1, "clip_ids": clip_ids, "buckets": buckets, "coc_len": coc_lens,
                "configs": [n for n, _ in cfgs], "per_clip": results}, indent=2))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
