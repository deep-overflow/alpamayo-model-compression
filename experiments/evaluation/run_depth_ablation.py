"""Turn each VLM layer off one at a time and measure what it costs.

Width pruning removes a fraction of every layer's units; this asks a different question:
is any whole layer unnecessary? That matters because the workload is step-bound -- the
expert is 5.5% of the FLOPs but 22-28% of wall-clock, because it runs ten sequential
denoise passes, and CoC decoding is autoregressive. Narrowing a layer speeds each step a
little; deleting one removes a step outright.

"Off" means kv-only: the layer's Q heads and MLP channels are masked, but k_proj/v_proj
still run, because the expert reads the VLM's per-layer KV cache and a missing entry
breaks it. That is ~185M parameters per layer (Q/O 33.5M + MLP 151M) and it is exactly
what `slim_lib.kvonly_layer_forward` does physically.

Layer scores cannot decide this. Taylor importance scales with the activation magnitude
at the gate, so a layer with small activations scores low without being unimportant --
which is why `select_mask` only ever ranks within a layer. So measure it directly.

Per clip the *unmasked* model rolls out one CoC, and every mask is then teacher-forced on
that same CoC. Sharing the context costs one rollout instead of 37 and gives dNLL a
consistent meaning: can the ablated model still predict the reasoning the intact model
produced. Trajectories are NOT shared -- each mask denoises its own eight samples, which
is where dminADE comes from. No ground-truth CoC is involved anywhere; the only label is
the GT trajectory from egomotion.

Gate D (pre-registered): depth has room if at least three layers show a median dminADE
below +0.05 m, the smallest effect this protocol resolves. Otherwise this is a negative
result and the axis closes.

Usage:
  python experiments/evaluation/run_depth_ablation.py --shard 0 --n-shards 4 --gpu 4
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import eval_lib as el
import mask_lib as ml
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_eval import eval_config
from sample_cache import clip_seed

MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kvonly_masks(layer, n_layers, n_heads, intermediate):
    """Keep-masks with every Q head and MLP channel of `layer` removed."""
    q = np.ones((n_layers, n_heads), dtype=np.float32)      # (36, 32)
    m = np.ones((n_layers, intermediate), dtype=np.float32)  # (36, 12288)
    q[layer] = 0.0
    m[layer] = 0.0
    return q, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="depth_ablation")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--manifest", default="val_500")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--num-clips", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="subset of layers to ablate; default is all of them")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"rows_s{args.shard}of{args.n_shards}.json"

    # the manifest is in greedy distribution-matched order, so any prefix is itself
    # matched -- taking the first N needs no separate draw
    man = pd.read_parquet(REPO / "outputs" / args.sets_id / f"{args.manifest}.parquet")
    clips = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()][: args.num_clips]
    clips = clips[args.shard::args.n_shards]

    set_determinism()
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    n_layers = tc.num_hidden_layers
    masks = ml.PruneMasks(layers, tc.num_attention_heads, tc.head_dim,
                          tc.intermediate_size, "cuda")
    todo = args.layers if args.layers is not None else list(range(n_layers))
    print(f"{args.exp_id} | shard {args.shard}/{args.n_shards} | {len(clips)} clips "
          f"x {len(todo)} layers + baseline | {gpu_name}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "per-layer kv-only ablation: what does removing each layer cost",
        "manifest": args.manifest, "cache": args.cache, "num_clips": args.num_clips,
        "layers": todo, "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "coc": "generated once by the UNMASKED model, teacher-forced into every mask",
        "shard": [args.shard, args.n_shards], "gpu": gpu_name,
        "model_revision": MODEL_REV,
    }, indent=2))

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    t_start = time.time()
    for ci, (clip_id, t0_us) in enumerate(clips):
        if clip_id in done:
            continue
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)

        base = clip_seed(args.seed, clip_id)
        traj_seeds = [base + k for k in range(args.k)]

        # one rollout with nothing masked; every config is scored against this context
        masks.reset()
        seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        rec = {"clip_id": clip_id, "bucket": el.bucket(gt_xy), "seed": base,
               "coc_len": int(coc_end - prompt_len), "layers": {}}
        ade, fde, nll = eval_config(model, inputs, seq_tf, prompt_len, coc_end,
                                    gt_xy, traj_seeds)
        rec["baseline"] = {"minADE": ade, "minFDE": fde, "nll": nll}

        for li in todo:
            q, m = kvonly_masks(li, n_layers, tc.num_attention_heads, tc.intermediate_size)
            masks.set(q=q, mlp=m)
            a, f, n = eval_config(model, inputs, seq_tf, prompt_len, coc_end,
                                  gt_xy, traj_seeds)
            rec["layers"][str(li)] = {"minADE": a, "minFDE": f, "nll": n}
        masks.reset()

        rows.append(rec)
        if len(rows) % 5 == 0 or ci + 1 == len(clips):
            rows_path.write_text(json.dumps(rows, indent=2))
        worst = max(todo, key=lambda li: rec["layers"][str(li)]["minADE"])
        best = min(todo, key=lambda li: rec["layers"][str(li)]["minADE"])
        print(f"[{ci + 1}/{len(clips)}] {clip_id[:8]} base={ade:.3f} "
              f"best=L{best}({rec['layers'][str(best)]['minADE']:.3f}) "
              f"worst=L{worst}({rec['layers'][str(worst)]['minADE']:.3f}) "
              f"({time.time() - t0:.0f}s)", flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} clips, {(time.time() - t_start) / 60:.1f} min -> {rows_path}",
          flush=True)


if __name__ == "__main__":
    main()
