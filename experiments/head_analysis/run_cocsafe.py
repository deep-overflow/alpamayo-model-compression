"""CoC-preserving compression sweep: prune only what BOTH objectives can spare.

The trajectory-only config bought 29.3% by sacrificing CoC (dNLL +2.86, mostly
layer 35 + the late-band decimation). Here units are selected by the dual score
max(rank_traj, rank_coc) within each layer, so anything important to either
objective is protected; layer 35 and the late-band decimation are excluded.
The expert config carries over unchanged -- the expert cannot affect CoC (its
CoC gradient is exactly zero, P1).

Output is the dADE-vs-dNLL frontier across ratios, so "preserved" can be read
off at any NLL threshold.

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_cocsafe.py --gpu 0 \
      --clip-offset 0 --num-clips 20 --exp-id cocsafe_s0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_eval import eval_config  # noqa: E402
from run_integrated import expert_masks  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def rank_norm(scores):
    """Per-layer rank in [0, 1]. (L, U) -> (L, U)."""
    out = np.zeros_like(scores, dtype=float)
    n = scores.shape[1]
    for i in range(scores.shape[0]):
        out[i] = np.argsort(np.argsort(scores[i])) / max(n - 1, 1)
    return out


def cocsafe_configs(imp, emag, n_layers, n_heads, intermediate, n_exp_layers, seed):
    all_l = list(range(n_layers))
    dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
    dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
    dual_kv = np.maximum(rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"]),
                         rank_norm(imp["coc_kv_k"] + imp["coc_kv_v"]))
    kv1 = ml.kv_group_mask(dual_kv, 1, n_heads, all_l)
    eq, em = expert_masks(imp, emag, n_exp_layers, "magnitude")

    cfgs = [("baseline", {"kind": "baseline"}, None, None, None, None)]
    for r in (0.10, 0.20, 0.30, 0.40):
        q = ml.select_mask(dual_q, r, all_l)
        m = ml.select_mask(dual_m, r, all_l)
        cfgs.append((f"dual_r{int(r * 100)}", {"kind": "dual", "ratio": r}, q, m, None, None))
    # single-objective references at the pivotal ratio
    for crit in ("traj", "coc"):
        q = ml.select_mask(imp[f"{crit}_vlm_q"], 0.30, all_l)
        m = ml.select_mask(imp[f"{crit}_vlm_mlp"], 0.30, all_l)
        cfgs.append((f"{crit}_r30", {"kind": "single", "criterion": crit, "ratio": 0.3},
                     q, m, None, None))
    # dual + KV1, and the full deployable CoC-safe config (+ expert)
    for r in (0.20, 0.30):
        q = ml.select_mask(dual_q, r, all_l) * kv1
        m = ml.select_mask(dual_m, r, all_l)
        cfgs.append((f"dual_r{int(r * 100)}_kv1", {"kind": "dual_kv", "ratio": r},
                     q, m, None, None))
        cfgs.append((f"cocsafe_full_r{int(r * 100)}", {"kind": "full", "ratio": r},
                     q, m, eq, em))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=20)
    ap.add_argument("--clip-file", type=str, default="outputs/combined_eval_clips.json")
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = json.loads((REPO / args.clip_file).read_text())
    clips = clips[args.clip_offset : args.clip_offset + args.num_clips]
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    tc = model.vlm.config.text_config
    ec = model.expert.config
    vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                           tc.head_dim, tc.intermediate_size, "cuda")
    emasks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                           ec.intermediate_size, "cuda")
    emag = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                               ec.intermediate_size)
    cfgs = cocsafe_configs(imp, emag, tc.num_hidden_layers, tc.num_attention_heads,
                           tc.intermediate_size, ec.num_hidden_layers, args.seed)
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "k_samples": args.k,
        "importance_from": args.importance, "seed": args.seed,
        "selection": "per-layer max(rank_traj, rank_coc); no layer-35 drop, no late decimation",
        "configs": [{"name": n, **m} for n, m, *_ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    results = {n: {"ade": [], "fde": [], "nll": []} for n, *_ in cfgs}
    buckets, clip_ids_done = [], []
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        buckets.append(el.bucket(gt_xy))
        clip_ids_done.append(clip_id)
        seeds = [args.seed + ci * 100 + k for k in range(args.k)]

        vmasks.reset(); emasks.reset()
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        for name, _, q, m, eq, em in cfgs:
            vmasks.set(q=q, mlp=m)
            emasks.set(q=eq, mlp=em)
            ade, fde, nll = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds)
            results[name]["ade"].append(ade)
            results[name]["fde"].append(fde)
            results[name]["nll"].append(nll)
        vmasks.reset(); emasks.reset()

        b = np.mean(results["baseline"]["ade"])
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} "
              f"baseMinADE={b:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            save(out_dir, cfgs, results, buckets, clip_ids_done, ci + 1)
    save(out_dir, cfgs, results, buckets, clip_ids_done, len(clips))
    print("saved ->", out_dir, flush=True)


def save(out_dir, cfgs, results, buckets, clip_ids, n):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "buckets": buckets, "clip_ids": clip_ids,
        "configs": [c[0] for c in cfgs], "meta": {c[0]: c[1] for c in cfgs},
        "per_clip": {k: v for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
