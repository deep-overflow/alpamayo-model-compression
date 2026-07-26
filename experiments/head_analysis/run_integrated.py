"""Integrated combined config: VLM graded width + KV1 + expert early40/late10.

Masks sit on BOTH towers at once. Because the VLM masks differ per config, every
config pays its own teacher-forced forward (unlike the expert-only sweep).

Configs: baseline, each tower's combined config alone (references), and the full
integrated config in two expert-criterion variants (trajectory Taylor vs weight
magnitude -- the expert sweep found magnitude equal-or-better at these ratios).

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_integrated.py --gpu 0 \
      --clip-offset 0 --num-clips 20 --exp-id integrated_s0
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

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
LATE_START = 22  # VLM: criteria diverge / trajectory importance drops after here
EXP_EARLY_END = 22  # expert: importance concentrates late, early span is weak


def vlm_combined_masks(imp, n_layers, n_heads, intermediate):
    """The measured-free VLM config: 0-21 @30%, 22-34 @50%, layer35 full, KV drop1."""
    tq, tm = imp["traj_vlm_q"], imp["traj_vlm_mlp"]
    kv = imp["traj_kv_k"] + imp["traj_kv_v"]
    early = list(range(0, LATE_START))
    late = list(range(LATE_START, n_layers - 1))
    q = ml.select_mask(tq, 0.30, early) * ml.select_mask(tq, 0.50, late)
    m = ml.select_mask(tm, 0.30, early) * ml.select_mask(tm, 0.50, late)
    q[n_layers - 1] = 0.0
    m[n_layers - 1] = 0.0
    q = q * ml.kv_group_mask(kv, 1, n_heads, list(range(n_layers)))
    return q, m


def expert_masks(imp, mag, n_layers, criterion):
    """The measured-free expert config: early(0-21) @40% + late @10%."""
    if criterion == "traj":
        sq, sm = imp["traj_exp_q"], imp["traj_exp_mlp"]
    else:
        sq, sm = mag
    early = list(range(EXP_EARLY_END))
    late = list(range(EXP_EARLY_END, n_layers))
    q = ml.select_mask(sq, 0.40, early) * ml.select_mask(sq, 0.10, late)
    m = ml.select_mask(sm, 0.40, early) * ml.select_mask(sm, 0.10, late)
    return q, m


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

    vq, vm = vlm_combined_masks(imp, tc.num_hidden_layers, tc.num_attention_heads,
                                tc.intermediate_size)
    eq_t, em_t = expert_masks(imp, emag, ec.num_hidden_layers, "traj")
    eq_m, em_m = expert_masks(imp, emag, ec.num_hidden_layers, "magnitude")

    # (name, meta, vlm_q, vlm_mlp, exp_q, exp_mlp)
    cfgs = [
        ("baseline", {"kind": "baseline"}, None, None, None, None),
        ("vlm_only", {"kind": "ref", "note": "combined_width_kv1"}, vq, vm, None, None),
        ("expert_only_traj", {"kind": "ref"}, None, None, eq_t, em_t),
        ("expert_only_mag", {"kind": "ref"}, None, None, eq_m, em_m),
        ("integrated_traj", {"kind": "integrated"}, vq, vm, eq_t, em_t),
        ("integrated_mag", {"kind": "integrated"}, vq, vm, eq_m, em_m),
    ]
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "k_samples": args.k,
        "importance_from": args.importance, "seed": args.seed,
        "vlm_config": "graded 30/50 + layer35 + KV drop1 (traj)",
        "expert_config": "early(0-21) 40% + late 10%",
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
