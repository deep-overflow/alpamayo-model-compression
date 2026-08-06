"""Stage E: does the label-free J-score preserve CoC as well as the dual criterion?

cocsafe protects reasoning by scoring every unit against the CoC NLL, which needs
reference CoC text for the target model. The J-lens score needs none: it only asks
how much of a unit's write lands in directions the unembedding can read out. G2
showed it tracks the CoC objective about twice as strongly as the trajectory one
(rho .60 vs .31 for Q heads), so the question here is whether that survives
contact with the actual metric.

Criteria, all at matched per-layer ratios so this is a criterion comparison and
not a width sweep:

  magnitude  weight norm -- the standard structured-pruning baseline
  traj       trajectory Taylor -- what integrated_mag used
  coc        CoC Taylor -- needs labels, the thing we want to do without
  cocsafe    max(rank traj, rank coc) -- the shipped dual criterion
  jspace     J-score -- label-free, NEW
  j_traj     max(rank J, rank traj) -- label-free analogue of cocsafe, NEW

Masks are VLM-only and no KV group is dropped: the J-lens covers the VLM tower,
and mixing in the expert or KV config would confound the comparison.

Usage:
  bash run_retry_host.sh 20 experiments/head_analysis/run_jspace_sweep.py --gpu 0 \
      --exp-id jsweep_s0 --num-clips 20
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
from run_cocsafe import rank_norm  # noqa: E402
from run_eval import eval_config  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def criterion_scores(imp, jl, mag_q, mag_mlp):
    """(L, U) score arrays per criterion; higher = keep."""
    return {
        "magnitude": (mag_q, mag_mlp),
        "traj": (imp["traj_vlm_q"], imp["traj_vlm_mlp"]),
        "coc": (imp["coc_vlm_q"], imp["coc_vlm_mlp"]),
        "cocsafe": (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),
                    np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))),
        "jspace": (jl["q_j"], jl["mlp_j"]),
        "j_traj": (np.maximum(rank_norm(jl["q_j"]), rank_norm(imp["traj_vlm_q"])),
                   np.maximum(rank_norm(jl["mlp_j"]), rank_norm(imp["traj_vlm_mlp"]))),
    }


def sweep_configs(imp, jl, mag_q, mag_mlp, n_layers, ratios):
    scores = criterion_scores(imp, jl, mag_q, mag_mlp)
    all_l = list(range(n_layers))
    cfgs = [("baseline", {"kind": "baseline"}, None, None)]
    for crit, (sq, sm) in scores.items():
        for r in ratios:
            cfgs.append((
                f"{crit}_r{int(r * 100)}",
                {"kind": "criterion", "criterion": crit, "ratio": r},
                ml.select_mask(sq, r, all_l), ml.select_mask(sm, r, all_l),
            ))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=20)
    ap.add_argument("--clip-file", type=str, default="outputs/combined_eval_clips.json")
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--jlens", type=str, default="jlens_coc")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.20, 0.30])
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
    jl = dict(np.load(REPO / "outputs" / args.jlens / "jlens.npz"))

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
    vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                           tc.head_dim, tc.intermediate_size, "cuda")
    cfgs = sweep_configs(imp, jl, jl["mag_q"], jl["mag_mlp"], tc.num_hidden_layers, args.ratios)
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "k_samples": args.k, "ratios": args.ratios,
        "importance_from": args.importance, "jlens_from": args.jlens, "seed": args.seed,
        "selection": "per-layer, VLM only, no KV drop, no expert mask -- criterion comparison",
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

        # the CoC context every config is scored against comes from the unmasked
        # model, so dNLL measures damage to the same target sequence
        vmasks.reset()
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        for name, _, q, m in cfgs:
            vmasks.set(q=q, mlp=m)
            ade, fde, nll = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds)
            results[name]["ade"].append(ade)
            results[name]["fde"].append(fde)
            results[name]["nll"].append(nll)
        vmasks.reset()

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
    base = np.array(results["baseline"]["nll"])
    base_ade = np.array(results["baseline"]["ade"])
    lines = [f"criterion sweep, {n} clips", "",
             f"{'config':<20} {'minADE':>8} {'dADE':>8} {'NLL':>8} {'dNLL':>8}"]
    for name in results:
        ade = np.array(results[name]["ade"])
        nll = np.array(results[name]["nll"])
        lines.append(f"{name:<20} {ade.mean():8.3f} {ade.mean() - base_ade.mean():+8.3f} "
                     f"{nll.mean():8.3f} {nll.mean() - base.mean():+8.3f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
