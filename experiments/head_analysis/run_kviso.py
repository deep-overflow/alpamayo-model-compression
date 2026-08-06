"""KV-axis isolation: does the 8->32-clip J re-estimate's KV-group choice matter?

Stage 1 (jsweep32) showed the j_traj width selection is noise-robust -- the
max(rank traj, rank J) guardrail absorbed the ~25% pick churn -- but the KV-1
drop flips in 11/36 layers between jlens_coc and jlens_coc32: only 8 candidates
per layer, ranks quantized to 1/7, exact ties in layers 1/25/35. This run holds
the j_traj r20 width masks fixed (built from jlens_coc32) and varies ONLY the
KV-group drop:

  baseline       no mask
  width32        width masks only, no KV drop    (sizes the KV-drop cost)
  width32_kv8    + KV-1 selected from jlens_coc   (the shipped choice)
  width32_kv32   + KV-1 selected from jlens_coc32 (the re-estimate's choice)

VLM-side masks only, mirroring run_jspace_sweep: both KV variants are treated
identically, so the paired kv32-kv8 difference isolates the choice itself.

Gate (plans/2026-08-05_jtraj-reselect-32clip.md, Stage 1b): kv32-kv8 paired
dNLL ~ 0 with p > 0.05 -> KV axis insensitive, Stage 2 unnecessary; kv32
significantly better -> rebuild slim_j_traj32_r20 and reconsider closed-loop.

Usage:
  bash run_retry_host.sh 20 experiments/head_analysis/run_kviso.py --gpu 2 \
      --exp-id kviso_v1 --num-clips 80
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


def kv_choice_mask(jl, kv_taylor, n_layers, n_kv, n_heads, all_l):
    """Q-head keep-mask implied by dropping each layer's weakest KV group.

    Same coupling make_slim uses for j_traj_full: a group's J-mass is the summed
    squared J-score of the VLM Q heads it feeds (group h -> heads [4h, 4h+4)).
    """
    per = n_heads // n_kv
    j_kv = (jl["q_j"] ** 2).reshape(n_layers, n_kv, per).sum(-1)  # (L, KV)
    dual_kv = np.maximum(kv_taylor, rank_norm(j_kv))  # (L, KV)
    return ml.kv_group_mask(dual_kv, 1, n_heads, all_l), dual_kv


def build_configs(imp, jl8, jl32, n_layers, n_heads, n_kv, ratio):
    all_l = list(range(n_layers))
    # width: the j_traj selection with the reasoning half from the 32-clip lens
    dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(jl32["q_j"]))  # (L, H)
    dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(jl32["mlp_j"]))  # (L, I)
    vq_w = ml.select_mask(dual_q, ratio, all_l)  # (L, H)
    vm = ml.select_mask(dual_m, ratio, all_l)  # (L, I)
    kv_taylor = rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"])  # (L, KV)
    kv8, _ = kv_choice_mask(jl8, kv_taylor, n_layers, n_kv, n_heads, all_l)
    kv32, _ = kv_choice_mask(jl32, kv_taylor, n_layers, n_kv, n_heads, all_l)
    changed = [li for li in all_l if not np.array_equal(kv8[li], kv32[li])]
    return [
        ("baseline", {"kind": "baseline"}, None, None),
        ("width32", {"kind": "width_only"}, vq_w, vm),
        ("width32_kv8", {"kind": "width+kv", "kv_from": "jlens_coc"}, vq_w * kv8, vm),
        ("width32_kv32", {"kind": "width+kv", "kv_from": "jlens_coc32"}, vq_w * kv32, vm),
    ], changed


def save(out_dir, cfgs, results, buckets, clip_ids, n):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "buckets": buckets, "clip_ids": clip_ids,
        "configs": [c[0] for c in cfgs], "meta": {c[0]: c[1] for c in cfgs},
        "per_clip": {k: v for k, v in results.items()},
    }, indent=2))
    base_nll = np.array(results["baseline"]["nll"])
    base_ade = np.array(results["baseline"]["ade"])
    lines = [f"KV-axis isolation, {n} clips", "",
             f"{'config':<16} {'minADE':>8} {'dADE':>8} {'NLL':>8} {'dNLL':>8}"]
    for name in results:
        ade = np.array(results[name]["ade"])
        nll = np.array(results[name]["nll"])
        lines.append(f"{name:<16} {ade.mean():8.3f} {ade.mean() - base_ade.mean():+8.3f} "
                     f"{nll.mean():8.3f} {nll.mean() - base_nll.mean():+8.3f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=80)
    ap.add_argument("--clip-file", type=str, default="outputs/combined_eval_clips.json")
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--jlens-old", type=str, default="jlens_coc")
    ap.add_argument("--jlens-new", type=str, default="jlens_coc32")
    ap.add_argument("--ratio", type=float, default=0.20)
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
    jl8 = dict(np.load(REPO / "outputs" / args.jlens_old / "jlens.npz"))
    jl32 = dict(np.load(REPO / "outputs" / args.jlens_new / "jlens.npz"))

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
    cfgs, changed = build_configs(imp, jl8, jl32, tc.num_hidden_layers,
                                  tc.num_attention_heads, tc.num_key_value_heads, args.ratio)
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}; "
          f"KV choice differs in {len(changed)} layers: {changed}", flush=True)
    vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                           tc.head_dim, tc.intermediate_size, "cuda")

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "k_samples": args.k, "ratio": args.ratio,
        "importance_from": args.importance, "jlens_old": args.jlens_old,
        "jlens_new": args.jlens_new, "kv_changed_layers": changed,
        "selection": "j_traj r20 width from jlens_coc32 held fixed; KV-1 drop varies; "
                     "VLM-side masks only (run_jspace_sweep protocol)",
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
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
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
        seq_tf = roll["sequences"][:, :coc_end].clone()  # (1, T)
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


if __name__ == "__main__":
    main()
