"""KV-group ablation fidelity fix: mask BOTH towers' Q heads for dropped groups.

The earlier KV-group configs masked only the VLM-side Q heads (4/group), so the
expert kept reading the group's cache and the claimed "KV cache -12.5%" was not
actually tested. Masking the expert-side heads (2/group) fully is CONSERVATIVE:
it also removes their attention over the expert's own 64 tokens, which physical
group removal would keep. So "both free" -> the cache-saving claim is supported
with margin; "both regresses" -> the claim must be withdrawn.

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_kvfix.py --gpu 0 \
      --clip-offset 0 --num-clips 20 --exp-id kvfix_s0
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


def kv_masks_both(scores_kv, n_drop, n_vlm_heads, n_exp_heads):
    """Per layer, drop the n weakest KV groups on BOTH towers' Q heads."""
    n_layers, n_kv = scores_kv.shape
    per_v = n_vlm_heads // n_kv  # 4
    per_e = n_exp_heads // n_kv  # 2
    vq = np.ones((n_layers, n_vlm_heads))
    eq = np.ones((n_layers, n_exp_heads))
    for li in range(n_layers):
        for g in np.argsort(scores_kv[li])[:n_drop]:
            vq[li, g * per_v : (g + 1) * per_v] = 0.0
            eq[li, g * per_e : (g + 1) * per_e] = 0.0
    return vq, eq


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

    kv = imp["traj_kv_k"] + imp["traj_kv_v"]
    vq1, eq1 = kv_masks_both(kv, 1, tc.num_attention_heads, ec.num_attention_heads)
    vq2, eq2 = kv_masks_both(kv, 2, tc.num_attention_heads, ec.num_attention_heads)

    # (name, meta, vlm_q, exp_q)
    cfgs = [
        ("baseline", {"kind": "baseline"}, None, None),
        ("kv1_vlm", {"kind": "kv", "side": "vlm", "n_drop": 1}, vq1, None),
        ("kv1_expert", {"kind": "kv", "side": "expert", "n_drop": 1}, None, eq1),
        ("kv1_both", {"kind": "kv", "side": "both", "n_drop": 1}, vq1, eq1),
        ("kv2_both", {"kind": "kv", "side": "both", "n_drop": 2}, vq2, eq2),
    ]
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "k_samples": args.k,
        "importance_from": args.importance, "seed": args.seed,
        "purpose": "fidelity fix: KV group removal must silence both towers' Q heads",
        "configs": [{"name": n, **m} for n, m, _, _ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    results = {n: {"ade": [], "fde": [], "nll": []} for n, _, _, _ in cfgs}
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

        for name, _, vq, eq in cfgs:
            vmasks.set(q=vq)
            emasks.set(q=eq)
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
