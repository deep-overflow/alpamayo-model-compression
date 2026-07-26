"""Expert-tower masking sweep: how much of the action expert's width is free?

Expert masks do not touch the VLM, so each clip needs only ONE rollout + ONE
teacher-forced VLM forward; every config then reuses that cache and pays only its
K denoisings. CoC NLL is likewise config-independent (the expert cannot affect
it -- P1 measured exactly zero CoC gradient on expert gates), so it is recorded
once per clip as a sanity datum, not per config.

Criteria: trajectory Taylor (P1), weight magnitude, random. There is no CoC
criterion for the expert -- its CoC importance is structurally zero.
Scopes: all 36 layers, and early (0-21) -- P1 found expert trajectory importance
concentrated in LATE layers (opposite of the VLM), so early is the weak span.

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_expert_ablation.py --gpu 0 \
      --clip-file outputs/combined_eval_clips.json --num-clips 40
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

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
EARLY_END = 22  # expert importance grows with depth, so layers 0-21 are the weak span


def expert_configs(imp, mag, n_layers, n_heads, intermediate, ratios, seed):
    rng = np.random.RandomState(seed)
    tq, tm = imp["traj_exp_q"], imp["traj_exp_mlp"]
    scopes = {"all": list(range(n_layers)), "early": list(range(EARLY_END))}
    cfgs = [("baseline", {"kind": "baseline"}, None, None)]
    for scope, layers_scope in scopes.items():
        for ratio in ratios:
            for crit in ("traj", "magnitude", "random"):
                if crit == "traj":
                    q = ml.select_mask(tq, ratio, layers_scope)
                    m = ml.select_mask(tm, ratio, layers_scope)
                elif crit == "magnitude":
                    q = ml.select_mask(mag[0], ratio, layers_scope)
                    m = ml.select_mask(mag[1], ratio, layers_scope)
                else:
                    q = ml.select_mask(np.zeros((n_layers, n_heads)), ratio, layers_scope, rng=rng)
                    m = ml.select_mask(np.zeros((n_layers, intermediate)), ratio, layers_scope,
                                       rng=rng)
                cfgs.append((f"exp_{crit}_{scope}_r{int(ratio * 100)}",
                             {"kind": "expert_width", "criterion": crit, "scope": scope,
                              "ratio": ratio}, q, m))
    return cfgs


@torch.no_grad()
def denoise_minade(model, inputs, cache, rope_deltas, prefill, gt_xy, seeds):
    """K denoisings on a prebuilt cache -> (minADE, minFDE)."""
    device = inputs["input_ids"].device
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    preds = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask, seed=s)
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    return el.min_metrics(np.stack(preds), gt_xy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=40)
    ap.add_argument("--clip-file", type=str, default="outputs/combined_eval_clips.json")
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.25, 0.40, 0.50])
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

    ec = model.expert.config
    masks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                          ec.intermediate_size, "cuda")
    mag = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                              ec.intermediate_size)
    cfgs = expert_configs(imp, mag, ec.num_hidden_layers, ec.num_attention_heads,
                          ec.intermediate_size, args.ratios, args.seed)
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "tower": "expert",
        "num_clips": len(clips), "clip_ids": clips, "clip_offset": args.clip_offset,
        "k_samples": args.k, "ratios": args.ratios, "early_end": EARLY_END,
        "importance_from": args.importance, "seed": args.seed,
        "protocol": "one shared VLM forward per clip; expert masks affect denoising only",
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

        masks.reset()
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        # one VLM forward serves every expert config
        attention_mask = torch.ones_like(seq_tf)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=attention_mask,
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
            logits = model.vlm.lm_head(
                out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
            nll = torch.nn.functional.cross_entropy(
                logits[0], seq_tf[0, coc_start:coc_end]).item()
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        prefill = cache.get_seq_length()
        del out

        for name, _, q, m in cfgs:
            masks.set(q=q, mlp=m)
            ade, fde = denoise_minade(model, inputs, cache, rope_deltas, prefill, gt_xy, seeds)
            results[name]["ade"].append(ade)
            results[name]["fde"].append(fde)
            results[name]["nll"].append(nll)  # identical by construction; kept for merge compat
        masks.reset()
        del cache

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
        "configs": [n_ for n_, _, _, _ in cfgs],
        "meta": {n_: m for n_, m, _, _ in cfgs},
        "per_clip": {k: v for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
