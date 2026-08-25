"""Pathway map, Stage 1: expert <- cache-span attention knockout.

The expert reads the VLM's per-layer KV cache; `denoise_with_cache` already takes a
`prefix_mask` over cache positions which `_build_expert_pos_ids_and_attn_mask` turns
into `finfo.min` on the 4D expert attention mask. So blocking a span is a mask edit,
not surgery.

Per clip: one rollout (reference CoC) -> one teacher-forced VLM forward -> one cache.
Every config then reuses that identical cache and only changes `prefix_mask`, so the
VLM side is bit-identical across configs and the CoC NLL is constant; the only thing
that varies is which cache positions the expert may attend to. Denoise seeds are
shared across configs, so all comparisons are clip-paired.

Because VLM prefill is causal, the prompt K/V cannot be modified by the CoC generated
after it. The CoC's only channel to the expert is its own cache positions, so blocking
`[coc_start, coc_end)` removes 100% of the CoC's influence on the trajectory.

Pre-registered gates (plans/2026-08-25_pathway-map.md):
  G0  each span block beats a size-matched random block
  G1  damage ranking differs from attention-mass ranking (Spearman < 0.9)
  G2  X3 (generated CoC) paired minADE CI excludes 0
  G3  X1 (vision) is catastrophic -- positive control; without it a G2 null is
      "no power", not "no effect"

Usage:
  bash experiments/head_analysis/run_retry_host.sh 20 \
      experiments/head_analysis/run_pathway.py --gpu 4 --num-clips 50 --k 8 \
      --outputs-root /home/cvlab21/project/chan/alpamayo-model-compression/outputs
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAM_NAMES = ["cam0", "cam1", "cam2", "cam3"]


def clip_seed(seed, clip_id):
    """Seed from clip identity, not loop index, so shards agree with a whole run."""
    return int(hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()[:4], 16)


def build_configs(spans, prompt_len, prefill, rng):
    """(name, meta, block_index_tensor) for every knockout config.

    Index tensors address cache positions in [0, prefill). Prompt spans come from
    compute_spans (length prompt_len); the CoC occupies [prompt_len, prefill).
    """
    def pad(mask_prompt):
        """Boolean prompt mask -> boolean cache mask (CoC region False)."""
        out = torch.zeros(prefill, dtype=torch.bool)
        out[:prompt_len] = mask_prompt
        return out

    vision = pad(spans["vision"])
    hist = pad(spans["hist"])
    sink = pad(spans["sink"])
    text = pad(spans["text"])
    coc = torch.zeros(prefill, dtype=torch.bool)
    coc[prompt_len:] = True

    cfgs = [("X0_none", {"span": "none", "n": 0}, None)]
    named = [
        ("X1_vision", "vision", vision),
        ("X2_hist", "traj_history", hist),
        ("X3_coc", "generated_coc", coc),
        ("X4_text", "prompt_text", text),
        ("X5_sink", "sink_pos0", sink),
    ]
    for name, tag, m in named:
        cfgs.append((name, {"span": tag, "n": int(m.sum())}, m))

    # everything except vision
    nonvis = (hist | sink | text | coc)
    cfgs.append(("X6_all_but_vision", {"span": "hist+text+sink+coc", "n": int(nonvis.sum())},
                 nonvis))

    # per-camera vision blocks
    if spans["per_camera"] is not None:
        for c in range(spans["per_camera"].shape[0]):
            m = pad(spans["per_camera"][c])
            cfgs.append((f"X1c{c}_{CAM_NAMES[c] if c < 4 else c}",
                         {"span": f"vision_camera_{c}", "n": int(m.sum())}, m))

    # G0: size-matched random controls (seeded per clip -> reproducible)
    for name, tag, m in named:
        k = int(m.sum())
        if k == 0 or k >= prefill:
            continue
        idx = torch.from_numpy(rng.choice(prefill, size=k, replace=False))
        rm = torch.zeros(prefill, dtype=torch.bool)
        rm[idx] = True
        cfgs.append((f"{name}_rand", {"span": f"random_{k}", "n": k, "control_for": name}, rm))

    return cfgs


@torch.no_grad()
def denoise_minade(model, inputs, cache, rope_deltas, prefill, block, gt_xy, seeds):
    """K denoisings on a fixed cache with `block` positions masked out of the expert's view."""
    device = inputs["input_ids"].device
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    if block is not None:
        prefix_mask[0, block.to(device)] = 0
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


def save(out_dir, cfg_names, cfg_meta, results, buckets, clip_ids, nll, n):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "buckets": buckets, "clip_ids": clip_ids,
        "configs": cfg_names, "meta": cfg_meta,
        "coc_nll": nll,  # identical across configs by construction; recorded once per clip
        "per_clip": results,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--exp-id", type=str, default="pathway_x_v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--outputs-root", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.outputs_root) if args.outputs_root else REPO / "outputs"
    out_dir = root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((root / "split.json").read_text())
    clips = split[args.split][args.clip_offset : args.clip_offset + args.num_clips]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    results, buckets, clip_ids_done, nlls = {}, [], [], []
    cfg_names, cfg_meta = None, None
    mass_ref = {"vision": 0.7255, "prompt_text": 0.1667, "generated_coc": 0.0230,
                "traj_history": 0.0393, "sink_pos0": 0.0109}

    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        spans = lib.compute_spans(model, inputs["input_ids"])
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        base_seed = clip_seed(args.seed, clip_id)
        seeds = [base_seed + k for k in range(args.k)]

        torch.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        # one teacher-forced forward -> the cache every config shares
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
            logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
            nll = torch.nn.functional.cross_entropy(
                logits[0], seq_tf[0, coc_start:coc_end]).item()
            cache = out.past_key_values
            prefill = cache.get_seq_length()

            rng = np.random.RandomState(base_seed)
            cfgs = build_configs(spans, prompt_len, prefill, rng)
            if cfg_names is None:
                cfg_names = [c[0] for c in cfgs]
                cfg_meta = {c[0]: c[1] for c in cfgs}
                results = {n: {"ade": [], "fde": []} for n in cfg_names}
                print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)
                (out_dir / "config.json").write_text(json.dumps({
                    "model": "nvidia/Alpamayo-1.5-10B", "stage": "X (expert <- cache span)",
                    "eval_split": args.split, "num_clips": len(clips), "clip_ids": clips,
                    "clip_offset": args.clip_offset, "k_samples": args.k, "seed": args.seed,
                    "seed_from": "sha256(f'{seed}:{clip_id}')[:4] -- shard-invariant",
                    "protocol": ("one rollout + one teacher-forced forward per clip; all configs "
                                 "share that cache and differ only in prefix_mask; denoise seeds "
                                 "shared across configs -> clip-paired"),
                    "configs": [{"name": n, **m} for n, m, _ in cfgs],
                    "attention_mass_reference": mass_ref,
                    "plan": "plans/2026-08-25_pathway-map.md",
                    "gpu": torch.cuda.get_device_name(device),
                }, indent=2))

            for name, _, block in cfgs:
                ade, fde = denoise_minade(model, inputs, cache, out.rope_deltas, prefill,
                                          block, gt_xy, seeds)
                results[name]["ade"].append(ade)
                results[name]["fde"].append(fde)

        buckets.append(el.bucket(gt_xy))
        clip_ids_done.append(clip_id)
        nlls.append(nll)
        del out, cache

        base = results["X0_none"]["ade"][-1]
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} coc_len={coc_end - coc_start:3d} "
              f"base={base:.3f} vision={results['X1_vision']['ade'][-1]:.3f} "
              f"coc={results['X3_coc']['ade'][-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            save(out_dir, cfg_names, cfg_meta, results, buckets, clip_ids_done, nlls, ci + 1)

    save(out_dir, cfg_names, cfg_meta, results, buckets, clip_ids_done, nlls, len(clips))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
