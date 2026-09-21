"""Token-targeted ablation: do the units the score anatomy singles out carry the function?

plans/2026-09-21_importance-causal-validation.md, part B. The unit sets come from
make_token_sets.py (chosen on the dense model's calib_100 scores); this runner removes each
set from the dense model with mask_lib.PruneMasks and reads, on HELD-OUT clips, three things
per config, all paired with the dense model on the same clip:

  nll      CoC NLL of the DENSE model's rollout, teacher-forced through the masked model
  fm_loss  the 10-step GT-anchored flow-matching loss on the masked model's cache, with the
           noise stream of run_importance (prune_lib.expert_fm_loss)
  ade/fde  minADE/minFDE over K denoisings on that same cache (skipped for the random sets
           that are only there as the null of the two losses)

One masked teacher-forced VLM forward serves all three. The reference text is generated once
per clip with the masks off, so every config sees the same tokens at the same positions;
seeds are clip-derived (sample_cache.clip_seed), so shards and whole runs agree.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/head_analysis/run_token_ablation.py --gpu 4 --exp-id tokabl_v1_s0 \
      --shard 0 --n-shards 4
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

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


@torch.no_grad()
def eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, x1, fm_seed, fm_steps, seeds):
    """One masked teacher-forced forward -> CoC NLL, FM loss, and minADE/minFDE over `seeds`
    (None when seeds is empty)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end]).item()
    cache = out.past_key_values
    prefill = cache.get_seq_length()
    fm = pl.expert_fm_loss(model, cache, out.rope_deltas, x1, fm_steps, fm_seed, prefill)
    ade = fde = None
    if seeds:
        offset = torch.tensor([prefill], device=seq_tf.device)
        prefix_mask = torch.ones(1, prefill, device=seq_tf.device, dtype=torch.long)
        preds = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in seeds:
                action = lib.denoise_with_cache(model, cache, out.rope_deltas, offset,
                                                prefix_mask, seed=s)
                pred_xyz, _ = model.action_space.action_to_traj(
                    action.float(), inputs["ego_history_xyz"][:, -1].float(),
                    inputs["ego_history_rot"][:, -1].float(),
                )
                preds.append(pred_xyz[0, :, :2].cpu().numpy())
        ade, fde = el.min_metrics(np.stack(preds), gt_xy)  # (K, 64, 2) vs (64, 2)
    del out, cache
    return nll, fm, ade, fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tokabl_sets_v1", help="exp id written by make_token_sets.py")
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--manifest", default="indist_500", help="clip manifest under outputs/eval_sets")
    ap.add_argument("--cache", default="eval", help="sample cache the manifest's clips live in")
    ap.add_argument("--num-clips", type=int, default=100, help="first N of the manifest, before sharding")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--arm-masks", nargs="+", default=None, metavar="NAME=NPZ",
                    help="instead of the unit sets: whole pruned arms, each an npz with q_mask/mlp_mask "
                         "(slim_to_mask.py). Same readouts, so the arms of run_gradient_anatomy --mask can "
                         "be read on held-out clips, where no criterion has seen the text or the actions")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sc.calib_samples(REPO, args.manifest)[: args.num_clips][args.shard :: args.n_shards]
    # (name, q keep mask or None, mlp keep mask or None, sample trajectories?)
    cfgs = [("dense", None, None, True)]
    set_meta = {}
    if args.arm_masks:
        for spec in args.arm_masks:
            name, path = spec.split("=", 1)
            z = np.load(REPO / path)
            cfgs.append((f"arm_{name}", z["q_mask"].astype(np.float32), z["mlp_mask"].astype(np.float32), True))
            set_meta[f"arm_{name}"] = {"axis": "both", "set": "arm", "mask": path}
    else:
        sets = np.load(REPO / "outputs" / args.sets / "sets.npz")
        set_meta = json.loads((REPO / "outputs" / args.sets / "sets.json").read_text())["configs"]
        for name in sets.files:
            m = set_meta[name]
            keep = sets[name].astype(np.float32)  # (36, U)
            sampled = not m["set"].startswith("R") or m["set"] == "R0"
            cfgs.append((name, keep if m["axis"] == "q" else None,
                         keep if m["axis"] == "mlp" else None, sampled))

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    tc = model.vlm.config.text_config
    masks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                          tc.head_dim, tc.intermediate_size, "cuda")
    print(f"{len(cfgs)} configs ({sum(c[3] for c in cfgs)} sampled at K={args.k}) x {len(clips)} clips",
          flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "plan": "plans/2026-09-21_importance-causal-validation.md (part B)",
        "sets": None if args.arm_masks else args.sets, "arm_masks": args.arm_masks,
        "manifest": args.manifest, "cache": args.cache,
        "num_clips": len(clips), "clip_ids": [c for c, _ in clips],
        "shard": args.shard, "n_shards": args.n_shards, "k_samples": args.k,
        "seed": args.seed, "seed_rule": "sample_cache.clip_seed: sha256(f'{seed}:{clip_id}')[:4]",
        "fm_steps": args.fm_steps, "max_gen": args.max_gen,
        "protocol": ("reference CoC generated once per clip with the masks off; per config one "
                     "masked teacher-forced VLM forward -> CoC NLL of that text, the GT-anchored "
                     "FM loss on its cache (run_importance's noise stream) and K denoisings"),
        "configs": [{"name": n, "sampled": s, **set_meta.get(n, {})} for n, _, _, s in cfgs],
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
    }, indent=2))

    results = {n: {"nll": [], "fm": [], "ade": [], "fde": []} for n, _, _, _ in cfgs}
    buckets, clip_ids, coc_lens = [], [], []
    for ci, (clip_id, clip_t0) in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
        base = sc.clip_seed(args.seed, clip_id)
        seeds = [base + k for k in range(args.k)]

        masks.reset()
        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()  # (1, T)
        del roll

        for name, q, mlp, sampled in cfgs:
            masks.set(q=q, mlp=mlp)
            nll, fm, ade, fde = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, x1,
                                            base, args.fm_steps, seeds if sampled else [])
            r = results[name]
            r["nll"].append(nll)
            r["fm"].append(fm)
            r["ade"].append(ade)
            r["fde"].append(fde)
        masks.reset()

        buckets.append(el.bucket(gt_xy))
        clip_ids.append(clip_id)
        coc_lens.append(int(coc_end - coc_start))
        d = results["dense"]
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} coc={coc_lens[-1]} "
              f"dense nll={d['nll'][-1]:.4f} fm={d['fm'][-1]:.4f} minADE={d['ade'][-1]:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            (out_dir / "metrics.json").write_text(json.dumps({
                "n_clips": ci + 1, "clip_ids": clip_ids, "buckets": buckets, "coc_len": coc_lens,
                "configs": [n for n, _, _, _ in cfgs], "per_clip": results}, indent=2))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
