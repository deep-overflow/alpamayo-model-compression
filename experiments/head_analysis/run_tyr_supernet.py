"""Tyr baseline, algorithm 1: prune-to-supernet on the Alpamayo VLM.

Block-sequential: for each VLM layer, collect the input Hessians of o_proj and
down_proj over the calib_100 fused-prompt prefills (all tokens, no labels), solve
the OSSCAR removal+reconstruction at every sparsity level, save each level's full
weight, and write the expected (level-0) weight back into the model so the next
block sees error-accumulated inputs (upstream --error_accumulation, median version).
Level 0 is the u40_v2 budget exactly: cut 13/32 Q heads and 4898/12288 MLP channels
per layer; levels step by --head-step heads / --mlp-step channels, so any
type-conserving level assignment keeps the removed-parameter total at u40's
-2,657,452,032 (gate T0 in plans/2026-08-20_tyr-baseline.md).

Outputs (outputs/<exp-id>/): layers.NN.{self_attn.o_proj,mlp.down_proj}/<level>.pth
(bf16, full (out,in) tensors), metadata.json, summary.txt.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 720 \
      experiments/head_analysis/run_tyr_supernet.py --gpu 4,5,6,7
  # smoke: --num-clips 2 --blocks 2 --num-levels 3
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
import sample_cache as sc
import tyr_lib as tyr
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

REPO = Path(__file__).resolve().parents[2]


def preload_inputs(model, processor, calib, cache_name):
    """Build fused inputs once per clip; keep on CPU (pixel tensors are large)."""
    store = []
    for clip_id, t0 in calib:
        data = sc.load_cached(sc.path_for(cache_name, clip_id, t0))
        inp = lib.build_inputs(model, processor, data, "cuda")
        store.append({
            "clip_id": clip_id,
            "input_ids": inp["input_ids"].cpu(),
            "pixel_values": inp["tokenized_data"]["pixel_values"].cpu(),
            "image_grid_thw": inp["tokenized_data"]["image_grid_thw"].cpu(),
        })
    return store


def prefill(model, item):
    ids = item["input_ids"].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.vlm.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=item["pixel_values"].cuda(),
            image_grid_thw=item["image_grid_thw"].cuda(),
            use_cache=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="tyr_supernet_u40")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--num-levels", type=int, default=9)
    ap.add_argument("--head-cut", type=int, default=13)
    ap.add_argument("--head-step", type=int, default=1)
    ap.add_argument("--mlp-cut", type=int, default=4898)
    ap.add_argument("--mlp-step", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=None, help="first N blocks only (smoke)")
    ap.add_argument("--selection", choices=["osscar", "dual"], default="osscar",
                    help="osscar: Tyr local pruner chooses units; dual: units fixed by the\n"
                         "dual ranking (importance), only OSSCAR reconstruction applied")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--damp", type=float, default=1e-2,
                    help="Hessian damping x mean diag (upstream 1e-2; v2 uses 1.0)")
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

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
    layers = model.vlm.model.language_model.layers
    n_blocks = min(args.blocks or len(layers), len(layers))
    keeps_q = tyr.level_keeps(tc.num_attention_heads, args.head_cut, args.head_step,
                              args.num_levels)
    keeps_m = tyr.level_keeps(tc.intermediate_size, args.mlp_cut, args.mlp_step,
                              args.num_levels)

    if args.selection == "dual":
        imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
        dual_q, dual_m = tyr.dual_scores(imp)
    print(f"preloading {len(calib)} clip inputs...", flush=True)
    store = preload_inputs(model, processor, calib, args.cache)

    layer_names, level_map = [], {}
    for i in range(n_blocks):
        for suffix in ("mlp.down_proj", "self_attn.o_proj"):
            name = f"layers.{i:02d}.{suffix}"
            layer_names.append(name)
            level_map[name] = sorted(keeps_m if "mlp" in suffix else keeps_q)

    (out_dir / "metadata.json").write_text(json.dumps({
        "model_revision": MODEL_REV, "num_clips": len(calib),
        "clip_ids": [c for c, _ in calib],
        "head_cut": args.head_cut, "head_step": args.head_step,
        "mlp_cut": args.mlp_cut, "mlp_step": args.mlp_step,
        "num_levels": args.num_levels, "damp": args.damp,
        "selection": args.selection, "importance": args.importance,
        "levels_q": {str(k): v for k, v in keeps_q.items()},
        "levels_mlp": {str(k): v for k, v in keeps_m.items()},
        "layer_names": layer_names,
        "hessian_tokens": "full fused prompt prefill, no labels",
        "error_accumulation": "median (expected level written back per block)",
    }, indent=2))

    t_all = time.time()
    for i in range(n_blocks):
        t0 = time.time()
        o_proj = layers[i].self_attn.o_proj
        down = layers[i].mlp.down_proj
        hooks = {"o": tyr.HessianHook(o_proj), "m": tyr.HessianHook(down)}
        for item in store:
            prefill(model, item)
        for h in hooks.values():
            h.remove()
        t_fwd = time.time() - t0

        for mod, hook, keeps, n_groups, upd in (
            (o_proj, hooks["o"], keeps_q, tc.num_attention_heads, 1),
            (down, hooks["m"], keeps_m, tc.intermediate_size, 16),
        ):
            suffix = "self_attn.o_proj" if mod is o_proj else "mlp.down_proj"
            name = f"layers.{i:02d}.{suffix}"
            gs = mod.in_features // n_groups
            if args.selection == "dual":
                scores = (dual_q if mod is o_proj else dual_m)[i]
                keep_sets = {}
                for keep in set(keeps.values()):
                    cut_g = set(tyr.cut_lowest(scores, n_groups - keep).tolist())
                    kept_g = [g for g in range(n_groups) if g not in cut_g]
                    keep_sets[keep] = [g * gs + d for g in kept_g for d in range(gs)]
                sols = tyr.reconstruct_levels(mod, hook.H, keep_sets, damp=args.damp)
            else:
                sols = tyr.prune_levels(mod, hook.H, n_groups, sorted(set(keeps.values())),
                                        update_iter=upd, damp=args.damp)
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            for lv, keep in keeps.items():
                w = sols[keep]
                n_zero = int((w.abs().sum(0).reshape(n_groups, gs).sum(1) == 0).sum())
                assert n_zero == n_groups - keep, (name, lv, n_zero, keep)
                torch.save(w.to(torch.bfloat16).cpu(), d / f"{lv}.pth")
            # error accumulation: expected (level-0) weight goes into the model
            mod.weight.data.copy_(sols[keeps[0]].to(mod.weight.dtype))
        del hooks
        torch.cuda.empty_cache()
        print(f"[block {i + 1}/{n_blocks}] fwd {t_fwd:.0f}s total "
              f"{time.time() - t0:.0f}s", flush=True)

    (out_dir / "summary.txt").write_text(
        f"tyr supernet: {n_blocks} blocks x {len(keeps_q)}/{len(keeps_m)} levels, "
        f"{len(calib)} clips, {time.time() - t_all:.0f}s\n"
        f"level-0 budget: cut {args.head_cut}/32 heads, {args.mlp_cut}/12288 channels "
        f"per layer (u40_v2)\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
