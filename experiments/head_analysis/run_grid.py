"""Criterion x allocation grid -- which factor actually preserved the reasoning channel?

The shipped comparison is confounded: slim_cocsafe_r30 and slim_integrated_mag differ in
(i) the within-layer criterion (dual max-rank vs trajectory-only Taylor), (ii) the layerwise
budget (uniform vs late-heavy) and (iii) layer-35 removal, all at once. Their layers 0-17
profiles are nearly identical, so everything that separated them lives in layers 18-35 and
cannot currently be attributed to any one factor.

This run fills the 2x4 grid at a matched budget, holding (iii) fixed (no layer-35 drop, no KV
drop anywhere) so only (i) x (ii) vary:

  criterion   traj  = I_traj Taylor
              dual  = per-layer max(rank(I_traj), rank(I_CoC))
  allocation  uniform = same prune ratio every layer
              late    = slim_integrated_mag's realized per-layer profile (the shipped shape)
              agree   = keep_l proportional to the union of both objectives' top-q sets,
                        i.e. budget follows channel agreement (analyze_agreement.py)
              depthprior = linear tilt with depth, slope matched to agree's early/late ratio.
                        Uses NO importance and no CoC information at all.

All eight cells remove the same fraction of VLM q/o+MLP params, so any difference is
attribution, not compression. Two cells decide the reading:
  traj x agree      survives -> the layerwise budget is the mechanism, not the unit criterion
                       (note: agree still derives that budget from both objectives, so this
                        means CoC information is needed at layer granularity, not per unit)
  traj x depthprior survives -> even the layer-level CoC information is unnecessary and the
                       whole effect is "protect the late layers".

Endpoints per clip and config:
  free-running CoC degeneracy  -- the endpoint that exposed the collapse closed-loop (86.3%)
  teacher-forced CoC NLL       -- cheap continuous measure on the baseline's own CoC
  minADE / minFDE over K seeds -- action channel, paired seeds shared across configs

Usage:
  bash experiments/head_analysis/run_retry_host.sh 20 experiments/head_analysis/run_grid.py \
      --gpu 0 --exp-id grid_smoke --num-clips 2 --k 2
  bash experiments/head_analysis/run_retry_host.sh 20 experiments/head_analysis/run_grid.py \
      --gpu 0 --exp-id grid_v1 --num-clips 50 --k 8
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
from analyze_agreement import fill_tail, match_budget, per_layer_agreement  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_cocsafe import rank_norm  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


BANDS = [(0, 5), (6, 17), (18, 29), (30, 35)]
P_HEAD, P_MLPC = 2 * 4096 * 128, 3 * 4096


def depth_prior_keep(target, n_layers, tilt, n_heads, intermediate):
    """keep_l = alpha * (1 + tilt * l/(L-1)), alpha solved so the budget matches `target`.

    A monotone tilt with depth and nothing else -- no importance, no CoC information at all.
    This is the control that separates "the layerwise budget matters" from "the budget has to
    be derived from both objectives": if it matches `agree`, a plain depth prior suffices.
    """
    frac = np.arange(n_layers) / max(n_layers - 1, 1)
    shape = 1 + tilt * frac
    full = n_layers * (n_heads * P_HEAD + intermediate * P_MLPC)
    lo, hi = 0.01, 1.0 / shape.max()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        keep = np.clip(mid * shape, 0, 1)
        removed = 1 - (keep * (n_heads * P_HEAD + intermediate * P_MLPC)).sum() / full
        if removed > target:
            lo = mid
        else:
            hi = mid
    return np.clip(0.5 * (lo + hi) * shape, 0, 1)


def match_tilt(keep_ref, n_layers):
    """Tilt whose early/late band ratio reproduces keep_ref's, so the two profiles differ
    only in within-profile detail rather than in overall slope."""
    frac = np.arange(n_layers) / max(n_layers - 1, 1)
    (e0, e1), (l0, l1) = BANDS[0], BANDS[-1]
    r = float(np.mean(keep_ref[l0:l1 + 1]) / np.mean(keep_ref[e0:e1 + 1]))
    be, bl = float(np.mean(frac[e0:e1 + 1])), float(np.mean(frac[l0:l1 + 1]))
    return (r - 1) / (bl - r * be)


def allocations(imp, ref_meta, n_layers, n_heads, intermediate, keep_q):
    """Per-layer prune ratios for the four allocations, all at the reference budget.

    Returns {name: (ratios_q, ratios_mlp)} plus the matched removed fraction. The budget is
    expressed in q/o+MLP params, so a uniform ratio r removes exactly fraction r of them.
    """
    kq_ref = np.array([len(layer["q"]) / n_heads for layer in ref_meta["vlm"]])
    km_ref = np.array([len(layer["mlp"]) / intermediate for layer in ref_meta["vlm"]])
    full = n_layers * (n_heads * P_HEAD + intermediate * P_MLPC)
    target = 1 - (kq_ref * n_heads * P_HEAD + km_ref * intermediate * P_MLPC).sum() / full

    u_q = fill_tail(per_layer_agreement(imp["coc_vlm_q"], imp["traj_vlm_q"], keep_q)[1])
    u_m = fill_tail(per_layer_agreement(imp["coc_vlm_mlp"], imp["traj_vlm_mlp"], keep_q)[1])
    alpha, got, kq_ag, km_ag = match_budget(target, u_q, u_m, n_layers)

    tilt = match_tilt(kq_ag, n_layers)
    kq_dp = depth_prior_keep(target, n_layers, tilt, n_heads, intermediate)

    allocs = {
        "uniform": (np.full(n_layers, target), np.full(n_layers, target)),
        "late": (1 - kq_ref, 1 - km_ref),
        "agree": (1 - kq_ag, 1 - km_ag),
        "depthprior": (1 - kq_dp, 1 - kq_dp),
    }
    info = {"target_removed": float(target), "agree_alpha": float(alpha),
            "agree_removed": float(got), "depthprior_tilt": float(tilt),
            "union_q": u_q.tolist(), "union_mlp": u_m.tolist()}
    return allocs, info


def grid_configs(imp, ref_meta, n_layers, n_heads, intermediate, keep_q):
    """baseline + {traj, dual} x 4 allocations, matched budget, no KV/layer-35 factor."""
    allocs, info = allocations(imp, ref_meta, n_layers, n_heads, intermediate, keep_q)
    crits = {
        "traj": (imp["traj_vlm_q"], imp["traj_vlm_mlp"]),
        "dual": (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),
                 np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))),
    }
    full = n_layers * (n_heads * P_HEAD + intermediate * P_MLPC)

    cfgs = [("baseline", {"kind": "baseline"}, None, None)]
    for crit, (sq, sm) in crits.items():
        for alloc, (rq, rm) in allocs.items():
            q = ml.select_mask_ratios(sq, rq)
            m = ml.select_mask_ratios(sm, rm)
            removed = 1 - (q.sum(1) * P_HEAD + m.sum(1) * P_MLPC).sum() / full
            cfgs.append((f"{crit}_{alloc}",
                         {"kind": "grid", "criterion": crit, "allocation": alloc,
                          "removed_qo_mlp": float(removed),
                          "keep_q_bands": [float(q[a:b + 1].mean()) for a, b in BANDS]},
                         q, m))
    return cfgs, info


@torch.no_grad()
def eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds, max_gen):
    """Free-running CoC + teacher-forced NLL + K denoisings, all under the active mask."""
    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed_all(seeds[0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
    text = model.tokenizer.decode(roll["sequences"][0, roll["prompt_len"]:roll["eos_pos"]],
                                  skip_special_tokens=True)
    health = el.coc_degenerate(text)
    del roll

    attention_mask = torch.ones_like(seq_tf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=attention_mask,
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end]).item()
        cache = out.past_key_values
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device=seq_tf.device)
        prefix_mask = torch.ones(1, prefill, device=seq_tf.device, dtype=torch.long)
        preds = []
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, out.rope_deltas, offset, prefix_mask,
                                            seed=s)
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    ade, fde = el.min_metrics(np.stack(preds), gt_xy)
    del out, cache
    return ade, fde, nll, health, text


def save(out_dir, cfgs, results, buckets, clip_ids, n):
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "buckets": buckets, "clip_ids": clip_ids,
        "configs": [c[0] for c in cfgs], "meta": {c[0]: c[1] for c in cfgs},
        "per_clip": results,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=40)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--ref-slim", type=str, default="slim_integrated_mag")
    ap.add_argument("--keep-q", type=float, default=0.5,
                    help="top-q of each objective treated as must-keep for the agree allocation")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--clip-offset", type=int, default=0)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split[args.split][args.clip_offset : args.clip_offset + args.num_clips]
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    ref_meta = json.loads((REPO / "outputs" / args.ref_slim / "slim_meta.json").read_text())

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
    layers = model.vlm.model.language_model.layers
    masks = ml.PruneMasks(layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size, "cuda")
    cfgs, info = grid_configs(imp, ref_meta, tc.num_hidden_layers, tc.num_attention_heads,
                              tc.intermediate_size, args.keep_q)
    print(f"{len(cfgs)} configs x {len(clips)} clips x K={args.k}", flush=True)
    for name, meta, *_ in cfgs:
        if meta["kind"] == "grid":
            print(f"  {name:16s} removed {meta['removed_qo_mlp']:.3f} "
                  f"keep_q by band {[round(v, 3) for v in meta['keep_q_bands']]}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "purpose": "criterion x allocation grid at matched budget",
        "eval_split": args.split, "num_clips": len(clips), "clip_ids": clips,
        "clip_offset": args.clip_offset, "k_samples": args.k, "seed": args.seed,
        "importance_from": args.importance, "ref_slim": args.ref_slim, "keep_q": args.keep_q,
        "held_fixed": "no KV group drop, no layer-35 removal, expert tower untouched",
        "allocation_info": info,
        "configs": [{"name": n, **m} for n, m, *_ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    results = {n: {"ade": [], "fde": [], "nll": [], "degenerate": [], "degenerate_strict": [],
                   "empty": [], "soup": [], "coc_len": [], "char_run": [], "text": []}
               for n, *_ in cfgs}
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

        # reference CoC from the unmasked model, teacher-forced target for every config
        masks.reset()
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        for name, _, q, m in cfgs:
            masks.set(q=q, mlp=m)
            ade, fde, nll, health, text = eval_config(model, inputs, seq_tf, coc_start, coc_end,
                                                      gt_xy, seeds, args.max_gen)
            r = results[name]
            r["ade"].append(ade)
            r["fde"].append(fde)
            r["nll"].append(nll)
            r["coc_len"].append(health["len"])
            r["char_run"].append(health["char_run"])
            r["text"].append(text)
            for k in ("degenerate", "degenerate_strict", "empty", "soup"):
                r[k].append(health[k])
        masks.reset()

        b = results["baseline"]
        print(f"[{ci + 1}/{len(clips)}] {clip_id} {buckets[-1]:10s} "
              f"baseMinADE={np.mean(b['ade']):.3f} baseDegen={np.mean(b['degenerate']):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(clips):
            save(out_dir, cfgs, results, buckets, clip_ids_done, ci + 1)
    save(out_dir, cfgs, results, buckets, clip_ids_done, len(clips))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
