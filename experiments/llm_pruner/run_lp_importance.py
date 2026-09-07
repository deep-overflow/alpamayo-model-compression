# Ported from soowon's alpamayo1.5 research tree
#   /home/cvlab21/project/soowon/vla-ad/alpamayo1.5/experiments/llm_pruner/run_lp_importance.py
# copied 2026-09-06 to run the second-order (param_mix) arm that was never executed there.
# That tree had itself vendored analysis_lib / sample_cache / expert_per_clip / slim_lib
# FROM this repo on 2026-08-09; those copies are deliberately NOT brought back -- this
# repo's versions are newer (expert_per_clip carries an AcceleratorError fix, slim_lib a
# pinned MODEL_REV and write_state, sample_cache calib_samples()), and this code was
# written against them in the first place.
# Changes from the original are marked `PORT:`; the importance formulas are otherwise
# untouched, because upstream fidelity is the whole point of this file.
"""LLM-Pruner stage 2: grouped-Taylor importance over the calibration set.

Upstream (`hf_prune.py:128-148`) takes 10 bookcorpus sequences of 64 tokens, runs one
`model(x, labels=x).loss.backward()`, and reads `weight.grad` off every prunable tensor.
Here the calibration set is `calib_100` -- 100 driving clips from the official train
split, distribution-matched, disjoint from val_500/test_500/ood -- and the loss is the
model's own chain-of-causation NLL (objective A), optionally plus the action expert's
flow-matching MSE (objective B, not upstream-canonical; see lp_objectives.py).

Gradients are summed elementwise across all clips before any importance is computed, so
the semantics match upstream's single-batch backward.  6.6 B fp32 cannot sit next to a
22 GB model on a 49 GB card, so the sum lives in host RAM and each weight's `.grad` is
consumed and dropped mid-backward (lp_core.GradAccumulator).

Writes to REPO/outputs/<exp-id>/:
  lp_importance.npz  q/mlp scores for every criterion arm
  grad_sums.npz      the raw accumulated gradients (optional, --save-grads; ~22 GB)
  config.json, metrics.json

Usage:
  bash experiments/llm_pruner/run_host.sh 30 experiments/llm_pruner/run_lp_importance.py \
      --gpu 4 --num-clips 100 --exp-id lp_importance_v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib  # noqa: E402
import lp_core as lp  # noqa: E402
import lp_objectives as obj  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def process_clip(model, processor, data, seed, args, accum, objectives):
    """One clip -> one backward per enabled objective, straight into the accumulator."""
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]
    del roll

    rec = {"prompt_len": prompt_len, "coc_len": coc_end - coc_start, "fm_loss": None}

    # ---- objective A: CoC NLL (LLM-Pruner canonical) ----------------------------------
    if "coc" in objectives:
        accum.begin("coc")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _, _ = obj.vlm_forward(model, seq_tf, inputs["tokenized_data"],
                                           use_cache=False)
        nll = obj.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
        nll.backward()
        accum.end()
        rec["coc_nll"] = float(nll.detach())
        del hidden, nll
        _clear_grads(model)

    # ---- objective B: trajectory flow matching (through the KV cache) ------------------
    if "traj" in objectives:
        accum.begin("traj")
        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, cache, rope_deltas = obj.vlm_forward(model, seq_tf, inputs["tokenized_data"],
                                                    use_cache=True)
        n_vlm = len(lp.text_layers(model))
        cache_t = obj.retain_cache_grads(cache, n_vlm)
        prefill = cache.get_seq_length()
        fm_loss, grads = obj.traj_fm_backward(model, cache, rope_deltas, x1,
                                              args.fm_steps, seed, prefill)
        # one chained VLM backward instead of fm_steps of them
        ts, gs = [], []
        for (k, v), (gk, gv) in zip(cache_t, grads):
            for tensor, grad in ((k, gk), (v, gv)):
                if grad is not None:
                    ts.append(tensor)
                    gs.append(grad.to(tensor.dtype))
        assert ts, "trajectory objective produced no cache gradients"
        torch.autograd.backward(ts, gs)
        accum.end()
        rec["fm_loss"] = fm_loss
        del cache, cache_t, grads, ts, gs, x1
        _clear_grads(model)

    rec["peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    return rec


def _clear_grads(model):
    """The hooks null each prunable grad as it is produced; catch anything else."""
    for p in model.parameters():
        p.grad = None


def dual_arms(arms, variants, scope):
    """`dual_<variant>` = per-layer max(rank(traj), rank(coc)) -- docs/llm_pruner_dual.md §2.

    Only emitted when both single-objective arms are present, so a `--objectives coc` run
    still writes a valid npz; `lp_prune.py` reports the available arms if one is missing.
    """
    out = {}
    for variant in variants:
        parts = [f"traj_{variant}", f"coc_{variant}"]   # reference order; max() commutes
        if not all(p in arms for p in parts):
            continue
        out[f"dual_{variant}"] = (
            lp.fuse_max([arms[p][0] for p in parts], scope),
            lp.fuse_max([arms[p][1] for p in parts], scope),
        )
    return out


def build_arms(model, accum, scope, args, weights):
    arms = {}
    for name in accum.objectives:
        if accum.n_samples[name] == 0:
            continue
        for variant in args.taylor:
            if variant in ("param_second", "param_mix") and accum.grad_sq is None:
                continue
            q, m = lp.group_importance(model, accum, name, taylor=variant,
                                       group_reduction=args.grouping_strategy,
                                       weights=weights, first_order=args.first_order)
            assert np.any(q[scope]) and np.any(m[scope]), \
                f"arm {name}_{variant} is identically zero -- no gradient reached the weights"
            arms[f"{name}_{variant}"] = (q, m)
    arms.update(dual_arms(arms, args.taylor, scope))
    for p, tag in ((2, "l2"), (1, "l1")):
        arms[f"mag_{tag}"] = lp.magnitude_importance(model, scope, p=p, weights=weights)
    arms["random"] = lp.random_importance(model, scope, seed=args.seed)
    return arms


def save(out_dir, arms, records, geom, scope):
    flat = {}
    for name, (q, m) in arms.items():
        flat[f"{name}_q"] = q
        flat[f"{name}_mlp"] = m
    np.savez(out_dir / "lp_importance.npz", scope=np.asarray(scope), **flat)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"n_clips": len(records), "geometry": geom, "scope": scope,
         "arms": sorted(arms), "per_clip": records}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", default="lp_importance_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--objectives", default="coc,traj",
                    help="comma list of coc,traj; coc alone is LLM-Pruner canonical")
    ap.add_argument("--taylor", default="param_first,vectorize",
                    help=f"comma list from {lp.TAYLOR_VARIANTS}; param_second/param_mix "
                         "additionally need --second-order")
    ap.add_argument("--grouping-strategy", default="sum", choices=lp.GROUP_REDUCTIONS)
    ap.add_argument("--second-order", action="store_true",
                    help="also accumulate sum(g^2) for param_second/param_mix (doubles host RAM)")
    # PORT: see lp_core._salience. Only param_mix is affected; "sum" reproduces the
    # original tree's stored arrays, "mean" is what upstream's two-pass recipe computes.
    ap.add_argument("--first-order", default="mean", choices=("mean", "sum"),
                    help="scale of the first-order term: per-sample mean (upstream) or "
                         "the accumulator's raw sum (original port). param_first and "
                         "param_second rank identically either way")
    ap.add_argument("--layer-start", type=int, default=4,
                    help="LLM-Pruner --block_*_layer_start (upstream: skip the first four)")
    ap.add_argument("--layer-end", type=int, default=34,
                    help="LLM-Pruner --block_*_layer_end, exclusive (upstream: skip the last two)")
    # lp_importance_v1 peaked at 42.69 GiB, so the old 40.0 default let the pass start on a
    # card it would later OOM on (the operator had to override it by hand)
    ap.add_argument("--reserve-gb", type=float, default=46.0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--save-grads", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=50,
                    help="rebuild the arms mid-run every N clips; the rebuild reduces "
                         "~5.5B params per arm, so keep it rare")
    args = ap.parse_args()
    args.taylor = [t for t in args.taylor.split(",") if t]
    objectives = [o for o in args.objectives.split(",") if o]
    for t in args.taylor:
        assert t in lp.TAYLOR_VARIANTS, t
    # build_arms() silently skips these when grad_sq was never accumulated, so the pass
    # would run for ~80 minutes and then write an npz with the requested arm missing --
    # surfacing only as a bare KeyError in lp_prune.py. Refuse up front instead.
    need_g2 = [t for t in args.taylor if t in ("param_second", "param_mix")]
    if need_g2 and not args.second_order:
        raise SystemExit(
            f"--taylor {','.join(need_g2)} needs the elementwise Fisher term; pass "
            f"--second-order (it doubles host RAM: ~41 GB -> ~82 GB) or drop those variants.")

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_clips(REPO, args.calib_manifest)[: args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV,
                                        dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    # k/v are frozen, so without this the layer-0 cache tensors carry no autograd graph
    # and the trajectory objective cannot be chained back into the VLM.
    model.vlm.enable_input_require_grads()

    geom = lp.text_geometry(model)
    scope = lp.layer_scope(geom["n_layers"], args.layer_start, args.layer_end)
    accum = lp.GradAccumulator(model, scope, objectives=objectives,
                               second_order=args.second_order)
    weights = lp.WeightCache()  # fp32 CPU mirror, filled on the first arm build
    host_gb = accum.nbytes() / 1024**3
    print(f"scope=layers[{scope[0]}..{scope[-1]}] ({len(scope)} layers), "
          f"host accumulators {host_gb:.1f} GB", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "method": "LLM-Pruner (arXiv:2305.11627) block-wise, grouped Taylor",
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "tower": "vlm.model.language_model (Cosmos-Reason2-8B / Qwen3-VL text)",
        "groups": {"attn": [m for m, _ in lp.ATTN_MEMBERS],
                   "mlp": [m for m, _ in lp.MLP_MEMBERS],
                   "frozen": list(lp.FROZEN_MEMBERS)},
        "objectives": objectives, "taylor": args.taylor,
        "grouping_strategy": args.grouping_strategy, "second_order": args.second_order,
        "first_order": args.first_order,
        "layer_scope": [args.layer_start, args.layer_end], "geometry": geom,
        "num_clips": len(calib), "clip_ids": calib, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "calib_manifest": args.calib_manifest, "fm_steps": args.fm_steps,
        "gpu": torch.cuda.get_device_name(device),
        "host_accumulator_gb": round(host_gb, 2),
    }, indent=2))

    records = []
    for ci, clip_id in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for("calib", clip_id, sc.CALIB_T0))
        torch.cuda.reset_peak_memory_stats()
        rec = process_clip(model, processor, data, sc.clip_seed(args.seed, clip_id),
                           args, accum, objectives)
        rec["clip_id"] = clip_id
        records.append(rec)
        fm = "-" if rec["fm_loss"] is None else f"{rec['fm_loss']:.4f}"
        print(f"[{ci + 1}/{len(calib)}] {clip_id} coc={rec['coc_len']} fm={fm} "
              f"peak={rec['peak_gb']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % args.snapshot_every == 0 and ci + 1 < len(calib):
            save(out_dir, build_arms(model, accum, scope, args, weights), records, geom, scope)

    arms = build_arms(model, accum, scope, args, weights)
    save(out_dir, arms, records, geom, scope)
    if args.save_grads:
        for name in objectives:
            np.savez(out_dir / f"grad_sums_{name}.npz", **accum.state(name))
    accum.remove()
    print(f"saved {len(arms)} arms -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
