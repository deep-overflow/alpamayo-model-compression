"""Denoising-step decomposition of the action expert's trajectory importance.

Why this exists: on the expert tower, plain weight magnitude beats the trajectory Taylor
criterion at r10-r25 (`outputs/expert_abl_summary`), while on the VLM tower the same
criterion family wins by a landslide (Wanda test_500 minADE 2.77 vs dual 0.74). The
structural difference between the towers is that the expert runs ten times, at ten
different diffusion times, behind one static mask. Two mechanisms could turn that into a
bad criterion, and this run measures both:

  1. sign cancellation -- `expert_fm_grads` never zeroes the gate grads between steps, so
     the shipped score is |sum_s dL_s/dg| while the clip axis uses sum_clips |.|. Steps of
     opposite sign erase each other on the step axis only.
  2. path mismatch -- the criterion measures on the training path (x_t on the straight line
     to the GT action, t = 0.05 .. 0.95) but the model is deployed on its own Euler
     iterates (t = 0.0 .. 0.9, error accumulating).

Modes:
  --mode fm     per-step gradients on the training path (`expert_fm_grads_stepwise`).
                --noise-mode per_step reproduces the shipped construction exactly, so
                summing the per-step grads and taking |.| must reproduce importance.npz
                (gate G0). --noise-mode shared pairs the steps on one noise draw, which is
                what the step-to-step comparison should use.
  --mode infer  per-step gradients along the model's own Euler trajectory, w.r.t. the final
                trajectory error (`expert_infer_grads_stepwise`). K noise draws per clip.

Selection signal is measured on calib_100 only, never on an evaluation set.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 240 \
      experiments/head_analysis/run_step_importance.py --gpu 4 \
      --mode fm --noise-mode per_step --exp-id stepimp_fm_perstep
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
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def build_cache(model, processor, data, args, seed):
    """Rollout + teacher-forced VLM forward, exactly as run_importance does it.

    Under no_grad throughout: nothing here needs a graph, because the expert measurement
    hangs off detached cache leaves. That is what keeps peak memory well under the 40.5 GB
    the dual-objective pass needs.
    """
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
        del roll
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
    return inputs, out.past_key_values, out.rope_deltas, coc_end - prompt_len


def reduce_draws(signed):
    """One draw's signed (S, L, U) grads -> the three per-clip aggregates.

    `shipped` is |sum_s g_s|, the quantity importance.npz stores; `abs_step` is |g_s| per
    step. Taking |.| per draw before averaging matches the clip axis, which does the same:
    damage from removing a unit does not cancel across draws just because its sign flips.
    """
    return {"abs_step": np.abs(signed), "signed_step": signed,
            "shipped": np.abs(signed.sum(0))}


def mean_dicts(ds):
    return {k: np.mean([d[k] for d in ds], axis=0) for k in ds[0]}


def run_fm(model, inputs, cache, rope_deltas, data, args, seed):
    """Per-step grads on the training path (one noise draw, as the shipped path uses)."""
    ec = model.expert.config
    gates = pl.UnitGates(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                         ec.intermediate_size, "cuda", torch.float32)
    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
    prefill = cache.get_seq_length()
    losses, q, mlp, kv_k, kv_v = pl.expert_fm_grads_stepwise(
        model, cache, rope_deltas, x1, args.fm_steps, seed, prefill, gates,
        noise_mode=args.noise_mode,
    )
    gates.remove()
    out = {k: reduce_draws(g) for k, g in
           (("q", q), ("mlp", mlp))}
    # kv_* already arrive as |k * dL/dk| per step, so there is no sign to preserve
    out["kv_k"] = {"abs_step": kv_k, "signed_step": kv_k, "shipped": kv_k.sum(0)}
    out["kv_v"] = {"abs_step": kv_v, "signed_step": kv_v, "shipped": kv_v.sum(0)}
    return out, float(np.mean(losses)), losses


def run_infer(model, inputs, cache, rope_deltas, data, args, seed):
    """Per-step grads along the model's own Euler path, over K noise draws."""
    ec = model.expert.config
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].to("cuda").float()  # (64, 2)
    prefill = cache.get_seq_length()
    draws = {"q": [], "mlp": []}
    losses = []
    for k in range(args.k):
        gates = pl.StepGates(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                             ec.intermediate_size, args.fm_steps, "cuda", torch.float32)
        loss, q, mlp, _ = pl.expert_infer_grads_stepwise(
            model, cache, rope_deltas, prefill, gates, gt_xy,
            inputs["ego_history_xyz"], inputs["ego_history_rot"],
            seed + k, n_steps=args.fm_steps,
        )
        gates.remove()
        losses.append(loss)
        draws["q"].append(reduce_draws(q))
        draws["mlp"].append(reduce_draws(mlp))
        del gates, q, mlp
    return {k: mean_dicts(v) for k, v in draws.items()}, float(np.mean(losses)), np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fm", "infer"], default="fm")
    ap.add_argument("--model", default=None,
                    help="slim ckpt dir (e.g. outputs/slim_dual_u40_v2) to measure the "
                         "expert importance conditioned on a pruned VLM's cache; "
                         "default = the dense base model")
    ap.add_argument("--noise-mode", choices=["per_step", "shared"], default="shared",
                    help="fm only; per_step reproduces the shipped construction (gate G0)")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=4, help="infer mode: noise draws per clip")
    ap.add_argument("--reserve-gb", type=float, default=34.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--no-perclip-mlp", action="store_true",
                    help="skip the ~600 MB per-clip MLP array (split-half then unavailable)")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    if args.model:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.model, device="cuda")
    else:
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    ec = model.expert.config
    S, L = args.fm_steps, ec.num_hidden_layers
    H, I = ec.num_attention_heads, ec.intermediate_size
    KV = model.vlm.config.text_config.num_key_value_heads

    keys = {"q": (S, L, H), "mlp": (S, L, I)}
    if args.mode == "fm":
        keys.update({"kv_k": (S, L, KV), "kv_v": (S, L, KV)})
    # abs_step  : mean over clips of |g_{u,s}|   -- the per-step importance
    # signed_step: mean over clips of g_{u,s}    -- diagnostics for the sign structure
    # shipped   : mean over clips of |sum_s g_s| -- exactly what importance.npz stores
    acc = {f"{k}_abs_step": np.zeros(s) for k, s in keys.items()}
    acc.update({f"{k}_signed_step": np.zeros(s) for k, s in keys.items()})
    acc.update({f"{k}_shipped": np.zeros(s[1:]) for k, s in keys.items()})
    per_clip = {"q_abs_step": [], "q_shipped": [], "mlp_abs_step": [], "mlp_shipped": [],
                "kv_k_abs_step": []}

    (out_dir / "config.json").write_text(json.dumps({
        "model": args.model or "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "per-denoising-step decomposition of the expert trajectory importance",
        "mode": args.mode, "noise_mode": args.noise_mode if args.mode == "fm" else None,
        "objective": ("flow-matching MSE vs GT action, per t" if args.mode == "fm"
                      else "MSE(final xy, GT xy) through the Euler chain"),
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib], "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "calib_manifest": args.calib_manifest, "cache": args.cache,
        "fm_steps": args.fm_steps, "k_draws": args.k if args.mode == "infer" else None,
        "max_gen": args.max_gen, "gpu": torch.cuda.get_device_name(device),
        "shapes": {k: list(v) for k, v in keys.items()},
    }, indent=2))

    records = []
    runner = run_fm if args.mode == "fm" else run_infer
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        torch.cuda.reset_peak_memory_stats()
        seed = sc.clip_seed(args.seed, clip_id)
        inputs, cache, rope_deltas, coc_len = build_cache(model, processor, data, args, seed)
        grads, loss, losses = runner(model, inputs, cache, rope_deltas, data, args, seed)

        for k, d in grads.items():                      # d: abs_step/signed_step/shipped
            for name, v in d.items():
                acc[f"{k}_{name}"] += v
        per_clip["q_abs_step"].append(grads["q"]["abs_step"].astype(np.float32))
        per_clip["q_shipped"].append(grads["q"]["shipped"].astype(np.float32))
        if "kv_k" in grads:
            per_clip["kv_k_abs_step"].append(grads["kv_k"]["abs_step"].astype(np.float32))
        if not args.no_perclip_mlp:
            # fp32, not fp16: these gradients sit at ~1e-7, which is fp16's subnormal range
            # (min normal 6.1e-5). Stored as fp16, 75% of the MLP entries underflowed to
            # exact zero and re-aggregating them carried a median relative error of 0.22 --
            # every per-clip MLP statistic computed from such a file is meaningless. Costs
            # 1.2 GB per run instead of 0.6 GB.
            per_clip["mlp_abs_step"].append(grads["mlp"]["abs_step"].astype(np.float32))
            per_clip["mlp_shipped"].append(grads["mlp"]["shipped"].astype(np.float32))

        peak = torch.cuda.max_memory_allocated() / 1024**3
        records.append({"clip_id": clip_id, "coc_len": int(coc_len), "loss": loss,
                        "losses": [round(float(x), 6) for x in losses],
                        "peak_gb": round(peak, 2)})
        del inputs, cache, grads
        print(f"[{ci + 1}/{len(calib)}] {clip_id} coc={coc_len} loss={loss:.4f} "
              f"peak={peak:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 10 == 0 or ci + 1 == len(calib):
            save(out_dir, acc, per_clip, records, ci + 1)
    save(out_dir, acc, per_clip, records, len(records))
    print("saved ->", out_dir, flush=True)


def save(out_dir, acc, per_clip, records, n):
    np.savez(out_dir / "step_importance.npz", **{k: v / max(n, 1) for k, v in acc.items()})
    stacks = {k: np.stack(v) for k, v in per_clip.items() if v}
    if stacks:
        np.savez(out_dir / "step_importance_perclip.npz", **stacks)
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": n, "per_clip": records},
                                                     indent=2))


if __name__ == "__main__":
    main()
