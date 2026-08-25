"""Does the grad-enabled Euler loop in expert_infer_grads_stepwise match the real sampler?

`FlowMatching.sample` is decorated `@torch.no_grad`, so measuring per-step gradients along
the inference path required reimplementing its Euler integration. A reimplementation that
silently drifts from the sampler would make every inference-path number describe a
trajectory the model never takes, so it is checked here rather than assumed: both are given
the SAME initial noise and the same cache, and their final actions are compared.

Gates are all-ones during the check, so any difference is integration, not masking.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/verify_step_euler.py --gpu 4 --num-clips 2
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_step_importance import build_cache  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=2)
    ap.add_argument("--exp-id", default="verify_step_euler")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--reserve-gb", type=float, default=36.0)
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

    ec = model.expert.config
    dims = model.action_space.get_action_space_dims()
    rows = []
    for clip_id, t0 in calib:
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0))
        seed = sc.clip_seed(args.seed, clip_id)
        inputs, cache, rope_deltas, _ = build_cache(model, processor, data, args, seed)
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)

        # draw x0 exactly as FlowMatching._euler would, then hand the same draw to both
        torch.cuda.manual_seed_all(seed)
        x0 = torch.randn(1, *dims, device="cuda")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # the sampler draws its own x0 from the cuda RNG; reseeding right before means
            # it draws this same x0
            torch.cuda.manual_seed_all(seed)
            ref = lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask,
                                         seed=seed).float()

        gt_xy = data["ego_future_xyz"][0, 0, :, :2].to("cuda").float()
        gates = pl.StepGates(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                             ec.intermediate_size, args.fm_steps, "cuda", torch.float32)
        _, _, _, mine = pl.expert_infer_grads_stepwise(
            model, cache, rope_deltas, prefill, gates, gt_xy,
            inputs["ego_history_xyz"], inputs["ego_history_rot"], seed,
            n_steps=args.fm_steps, x0=x0)
        gates.remove()

        d = (mine.float() - ref).abs()
        scale = ref.abs().mean().item()
        rows.append({"clip_id": clip_id, "max_abs_diff": float(d.max()),
                     "mean_abs_diff": float(d.mean()), "ref_scale": scale,
                     "rel": float(d.max() / max(scale, 1e-8))})
        print(f"{clip_id} max|diff| {d.max():.3e}  mean {d.mean():.3e}  "
              f"(action scale {scale:.3e}, rel {rows[-1]['rel']:.3e})", flush=True)
        del cache, inputs

    worst = max(r["rel"] for r in rows)
    # bf16 autocast through 10 sequential expert forwards; anything at 1e-2 or below is
    # accumulation order, anything larger means the loops actually differ
    verdict = "PASS" if worst < 1e-2 else "FAIL"
    print(f"worst relative diff {worst:.3e} -> {verdict}", flush=True)
    (out_dir / "summary.txt").write_text(
        f"euler reimplementation vs FlowMatching.sample, n={len(rows)}\n"
        f"worst relative diff {worst:.3e} -> {verdict}\n"
        + "\n".join(f"  {r['clip_id']} rel {r['rel']:.3e}" for r in rows) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"rows": rows, "worst_rel": worst, "verdict": verdict,
         "n_steps": args.fm_steps}, indent=2))
    print("saved ->", out_dir, flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
