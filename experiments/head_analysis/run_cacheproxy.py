"""G1 of plans/2026-08-29_cache-targeted-reconstruction.md: how much of the dense VLM's
cache does a slim checkpoint preserve, weighted by what the expert is sensitive to, and
what does that cost the DENSE expert?

Two models in one process: the dense model (rollout fixes the CoC; its expert denoises
every cache) and a slim checkpoint (load_slim) whose VLM produces the pruned cache for
the same teacher-forced text. Per clip:
  * per (layer, KV group) cache divergence rel_v = ||V_P - V_D|| / ||V_D|| (run_cachediff's
    Stage B measure) and its sensitivity-weighted sum  sum_{l,g} s[l,g] rel_v[l,g]^2
    with s from outputs/cacheuse_v1/maps_swap.npz (Stage C swap damage / shift)
  * A00 (dense cache) and A10 (pruned cache), both denoised by the dense expert with K
    shared seeds -> paired dminADE@K, the "cache-only" cost of the VLM cut
Gate G1: sensitivity-weighted shift down >= 50% vs dual_u40_v2 and the A10-A00 median CI
including 0. dual / dualr / dualrc_* are run one slim model at a time (same clips, same
seeds) so the arms are clip-paired.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 60 experiments/head_analysis/run_cacheproxy.py \
      --gpu 0 --slim outputs/slim_dualrc_u40_s16 --num-clips 200 --exp-id cacheproxy_dualrc_s16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import sample_cache as sc  # noqa: E402
import slim_lib as sl  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_cachediff import denoise_minade, tf_forward  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


@torch.no_grad()
def divergence(cache_d, cache_p, n_layers):
    """(L, G) relative V and K shift of the pruned cache, all positions."""
    rel_v = np.zeros((n_layers, 8))
    rel_k = np.zeros((n_layers, 8))
    for li in range(n_layers):
        kd, vd = lib.cache_layer_kv(cache_d, li)
        kp, vp = lib.cache_layer_kv(cache_p, li)
        kd, vd, kp, vp = (x[0].float() for x in (kd, vd, kp, vp))  # (G, T, D)
        rel_v[li] = ((vp - vd).norm(dim=(1, 2)) / vd.norm(dim=(1, 2)).clamp_min(1e-12)).cpu()
        rel_k[li] = ((kp - kd).norm(dim=(1, 2)) / kd.norm(dim=(1, 2)).clamp_min(1e-12)).cpu()
    return rel_v, rel_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--slim", required=True, help="slim checkpoint dir (relative to repo)")
    ap.add_argument("--num-clips", type=int, default=200)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--manifest", default="indist_500")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--sensitivity", default="cacheuse_v1/maps_swap.npz")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO / "outputs" / args.sets_id / f"{args.manifest}.parquet")
    rows = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)} for r in df.itertuples()]
    rows = rows[args.clip_offset: args.clip_offset + args.num_clips]
    sens = np.load(REPO / "outputs" / args.sensitivity)["sensitivity"]  # (36, 8), nan at layer 0
    sens = np.nan_to_num(sens)

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
    slim = sl.load_slim(REPO / args.slim, device="cuda")
    slim.eval()
    for p in slim.parameters():
        p.requires_grad_(False)
    lib.set_vlm_attn_impl(slim, "sdpa")
    smeta = json.loads((REPO / args.slim / "slim_meta.json").read_text())
    n_layers = model.vlm.config.text_config.num_hidden_layers
    print(f"slim {args.slim}: {smeta['config']} removed {smeta['params']['removed']:,}; "
          f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    res = {"clip_ids": [], "buckets": [], "nll_dense": [], "nll_slim": [], "ade_A00": [],
           "ade_A10": [], "fde_A00": [], "fde_A10": [], "wshift_v": [], "wshift_k": [],
           "shift_v_mean": []}
    rel_v_sum = np.zeros((n_layers, 8))
    rel_k_sum = np.zeros((n_layers, 8))
    meta = {"model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
            "slim": args.slim, "slim_config": smeta["config"], "manifest": args.manifest,
            "clip_offset": args.clip_offset, "k": args.k, "seed": args.seed,
            "sensitivity": args.sensitivity, "gpu": torch.cuda.get_device_name(device),
            "plan": "plans/2026-08-29_cache-targeted-reconstruction.md"}

    def save(n):
        (out_dir / "metrics.json").write_text(json.dumps({**meta, "n_clips": n, **res}, indent=1))
        np.savez(out_dir / "cacheproxy.npz", rel_v=rel_v_sum / max(n, 1),
                 rel_k=rel_k_sum / max(n, 1), sensitivity=sens)

    for ci, r in enumerate(rows):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, r["clip_id"], r["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        base = sc.clip_seed(args.seed, r["clip_id"])
        seeds = [base + k for k in range(args.k)]
        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll
            cache_d, rope_d, nll_d = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
            cache_p, rope_p, nll_p = tf_forward(slim, seq_tf, inputs, coc_start, coc_end)
        prefill = cache_d.get_seq_length()
        assert cache_p.get_seq_length() == prefill and torch.equal(rope_p, rope_d)
        rel_v, rel_k = divergence(cache_d, cache_p, n_layers)
        rel_v_sum += rel_v
        rel_k_sum += rel_k
        ade0, fde0 = denoise_minade(model, inputs, cache_d, rope_d, prefill, gt_xy, seeds)
        ade1, fde1 = denoise_minade(model, inputs, cache_p, rope_d, prefill, gt_xy, seeds)
        res["clip_ids"].append(r["clip_id"])
        res["buckets"].append(el.bucket(gt_xy))
        res["nll_dense"].append(nll_d)
        res["nll_slim"].append(nll_p)
        res["ade_A00"].append(float(ade0))
        res["ade_A10"].append(float(ade1))
        res["fde_A00"].append(float(fde0))
        res["fde_A10"].append(float(fde1))
        res["wshift_v"].append(float((sens * rel_v ** 2).sum()))
        res["wshift_k"].append(float((sens * rel_k ** 2).sum()))
        res["shift_v_mean"].append(float(rel_v[1:].mean()))
        del cache_d, cache_p
        torch.cuda.empty_cache()
        print(f"[{ci + 1}/{len(rows)}] {r['clip_id'][:8]} {res['buckets'][-1]:10s} "
              f"A00={ade0:.3f} A10={ade1:.3f} d={ade1 - ade0:+.3f} wshift_v={res['wshift_v'][-1]:.4f} "
              f"shift_v={res['shift_v_mean'][-1]:.3f} nll {nll_d:.3f}/{nll_p:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(rows):
            save(ci + 1)
    d = np.array(res["ade_A10"]) - np.array(res["ade_A00"])
    (out_dir / "summary.txt").write_text(
        f"cache proxy for {args.slim} ({smeta['config']}), {len(rows)} clips, K={args.k}\n"
        f"A10-A00 dminADE@{args.k}: median {np.median(d):+.4f}, mean {d.mean():+.4f}\n"
        f"sensitivity-weighted V shift: mean {np.mean(res['wshift_v']):.4f}; "
        f"plain V shift (layers 1-35 mean rel_v): {np.mean(res['shift_v_mean']):.4f}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
