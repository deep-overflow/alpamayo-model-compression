"""Equivalence + latency benchmark of the functional expert-denoise fast path.

Per clip: rollout + TF forward fix the cache, then three denoise paths run with the
same seeds -- A: release path (cache append/crop), B: fast path uncompiled,
C: fast path torch.compile(reduce-overhead). Reports per-denoise (10-step) latency
medians and trajectory deviations of B/C vs A. Compile/capture cost is timed
separately (first C call per shape).

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/bench_fastdenoise.py \
      --gpu 4 --exp-id fastdenoise_base
  ... --slim-ckpt outputs/slim_integrated_mag --exp-id fastdenoise_slim_int
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
import fast_denoise as fd  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def to_xy(model, inputs, action):
    pred_xyz, _ = model.action_space.action_to_traj(
        action.float(), inputs["ego_history_xyz"][:, -1].float(),
        inputs["ego_history_rot"][:, -1].float(),
    )
    return pred_xyz[0, :, :2].cpu().numpy()  # (64, 2)


def timed(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim-ckpt", type=str, default=None)
    ap.add_argument("--num-clips", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--pad", type=int, default=128)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    if args.slim_ckpt:
        model = sl.load_slim(REPO / args.slim_ckpt, device="cuda")
    else:
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    n_layers = model.expert.config.num_hidden_layers

    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split["val"][: args.num_clips]

    res = {p: {"ms": [], "dxy": []} for p in ("release", "fast", "compiled")}
    compile_times = []
    graphs = {}  # padded kv_len -> GraphedDenoiser
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        seeds = [args.seed + ci * 100 + k for k in range(args.k)]

        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1].clone()
            del roll
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        prefix = fd.build_prefix(cache, n_layers, pad_multiple=args.pad)
        del out

        ref = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # A: release path (warm one call first for fairness)
            lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask, seed=seeds[0])
            for s in seeds:
                a, ms = timed(lambda: lib.denoise_with_cache(
                    model, cache, rope_deltas, offset, prefix_mask, seed=s))
                ref[s] = to_xy(model, inputs, a)
                res["release"]["ms"].append(ms)
            # B: fast path, uncompiled
            for s in seeds:
                a, ms = timed(lambda: fd.fast_denoise(
                    model, prefix, rope_deltas, offset, seed=s))
                res["fast"]["ms"].append(ms)
                res["fast"]["dxy"].append(float(np.abs(to_xy(model, inputs, a) - ref[s]).max()))
            # C: CUDA-graph fast path (capture once per padded shape, timed apart)
            s_pad = prefix.get_seq_length()
            if s_pad not in graphs:
                g, ms = timed(lambda: fd.GraphedDenoiser(model, s_pad))
                graphs[s_pad] = g
                compile_times.append(ms)
            graphs[s_pad].load_clip(prefix, rope_deltas, offset)
            for s in seeds:
                a, ms = timed(lambda: fd.graphed_denoise(model, graphs[s_pad], seed=s))
                res["compiled"]["ms"].append(ms)
                res["compiled"]["dxy"].append(
                    float(np.abs(to_xy(model, inputs, a) - ref[s]).max()))
        print(f"[{ci + 1}/{len(clips)}] {clip_id} prefill={prefill} pad={prefix.get_seq_length()} "
              f"({time.time() - t0:.0f}s)", flush=True)

    lines = [f"expert denoise fast-path bench ({'slim ' + args.slim_ckpt if args.slim_ckpt else 'baseline'}, "
             f"{len(clips)} clips x K={args.k}, pad={args.pad})"]
    for p in ("release", "fast", "compiled"):
        ms = float(np.median(res[p]["ms"]))
        dxy = max(res[p]["dxy"]) if res[p]["dxy"] else 0.0
        lines.append(f"{p:9s} median {ms:7.1f} ms/denoise   max|dxy| vs release {dxy:.4f} m")
    lines.append(f"compile/capture first-call: {[f'{t:.0f}' for t in compile_times]} ms")
    txt = "\n".join(lines)
    print(txt, flush=True)
    (out_dir / "summary.txt").write_text(txt + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "slim_ckpt": args.slim_ckpt, "clips": clips, "k": args.k, "pad": args.pad,
        "per_call_ms": {p: res[p]["ms"] for p in res},
        "dxy": {p: res[p]["dxy"] for p in res}, "compile_ms": compile_times,
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "functional cache-immutable expert denoise: equivalence + latency",
        "slim_ckpt": args.slim_ckpt, "num_clips": len(clips), "k": args.k,
        "pad_multiple": args.pad, "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))


if __name__ == "__main__":
    main()
