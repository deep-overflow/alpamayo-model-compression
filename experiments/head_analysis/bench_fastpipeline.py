"""End-to-end pipeline benchmark: stock vs graph-optimized, per clip, same seeds.

Stock path:  run_rollout (generate) -> denoise_with_cache        (release semantics)
Fast path:   eager ViT+prefill -> graphed decode loop -> one EOS step (completes the
             cache like the release path) -> graphed expert denoise on the static KV.

Per clip both paths run and are stage-timed with CUDA events. Decode stages are also
reported per token; end-to-end additionally normalized to REF_STEPS CoC tokens so the
slim-integrated model's degenerate long rollouts do not distort the comparison.

Usage (PYTORCH_CUDA_ALLOC_CONF= required for graph capture):
  PYTORCH_CUDA_ALLOC_CONF= bash experiments/head_analysis/run_retry.sh 20 \
      experiments/head_analysis/bench_fastpipeline.py --gpu 0 --reserve-gb 8 \
      --exp-id fastpipe_base [--slim-ckpt outputs/slim_integrated_mag]
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
import fast_decode as fdec  # noqa: E402
import fast_denoise as fd  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from profile_stages import ModuleTimer  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")
REF_STEPS = 16  # baseline median CoC length (profile convention)


def timed(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def prefix_from_static(cache, true_len, n_layers, pad_multiple=128):
    """Padded StaticPrefixCache from the decoder's static KV buffers [0, true_len)."""
    s_pad = -(-true_len // pad_multiple) * pad_multiple
    keys, values = [], []
    for i in range(n_layers):
        k, v = lib.cache_layer_kv(cache, i)
        k = k[:, :, :true_len]
        v = v[:, :, :true_len]
        pad = s_pad - true_len
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        keys.append(k.contiguous())
        values.append(v.contiguous())
    return fd.StaticPrefixCache(keys, values, s_pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim-ckpt", type=str, default=None)
    ap.add_argument("--num-clips", type=int, default=12)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--cap", type=int, default=3456)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
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
    n_layers = model.vlm.config.text_config.num_hidden_layers

    dec, cap_ms = timed(lambda: fdec.GraphedDecoder(model, args.cap))
    print(f"decode capture {cap_ms:.0f} ms", flush=True)
    denoisers = {}  # s_pad -> GraphedDenoiser

    # same clip selection as profile_stages (seed permutation + offset, no warmup here)
    import pandas as pd
    clip_df = pd.read_parquet(REPO / "notebooks" / "clip_ids.parquet")
    order = np.random.RandomState(args.seed).permutation(len(clip_df))
    sel = order[args.clip_offset : args.clip_offset + args.num_clips + 1]
    clips = clip_df.iloc[sel]["clip_id"].tolist()  # first clip doubles as warmup

    rows = []
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        r = {"clip_id": clip_id, "warmup": ci == 0, "prompt_len": prompt_len}

        # ---- stock path (ModuleTimer splits prefill = language_model call 0) ----
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        mt = ModuleTimer()
        mt.attach(model.vlm.model.visual, "visual")
        mt.attach(model.vlm.model.language_model, "lm")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll, ms = timed(lambda: lib.run_rollout(
                model, inputs, max_generation_length=args.max_gen))
            times = mt.collect()
            mt.remove()
            r["stock_rollout_ms"] = ms
            r["stock_steps"] = roll["eos_pos"] + 1 - prompt_len
            r["stock_prefill_ms"] = sum(times.get("visual", [])) + times["lm"][0]
            r["stock_decode_ms"] = sum(times["lm"][1:])
            cache = roll["past_key_values"]
            offset = torch.tensor([cache.get_seq_length()], device="cuda")
            pm = torch.ones(1, cache.get_seq_length(), device="cuda", dtype=torch.long)
            _, ms = timed(lambda: lib.denoise_with_cache(
                model, cache, roll["rope_deltas"], offset, pm, seed=args.seed + ci))
            r["stock_denoise_ms"] = ms
        del roll, cache

        # ---- fast path ----
        with torch.no_grad():
            g, ms = timed(lambda: fdec.rollout_graphed(
                model, dec, inputs, max_generation_length=args.max_gen,
                seed=args.seed + ci))
            r["fast_rollout_ms"] = ms
            r["fast_steps"] = g["n_steps"]
            r["fast_prefill_ms"] = g["prefill_ms"]
            r["fast_decode_ms"] = g["loop_ms"]
            # complete the cache with the EOS token's KV (release-protocol parity),
            # then denoise off the static buffers
            eos_pos_abs = prompt_len + g["n_steps"] - 1
            def _denoise():
                dec.step(g["sequences"][:, -1:], eos_pos_abs)
                true_len = prompt_len + g["n_steps"]
                prefix = prefix_from_static(dec.cache, true_len, n_layers)
                s_pad = prefix.get_seq_length()
                if s_pad not in denoisers:
                    denoisers[s_pad] = fd.GraphedDenoiser(model, s_pad)
                    r["denoise_capture"] = True
                gd = denoisers[s_pad]
                gd.load_clip(prefix, dec.delta,
                             torch.tensor([true_len], device="cuda"))
                return fd.graphed_denoise(model, gd, seed=args.seed + ci)
            _, ms = timed(_denoise)
            r["fast_denoise_ms"] = ms
        rows.append(r)
        print(f"[{ci + 1}/{len(clips)}]{' W' if r['warmup'] else ''} {clip_id} "
              f"stock {r['stock_rollout_ms'] + r['stock_denoise_ms']:7.1f} "
              f"({r['stock_steps']} tok)  fast {r['fast_rollout_ms'] + r['fast_denoise_ms']:7.1f} "
              f"({r['fast_steps']} tok)  ({time.time() - t0:.0f}s)", flush=True)

    live = [r for r in rows if not r["warmup"] and not r.get("denoise_capture")]
    med = lambda k: float(np.median([r[k] for r in live]))  # noqa: E731
    stock_tok = float(np.median([r["stock_decode_ms"] / r["stock_steps"] for r in live]))
    fast_tok = float(np.median(
        [r["fast_decode_ms"] / max(r["fast_steps"] - 1, 1) for r in live]))
    # normalized end-to-end: (ViT+prefill) + decode/tok * REF_STEPS + denoise
    stock_norm = med("stock_prefill_ms") + stock_tok * REF_STEPS + med("stock_denoise_ms")
    fast_norm = med("fast_prefill_ms") + fast_tok * REF_STEPS + med("fast_denoise_ms")
    lines = [
        f"pipeline bench ({'slim ' + args.slim_ckpt if args.slim_ckpt else 'baseline'}, "
        f"n={len(live)} live clips, cap={args.cap})",
        f"vit+prefill   stock {med('stock_prefill_ms'):6.1f}   fast {med('fast_prefill_ms'):6.1f} ms",
        f"decode/tok    stock {stock_tok:6.1f}   fast {fast_tok:6.1f} ms",
        f"denoise       stock {med('stock_denoise_ms'):6.1f}   fast {med('fast_denoise_ms'):6.1f} ms",
        f"end-to-end (norm {REF_STEPS} tok)  stock {stock_norm:7.1f}   fast {fast_norm:7.1f} ms "
        f"->  {stock_norm / fast_norm:.2f}x",
    ]
    txt = "\n".join(lines)
    print(txt, flush=True)
    (out_dir / "summary.txt").write_text(txt + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "slim_ckpt": args.slim_ckpt, "cap": args.cap, "ref_steps": REF_STEPS,
        "decode_capture_ms": cap_ms, "per_clip": rows,
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "stock vs graph-optimized full pipeline, paired per clip",
        "slim_ckpt": args.slim_ckpt, "num_clips": len(clips), "clip_offset": args.clip_offset,
        "cap": args.cap, "seed": args.seed, "gpu": torch.cuda.get_device_name(device),
    }, indent=2))


if __name__ == "__main__":
    main()
