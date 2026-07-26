"""Stage-level latency / FLOPs / memory profile of Alpamayo-1.5 inference.

Replaces the analytic estimates (params x tokens) with measurements for the four
stages of one inference: ViT -> VLM prefill -> CoC decode -> expert denoising.

Timing uses CUDA events registered as module hooks, so nested python/dispatch
overhead is attributed to the enclosing stage rather than lost. `visual` is a
sibling of `language_model` inside Qwen3VLModel.forward, so the two never nest.

Usage:
  bash run_retry.sh 60 experiments/head_analysis/profile_stages.py --num-clips 10
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


class ModuleTimer:
    """Per-call CUDA-event timing for a set of named modules."""

    def __init__(self):
        self.records = []  # (name, start_event, end_event)
        self._pending = {}
        self._handles = []

    def attach(self, module, name):
        def pre(mod, args):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending[name] = ev

        def post(mod, args, output):
            start = self._pending.pop(name)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records.append((name, start, end))

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def collect(self):
        """Synchronize and return {name: [ms, ...]} in call order."""
        torch.cuda.synchronize()
        out = defaultdict(list)
        for name, start, end in self.records:
            out[name].append(start.elapsed_time(end))
        self.records = []
        return dict(out)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def cache_bytes(cache):
    total = 0
    for i in range(len(cache)):
        k, v = lib.cache_layer_kv(cache, i)
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def flop_counts_to_json(counts):
    """FlopCounterMode keys are torch ops; make them json-serializable."""
    return {mod: {str(op): int(f) for op, f in ops.items()} for mod, ops in counts.items()}


def profile_clip(model, processor, data, timer, seed, max_gen):
    """One full inference with per-stage timing. Returns a metrics dict."""
    torch.cuda.reset_peak_memory_stats()
    inputs = lib.build_inputs(model, processor, data, "cuda")
    spans = lib.compute_spans(model, inputs["input_ids"])
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
    torch.cuda.synchronize()
    rollout_wall = (time.perf_counter() - t0) * 1e3
    rollout_peak = torch.cuda.max_memory_allocated()
    rollout_times = timer.collect()

    eos_pos = roll["eos_pos"]
    coc_len = eos_pos + 1 - prompt_len
    prefill = roll["past_key_values"].get_seq_length()
    kv_bytes = cache_bytes(roll["past_key_values"])

    offset = torch.tensor([eos_pos + 1], device="cuda")
    prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lib.denoise_with_cache(
            model, roll["past_key_values"], roll["rope_deltas"], offset, prefix_mask, seed=seed
        )
    torch.cuda.synchronize()
    denoise_wall = (time.perf_counter() - t0) * 1e3
    denoise_peak = torch.cuda.max_memory_allocated()
    denoise_times = timer.collect()

    # language_model call 0 is the prefill; calls 1..n are the CoC decode steps
    lm = rollout_times.get("language_model", [])
    vit = rollout_times.get("visual", [])
    expert = denoise_times.get("expert", [])
    m = {
        "prompt_len": prompt_len,
        "vision_tokens": int(spans["vision"].sum()),
        "coc_len": coc_len,
        "prefill_len": prefill,
        "kv_cache_mb": kv_bytes / 1024**2,
        "vit_ms": float(np.sum(vit)),
        "vit_calls": len(vit),
        "prefill_ms": float(lm[0]) if lm else None,
        "decode_ms": float(np.sum(lm[1:])) if len(lm) > 1 else 0.0,
        "decode_steps": max(len(lm) - 1, 0),
        "lm_head_ms": float(np.sum(rollout_times.get("lm_head", []))),
        "expert_ms": float(np.sum(expert)),
        "expert_steps": len(expert),
        "rollout_wall_ms": rollout_wall,
        "denoise_wall_ms": denoise_wall,
        "total_wall_ms": rollout_wall + denoise_wall,
        "rollout_peak_gb": rollout_peak / 1024**3,
        "denoise_peak_gb": denoise_peak / 1024**3,
    }
    # ViT runs inside the prefill call of Qwen3VLModel but as a sibling of
    # language_model, so it is not contained in prefill_ms; the remainder is
    # embedding / masked_scatter / rope / sampling / python overhead.
    m["rollout_other_ms"] = rollout_wall - m["vit_ms"] - (m["prefill_ms"] or 0) - m["decode_ms"]
    m["denoise_other_ms"] = denoise_wall - m["expert_ms"]
    del roll
    torch.cuda.empty_cache()
    return m


def flops_pass(model, processor, data, seed, max_gen, depth):
    """Measure FLOPs for the rollout and the denoising separately.

    Runs under eager attention: torch's sdpa_flop_count asserts Q and KV have the
    same head count, which GQA (32 Q / 8 KV) violates. Eager materializes the same
    matmuls, so the FLOP totals are the ones we want either way.
    """
    lib.set_vlm_attn_impl(model, "eager")
    lib.set_expert_attn_impl(model, "eager")
    inputs = lib.build_inputs(model, processor, data, "cuda")
    out = {}

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    counter = FlopCounterMode(mods=model, depth=depth, display=False)
    with counter, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
    out["rollout"] = {
        "total": counter.get_total_flops(),
        "by_module": flop_counts_to_json(counter.get_flop_counts()),
    }

    eos_pos = roll["eos_pos"]
    prefill = roll["past_key_values"].get_seq_length()
    offset = torch.tensor([eos_pos + 1], device="cuda")
    prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
    counter = FlopCounterMode(mods=model, depth=depth, display=False)
    with counter, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lib.denoise_with_cache(
            model, roll["past_key_values"], roll["rope_deltas"], offset, prefix_mask, seed=seed
        )
    out["denoise"] = {
        "total": counter.get_total_flops(),
        "by_module": flop_counts_to_json(counter.get_flop_counts()),
    }
    out["coc_len"] = eos_pos + 1 - inputs["input_ids"].shape[1]
    out["attn_impl"] = "eager"
    del roll
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--exp-id", type=str, default="profile_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=24.0)
    ap.add_argument("--attn-impl", type=str, default="sdpa")
    ap.add_argument("--flop-depth", type=int, default=5)
    ap.add_argument("--no-flops", action="store_true")
    ap.add_argument("--gpu", type=int, default=None, help="pin to this cuda index")
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--slim-ckpt", type=str, default=None,
                    help="load a slim checkpoint dir (slim_lib.load_slim) instead of the HF model")
    ap.add_argument("--identity-gqa", action="store_true",
                    help="unpruned model on the slim attention path (gather-overhead control)")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_df = pd.read_parquet(REPO / "notebooks" / "clip_ids.parquet")
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(clip_df))
    n_total = args.num_clips + args.warmup
    sel = order[args.clip_offset : args.clip_offset + n_total]
    clip_ids = clip_df.iloc[sel]["clip_id"].tolist()

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    print("Loading model...", flush=True)
    if args.slim_ckpt:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.slim_ckpt, device="cuda")
    else:
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
        if args.identity_gqa:
            import slim_lib as sl
            sl.bind_identity(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    # production settings: sdpa everywhere (eager would inflate attention cost)
    lib.set_vlm_attn_impl(model, args.attn_impl)
    lib.set_expert_attn_impl(model, args.attn_impl)

    gpu_name = torch.cuda.get_device_name(device)
    param_counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "vlm_language_model": sum(p.numel() for p in model.vlm.model.language_model.parameters()),
        "vlm_visual": sum(p.numel() for p in model.vlm.model.visual.parameters()),
        "vlm_lm_head": sum(p.numel() for p in model.vlm.lm_head.parameters()),
        "expert": sum(p.numel() for p in model.expert.parameters()),
    }
    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "purpose": "measured per-stage latency / FLOPs / memory for one inference",
        "num_clips": args.num_clips,
        "warmup": args.warmup,
        "clip_ids": clip_ids,
        "seed": args.seed,
        "max_generation_length": args.max_gen,
        "attn_implementation": args.attn_impl,
        "dtype": "bfloat16",
        "gpu": gpu_name,
        "gpu_index": device.index,
        "clip_offset": args.clip_offset,
        "torch": torch.__version__,
        "param_counts": param_counts,
        "slim_ckpt": args.slim_ckpt,
        "identity_gqa": args.identity_gqa,
    }, indent=2))

    timer = ModuleTimer()
    timer.attach(model.vlm.model.visual, "visual")
    timer.attach(model.vlm.model.language_model, "language_model")
    timer.attach(model.vlm.lm_head, "lm_head")
    timer.attach(model.expert, "expert")

    records = []
    flops = None
    for ci, clip_id in enumerate(clip_ids):
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        m = profile_clip(model, processor, data, timer, args.seed + ci, args.max_gen)
        m["clip_id"] = clip_id
        m["warmup"] = ci < args.warmup
        records.append(m)
        tag = "warmup" if m["warmup"] else f"{ci - args.warmup + 1}/{args.num_clips}"
        print(
            f"[{tag}] {clip_id} vit={m['vit_ms']:.0f} prefill={m['prefill_ms']:.0f} "
            f"decode={m['decode_ms']:.0f}({m['decode_steps']}) expert={m['expert_ms']:.0f} "
            f"other={m['rollout_other_ms'] + m['denoise_other_ms']:.0f} "
            f"total={m['total_wall_ms']:.0f} ms",
            flush=True,
        )
        if not args.no_flops and flops is None and ci == args.warmup:
            timer.remove()
            print("measuring FLOPs (one clip, slow)...", flush=True)
            flops = flops_pass(model, processor, data, args.seed + ci, args.max_gen, args.flop_depth)
            print(f"  rollout {flops['rollout']['total'] / 1e12:.2f} TFLOP, "
                  f"denoise {flops['denoise']['total'] / 1e12:.2f} TFLOP", flush=True)
            lib.set_vlm_attn_impl(model, args.attn_impl)
            lib.set_expert_attn_impl(model, args.attn_impl)
            timer = ModuleTimer()
            timer.attach(model.vlm.model.visual, "visual")
            timer.attach(model.vlm.model.language_model, "language_model")
            timer.attach(model.vlm.lm_head, "lm_head")
            timer.attach(model.expert, "expert")

    timer.remove()
    (out_dir / "metrics.json").write_text(json.dumps(
        {"per_clip": records, "flops": flops}, indent=2
    ))
    write_summary(out_dir, records, flops, param_counts, gpu_name, args)
    print(f"saved -> {out_dir}", flush=True)


STAGES = [
    ("vit_ms", "ViT"),
    ("prefill_ms", "VLM prefill"),
    ("decode_ms", "CoC decode"),
    ("expert_ms", "Expert denoise"),
    ("rollout_other_ms", "rollout overhead"),
    ("denoise_other_ms", "denoise overhead"),
]


def write_summary(out_dir, records, flops, param_counts, gpu_name, args):
    live = [r for r in records if not r["warmup"]]
    med = {k: float(np.median([r[k] for r in live])) for k, _ in STAGES}
    total = float(np.median([r["total_wall_ms"] for r in live]))
    lines = [
        "Alpamayo-1.5-10B stage profile",
        f"  gpu={gpu_name}  attn={args.attn_impl}  dtype=bf16  n={len(live)} clips (+{args.warmup} warmup)",
        f"  params total {param_counts['total'] / 1e9:.3f}B",
        "",
        f"{'stage':<20}{'median ms':>12}{'% of total':>12}",
    ]
    for key, label in STAGES:
        lines.append(f"{label:<20}{med[key]:>12.1f}{med[key] / total * 100:>11.1f}%")
    lines.append(f"{'TOTAL (wall)':<20}{total:>12.1f}{100.0:>11.1f}%")

    lines += ["", "sequence / memory (median):"]
    for k, label in [
        ("prompt_len", "prompt tokens"), ("vision_tokens", "vision tokens"),
        ("coc_len", "CoC tokens"), ("decode_steps", "decode steps"),
        ("expert_steps", "expert steps"), ("kv_cache_mb", "KV cache MB"),
        ("rollout_peak_gb", "rollout peak GB"), ("denoise_peak_gb", "denoise peak GB"),
    ]:
        lines.append(f"  {label:<20}{np.median([r[k] for r in live]):>10.1f}")

    if flops:
        lines += ["", "FLOPs (one clip):"]
        rt, dt = flops["rollout"]["total"], flops["denoise"]["total"]
        lines.append(f"  rollout (ViT+prefill+decode) {rt / 1e12:>8.2f} TFLOP")
        lines.append(f"  denoise                      {dt / 1e12:>8.2f} TFLOP")
        lines.append(f"  total                        {(rt + dt) / 1e12:>8.2f} TFLOP")
        for scope in ("rollout", "denoise"):
            lines.append(f"  -- {scope} by module --")
            by_mod = flops[scope]["by_module"]
            tot = {m: sum(ops.values()) for m, ops in by_mod.items() if m != "Global"}
            for m, f in sorted(tot.items(), key=lambda x: -x[1])[:8]:
                lines.append(f"    {m:<48}{f / 1e12:>8.2f} TFLOP")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
