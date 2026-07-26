"""Pre-generate teacher CoC token sequences with the FULL model, for LoRA training.

The recovery FM loss builds its denoise cache from a VLM forward over [prompt + CoC +
traj_future_start]. To match the evaluation protocol (which denoises on the baseline
model's CoC context) and to avoid a slow per-step rollout (slim_integrated's degenerate
CoC runs to 256 tokens), the reference CoC is generated ONCE with the full model and
reused for every config, every step.

Stores per clip: the generated CoC token ids (sequences[prompt_len : eos_pos+1], i.e.
including the traj_future_start token). The prompt is rebuilt deterministically from the
clip by build_inputs at train time, so only the continuation needs saving.

Shardable across GPUs by clip offset.

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/make_teacher_refs.py \
      --gpu 0 --split train --offset 0 --num 300 --exp-id teacher_refs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--num", type=int, default=10_000)
    ap.add_argument("--exp-id", type=str, default="teacher_refs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=26.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split[args.split][args.offset : args.offset + args.num]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}  {args.split}[{args.offset}:{args.offset + len(clips)}]", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    shard_path = out_dir / f"{args.split}_{args.offset:04d}.json"
    refs = json.loads(shard_path.read_text()) if shard_path.exists() else {}
    for ci, clip_id in enumerate(clips):
        if clip_id in refs:
            continue
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_ids = roll["sequences"][0, prompt_len : roll["eos_pos"] + 1].tolist()
        refs[clip_id] = {"prompt_len": prompt_len, "coc_ids": coc_ids}
        del roll
        if (ci + 1) % 10 == 0 or ci + 1 == len(clips):
            shard_path.write_text(json.dumps(refs))
        print(f"[{ci + 1}/{len(clips)}] {clip_id} coc_len={len(coc_ids)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    shard_path.write_text(json.dumps(refs))
    print(f"saved {len(refs)} refs -> {shard_path}", flush=True)


if __name__ == "__main__":
    main()
