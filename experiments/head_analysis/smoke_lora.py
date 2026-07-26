"""Design-B LoRA gate: one grad VLM forward -> attached cache -> FM loss -> single
backward must reach BOTH expert LoRA and VLM LoRA, and fit in memory.

Verifies the three risks before writing the full trainer: (1) peft wraps the surgically
resized slim modules, (2) the trajectory gradient actually reaches VLM LoRA via the cache
(not just expert LoRA), (3) peak memory < card. Also checks merge_and_unload keeps slim
shapes.

Usage:
  PYTORCH_CUDA_ALLOC_CONF= bash experiments/head_analysis/run_retry.sh 10 \
      experiments/head_analysis/smoke_lora.py --gpu 4 --ckpt outputs/slim_integrated_mag
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import lora_lib as ll  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.token_utils import to_special_token  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/slim_integrated_mag")
    ap.add_argument("--reserve-gb", type=float, default=42.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    peft_model, base, meta = ll.load_slim_lora(REPO / args.ckpt, device="cuda")
    lib.set_vlm_attn_impl(base, "sdpa")
    lib.set_expert_attn_impl(base, "sdpa")
    trainable, total = ll.param_summary(peft_model)
    print(f"config={meta['config']}  trainable={trainable:,}  total={total:,}  "
          f"({trainable / total * 100:.3f}%)", flush=True)
    assert trainable > 0, "no trainable LoRA params"

    processor = helper.get_processor(base.tokenizer)
    eos_id = base.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clip_id = split["train"][0]

    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    inputs = lib.build_inputs(base, processor, data, "cuda")
    x1 = lib.gt_actions(base, data, "cuda").to(torch.float32)  # (1, 64, 2)

    # teacher-forced sequence (prompt + CoC + traj_future_start) via no-grad rollout
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(base, inputs, max_generation_length=256)
    seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1].clone()
    del roll

    torch.cuda.reset_peak_memory_stats()
    # Design B: grad VLM forward (attached cache) -> FM loss -> single backward
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = base.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        loss = ph_fm(base, cache, rope_deltas, seq_tf, eos_id, x1)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"FM loss {loss.item():.5f}  peak {peak:.1f} GB", flush=True)

    # gradient must have reached BOTH towers' LoRA
    def lora_grad(substr):
        g = [(n, p) for n, p in peft_model.named_parameters()
             if p.requires_grad and "lora_" in n and substr in n]
        with_grad = [n for n, p in g if p.grad is not None and p.grad.abs().sum() > 0]
        return len(g), len(with_grad)

    ve, vg = lora_grad("language_model.layers")
    ee, eg = lora_grad("expert.layers")
    print(f"VLM LoRA: {vg}/{ve} params with nonzero grad", flush=True)
    print(f"expert LoRA: {eg}/{ee} params with nonzero grad", flush=True)
    assert vg > 0, "VLM LoRA received NO gradient -- Design B cache path broken"
    assert eg > 0, "expert LoRA received no gradient"

    # merge keeps slim shapes
    shapes_before = {n: tuple(p.shape) for n, p in base.named_parameters()
                     if n.endswith(".weight") and "layers." in n and "lora" not in n}
    peft_model.merge_and_unload()
    import slim_lib as sl
    sl.check_slim(base)
    n_ok = sum(1 for n, p in base.named_parameters()
               if n in shapes_before and tuple(p.shape) == shapes_before[n])
    print(f"merge: {n_ok}/{len(shapes_before)} base weights kept slim shape", flush=True)
    print("SMOKE PASS", flush=True)


def ph_fm(model, cache, rope_deltas, seq, eos_id, x1):
    return ll.ph21.flow_matching_loss(model, cache, rope_deltas, seq, eos_id, x1)


if __name__ == "__main__":
    main()
