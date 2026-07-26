"""Phase P1b: dual-objective Taylor importance for Q heads, MLP channels, KV groups.

Per clip, one teacher-forced VLM forward serves both objectives:
  1. CoC NLL          -> backward (retain_graph)  -> "output" ranking
  2. trajectory FM MSE -> expert backward onto detached cache leaves, then one
     VLM backward seeded with the accumulated cache grads -> "cache" ranking

The two-stage trajectory backward costs one VLM backward instead of fm_steps of
them, which is what makes measuring both objectives affordable.

Usage:
  bash run_retry.sh 30 experiments/head_analysis/run_importance.py --gpu 0 --num-clips 50
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
import prune_lib as pl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def flat(pairs):
    return [t for kv in pairs for t in kv]


def process_clip(model, processor, data, args, seed, acc):
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    eos_pos = roll["eos_pos"]
    coc_start, coc_end = prompt_len, eos_pos + 1
    seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
    del roll

    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)

    vlm_gates = pl.UnitGates(
        model.vlm.model.language_model.layers,
        model.vlm.config.text_config.num_attention_heads,
        model.vlm.config.text_config.head_dim,
        model.vlm.config.text_config.intermediate_size,
        "cuda", torch.float32,
    )
    expert_gates = pl.UnitGates(
        model.expert.layers,
        model.expert.config.num_attention_heads,
        model.expert.config.head_dim,
        model.expert.config.intermediate_size,
        "cuda", torch.float32,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=True
        )
    n_vlm = len(model.vlm.model.language_model.layers)
    cache_t = pl.retain_cache_grads(cache, n_vlm)
    prefill = cache.get_seq_length()

    # ---- objective 1: CoC NLL ----
    nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
    nll.backward(retain_graph=True)
    acc["coc"]["vlm_q"] += vlm_gates.q_scores()
    acc["coc"]["vlm_mlp"] += vlm_gates.mlp_scores()
    k_s, v_s = pl.kv_group_scores(cache_t)
    acc["coc"]["kv_k"] += k_s
    acc["coc"]["kv_v"] += v_s
    vlm_gates.zero_grads()
    for k, v in cache_t:
        k.grad = None
        v.grad = None
    del hidden, nll

    # ---- objective 2: trajectory flow-matching ----
    fm_loss, grads, leaves = pl.expert_fm_grads(
        model, cache, rope_deltas, x1, args.fm_steps, seed, prefill
    )
    acc["traj"]["exp_q"] += expert_gates.q_scores()
    acc["traj"]["exp_mlp"] += expert_gates.mlp_scores()

    ts, gs = [], []
    for (k, v), (gk, gv) in zip(cache_t, grads):
        for t_, g_ in ((k, gk), (v, gv)):
            if g_ is not None:
                ts.append(t_)
                gs.append(g_.to(t_.dtype))
    torch.autograd.backward(ts, gs)
    acc["traj"]["vlm_q"] += vlm_gates.q_scores()
    acc["traj"]["vlm_mlp"] += vlm_gates.mlp_scores()
    k_s, v_s = pl.kv_group_scores(cache_t)
    acc["traj"]["kv_k"] += k_s
    acc["traj"]["kv_v"] += v_s

    peak = torch.cuda.max_memory_allocated() / 1024**3
    vlm_gates.remove()
    expert_gates.remove()
    del cache, cache_t, leaves, grads, ts, gs
    return {"coc_len": coc_end - coc_start, "prompt_len": prompt_len,
            "fm_loss": fm_loss, "peak_gb": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=50)
    ap.add_argument("--exp-id", type=str, default="importance_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--checkpoint", action="store_true",
                    help="see note below; incompatible with use_cache, off by default")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    calib = split["calib"][: args.num_clips]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    # all weights are frozen and the gates sit downstream of k/v, so without this
    # the cache tensors of the early layers carry no autograd graph
    model.vlm.enable_input_require_grads()
    # Gradient checkpointing is deliberately off. transformers forces use_cache=False
    # whenever checkpointing is active in training mode, but the KV cache is the very
    # thing this pass needs, and without it generation recomputes the RoPE index every
    # decode step and crashes. Measured peak without checkpointing is ~40.5 GB, which
    # fits the 48 GB cards.
    if args.checkpoint:
        model.vlm.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.vlm.model.language_model.train()

    tc = model.vlm.config.text_config
    ec = model.expert.config
    shapes = {
        "vlm_q": (tc.num_hidden_layers, tc.num_attention_heads),
        "vlm_mlp": (tc.num_hidden_layers, tc.intermediate_size),
        "kv_k": (tc.num_hidden_layers, tc.num_key_value_heads),
        "kv_v": (tc.num_hidden_layers, tc.num_key_value_heads),
        "exp_q": (ec.num_hidden_layers, ec.num_attention_heads),
        "exp_mlp": (ec.num_hidden_layers, ec.intermediate_size),
    }
    acc = {obj: {k: np.zeros(s) for k, s in shapes.items()} for obj in ("coc", "traj")}

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "purpose": "dual-objective Taylor importance for Q head / MLP channel / KV group",
        "objectives": {"coc": "CoC NLL", "traj": "flow-matching MSE vs GT trajectory"},
        "num_clips": len(calib), "clip_ids": calib, "seed": args.seed,
        "fm_steps": args.fm_steps, "gradient_checkpointing": args.checkpoint,
        "gpu": torch.cuda.get_device_name(device), "shapes": {k: list(v) for k, v in shapes.items()},
    }, indent=2))

    records = []
    for ci, clip_id in enumerate(calib):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        torch.cuda.reset_peak_memory_stats()
        rec = process_clip(model, processor, data, args, args.seed + ci, acc)
        rec["clip_id"] = clip_id
        records.append(rec)
        print(f"[{ci + 1}/{len(calib)}] {clip_id} coc={rec['coc_len']} "
              f"fm={rec['fm_loss']:.4f} peak={rec['peak_gb']:.1f}GB "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 10 == 0 or ci + 1 == len(calib):
            save(out_dir, acc, records, ci + 1)
    save(out_dir, acc, records, len(records))
    print("saved ->", out_dir, flush=True)


def save(out_dir, acc, records, n):
    arrays = {f"{obj}_{k}": v / max(n, 1) for obj, d in acc.items() for k, v in d.items()}
    np.savez(out_dir / "importance.npz", **arrays)
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": n, "per_clip": records}, indent=2))


if __name__ == "__main__":
    main()
