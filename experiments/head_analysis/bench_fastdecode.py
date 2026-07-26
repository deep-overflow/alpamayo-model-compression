"""Equivalence + latency benchmark of the graphed CoC decode.

Per clip: a stock rollout fixes a token sequence, then (a) the batched TF forward gives
reference logits, (b) the manual loop replays the same tokens teacher-forced, eager and
graphed -- logit deviation checks correctness, loop time / n_tokens gives a clean
decode ms/token under a controlled step count. Full sampled rollouts (stock generate vs
graphed loop) are timed for the end-to-end view.

Usage:
  PYTORCH_CUDA_ALLOC_CONF= bash experiments/head_analysis/run_retry.sh 10 \
      experiments/head_analysis/bench_fastdecode.py --gpu 4 --exp-id fastdecode_base
  ... --slim-ckpt outputs/slim_integrated_mag --exp-id fastdecode_slim_int
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
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


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
    ap.add_argument("--cap", type=int, default=3456)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
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

    dec, cap_ms = timed(lambda: fdec.GraphedDecoder(model, args.cap))
    print(f"capture {cap_ms:.0f} ms (cap={args.cap})", flush=True)

    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split["val"][: args.num_clips]

    res = {"stock_wall": [], "graph_wall": [], "eager_tok": [], "graph_tok": [],
           "dlogit": [], "dnll": [], "stock_steps": [], "graph_steps": []}
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]

        # stock sampled rollout (reference sequence + wall time)
        torch.manual_seed(args.seed + ci)
        torch.cuda.manual_seed_all(args.seed + ci)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll, ms = timed(lambda: lib.run_rollout(
                model, inputs, max_generation_length=args.max_gen))
        res["stock_wall"].append(ms)
        res["stock_steps"].append(roll["eos_pos"] + 1 - prompt_len)
        seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1].clone()
        del roll

        # batched TF forward -> reference logits over the CoC span
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=False,
            )
            ref_logits = model.vlm.lm_head(
                out.last_hidden_state[:, prompt_len - 1 : -1]).float()
        del out
        tgt = seq_tf[0, prompt_len:]
        ref_nll = torch.nn.functional.cross_entropy(ref_logits[0], tgt).item()

        # TF replay through the manual loop: eager then graphed (loop-only timing;
        # the decode loop runs n_coc - 1 steps, the first CoC logit comes from prefill)
        n_coc = seq_tf.shape[1] - prompt_len
        with torch.no_grad():
            lg_e, ms_e = fdec.tf_replay_logits(model, dec, inputs, seq_tf, graphed=False)
            lg_g, ms_g = fdec.tf_replay_logits(model, dec, inputs, seq_tf, graphed=True)
        res["eager_tok"].append(ms_e / max(n_coc - 1, 1))
        res["graph_tok"].append(ms_g / max(n_coc - 1, 1))
        res["dlogit"].append(float((lg_g - ref_logits).abs().max()))
        nll_g = torch.nn.functional.cross_entropy(lg_g[0], tgt).item()
        res["dnll"].append(nll_g - ref_nll)

        # sampled rollout through the graphed loop (end-to-end wall)
        with torch.no_grad():
            g, ms = timed(lambda: fdec.rollout_graphed(
                model, dec, inputs, max_generation_length=args.max_gen,
                seed=args.seed + ci))
        res["graph_wall"].append(ms)
        res["graph_steps"].append(g["n_steps"])
        print(f"[{ci + 1}/{len(clips)}] {clip_id} coc={n_coc} "
              f"eager {ms_e / n_coc:.1f} graph {ms_g / n_coc:.1f} ms/tok "
              f"max|dlogit| {res['dlogit'][-1]:.3f} dNLL {res['dnll'][-1]:+.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    med = lambda k: float(np.median(res[k]))  # noqa: E731
    lines = [
        f"graphed decode bench ({'slim ' + args.slim_ckpt if args.slim_ckpt else 'baseline'}, "
        f"{len(clips)} clips, cap={args.cap}, capture {cap_ms:.0f} ms)",
        f"TF replay  eager {med('eager_tok'):6.2f} ms/tok   graphed {med('graph_tok'):6.2f} ms/tok",
        f"equivalence  max|dlogit| {max(res['dlogit']):.3f}   max|dNLL| "
        f"{max(abs(d) for d in res['dnll']):.5f}",
        f"rollout wall  stock {med('stock_wall'):7.1f} ms ({res['stock_steps']})   "
        f"graphed {med('graph_wall'):7.1f} ms ({res['graph_steps']})",
    ]
    txt = "\n".join(lines)
    print(txt, flush=True)
    (out_dir / "summary.txt").write_text(txt + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "slim_ckpt": args.slim_ckpt, "clips": clips, "cap": args.cap,
        "capture_ms": cap_ms, **res, "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "purpose": "graphed static-KV decode: TF-replay equivalence + per-token latency",
        "slim_ckpt": args.slim_ckpt, "num_clips": len(clips), "cap": args.cap,
        "seed": args.seed, "gpu": torch.cuda.get_device_name(device),
    }, indent=2))


if __name__ == "__main__":
    main()
