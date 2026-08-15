"""Row-level quantization sensitivity of the VLM tower, from the CoC loss alone.

For every output row c of every pool tensor l and every candidate bit-width b,

    S^(b)_{l,c} = | < grad_{W_{l,c}} L_CoC , eps^(b)_{l,c} > | ,   eps^(b) = Q_b(W) - W

accumulated over the calibration clips (absolute value per clip, then summed -- the same
convention `prune_lib` uses, which keeps a unit that matters in opposite directions on
different clips from cancelling itself out).

This is the exact deterministic RTN perturbation contracted with the task gradient, not a
stochastic-noise model, so no `E[delta^2] = step^2/12` assumption enters. It is still a
first-order approximation to the finite quantization effect; `--validate` measures how
well it ranks the real thing (Gate G0).

alpha = 0 is the operating point for every b at once, so ONE backward yields the scores
for all bit-widths. There is only one objective here -- the trajectory loss is never
computed -- so there is also only one backward per clip, which makes this pass cheaper
than `run_importance.py` rather than more expensive.

`eps` is recomputed inside backward from W rather than stored: keeping it for all
tensors and bit-widths would be ~65 GB, while recomputing costs one elementwise pass over
a row chunk. Peak memory is therefore one chunk of the largest weight, not the model.

Only the VLM pool is scored (VLM text Linear + lm_head + ViT Linear). The expert cannot
be scored by this criterion at all: `prune_lib.coc_nll` ends at `model.vlm.lm_head`, so
the expert is not in the CoC graph and every expert row would score exactly zero.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 20 \
      experiments/evaluation/run_qsens_coc.py --exp-id qsens_coc_v1 --gpu 7
  ... --num-clips 1 --probe            # memory probe, no writes
  ... --validate 200                   # Gate G0: Spearman(S, true dL) over 200 rows
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import prune_lib as pl
import quant_lib as ql
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch

MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
SCORE_BITS = (0, 2, 4, 8)          # b=16 has eps=0, hence score 0; never stored
ROW_CHUNK = 2048                   # bounds the (chunk, in_features) fp32 temporaries


class ScoredLinear(torch.autograd.Function):
    """F.linear whose backward also contracts dL/dW with each bit-width's RTN error."""

    @staticmethod
    def forward(ctx, x, W, bias, slot):
        ctx.save_for_backward(x, W)
        ctx.slot = slot
        ctx.has_bias = bias is not None
        return F.linear(x, W, bias)

    @staticmethod
    def backward(ctx, gy):
        x, W = ctx.saved_tensors
        gx = gy @ W                                        # (..., I)
        slot = ctx.slot
        if slot["on"]:
            g2 = gy.reshape(-1, gy.shape[-1])              # (N, O)
            x2 = x.reshape(-1, x.shape[-1])                # (N, I)
            O = W.shape[0]
            for lo in range(0, O, ROW_CHUNK):
                hi = min(lo + ROW_CHUNK, O)
                # dL/dW for this row chunk -- exactly what a weight backward computes
                gW = g2[:, lo:hi].t().float() @ x2.float()      # (R, I)
                Wc = W[lo:hi]
                for bit in SCORE_BITS:
                    eps = (-Wc.float() if bit == 0
                           else ql.quant_error(Wc, bit, slot["group"]).float())
                    slot["acc"][bit][lo:hi] += (gW * eps).sum(-1).abs()
                del gW
        gb = gy.reshape(-1, gy.shape[-1]).sum(0) if ctx.has_bias else None
        return gx, None, gb, None


def bind_scoring(model, parts, group):
    """Swap every pool Linear onto ScoredLinear. Returns {name: slot}.

    Pool weights are flipped to requires_grad=True. Not to collect their gradients --
    ScoredLinear returns None in the weight slot, so no `.grad` is ever allocated -- but
    because autograd only walks a subgraph that contains something requiring grad. The
    ViT is the case that forces this: `pixel_values` needs no grad and every weight is
    frozen, so with `enable_input_require_grads()` alone the whole vision tower sits
    outside the graph, its backward never fires, and all 306,544 ViT rows score exactly
    zero -- which reads as "the vision encoder does not matter for reasoning" when it
    actually means "the vision encoder was never measured".
    """
    slots = {}
    for name, mod in ql.pool_modules(model, parts).items():
        mod.weight.requires_grad_(True)
        O = mod.weight.shape[0]
        slot = {"on": True, "group": group, "name": name,
                "acc": {b: torch.zeros(O, device=mod.weight.device, dtype=torch.float32)
                        for b in SCORE_BITS}}
        slots[name] = slot
        mod._qslot = slot

        def fwd(self, x, _s=slot):
            return ScoredLinear.apply(x, self.weight, self.bias, _s)

        mod.forward = fwd.__get__(mod, type(mod))
    return slots


def set_checkpointing(model, on):
    """Toggle gradient checkpointing on both VLM towers.

    It cannot stay on across the whole clip: transformers forces use_cache=False whenever
    checkpointing is active in training mode, and the rollout needs the KV cache -- without
    it generation recomputes the RoPE index every decode step and crashes inside
    `get_rope_index`. So the rollout runs with it off and only the teacher-forced
    forward/backward, which already passes use_cache=False, runs with it on.
    """
    for sub in (model.vlm.model.visual, model.vlm.model.language_model):
        if on:
            sub.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            sub.train()          # only re-enables dropout, which is 0.0 in this config
        else:
            sub.gradient_checkpointing_disable()
            sub.eval()


def coc_backward(model, processor, data, seed, max_gen, checkpoint):
    """One clip: roll out the CoC, teacher-force it, backward the NLL. Returns the loss."""
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]
    del roll

    if checkpoint:
        set_checkpointing(model, True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, cache, _ = pl.vlm_forward_with_grad(
                model, seq_tf, inputs["tokenized_data"], use_cache=False)
        nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
        nll.backward()
        val = float(nll.detach())
        del hidden, cache, nll
    finally:
        if checkpoint:
            set_checkpointing(model, False)
    return val, coc_end - coc_start, prompt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="qsens_coc_v1")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--parts", nargs="+", default=["vlm_text", "lm_head", "vit"])
    ap.add_argument("--reserve-gb", type=float, default=42.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--checkpoint", action="store_true",
                    help="gradient checkpointing; needed to fit the ViT graph")
    ap.add_argument("--probe", action="store_true", help="report memory only, no writes")
    ap.add_argument("--probe-clips", type=int, default=5)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    clips = sc.calib_clips(REPO, args.calib_manifest)[: args.num_clips]
    if args.probe:
        clips = clips[: args.probe_clips]

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
    # every pool weight is frozen, so without this the early layers carry no graph
    model.vlm.enable_input_require_grads()

    slots = bind_scoring(model, args.parts, args.group)
    n_rows = sum(len(s["acc"][0]) for s in slots.values())
    shapes = ql.pool_shapes(model, args.parts)
    n_par = sum(o * i for o, i in shapes.values())
    print(f"scoring {len(slots)} tensors / {n_rows} rows / {n_par / 1e9:.4f} B params "
          f"| bits {SCORE_BITS} | group {args.group}", flush=True)
    # reserve_gpu allocates and frees `--reserve-gb` to hold it in the caching allocator,
    # which leaves that transient in the peak counter; reset so peak_gb is this pass's own
    resident = torch.cuda.memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    if not args.probe:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({
            "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
            "purpose": "row-level quantization sensitivity from the CoC loss only",
            "objective": "CoC NLL of the model's own rollout (label-free)",
            "trajectory_loss_used": False,
            "score": "S^(b)_{l,c} = |<dL_CoC/dW_{l,c}, Q_b(W)-W>|, summed abs over clips",
            "score_bits": list(SCORE_BITS), "group": args.group, "parts": args.parts,
            "pool_tensors": len(slots), "pool_rows": n_rows, "pool_params": n_par,
            "num_clips": len(clips), "clip_ids": clips, "seed": args.seed,
            "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
            "calib_manifest": args.calib_manifest, "max_gen": args.max_gen,
            "gpu": torch.cuda.get_device_name(device),
        }, indent=2))

    recs, t_start = [], time.time()
    for ci, clip_id in enumerate(clips):
        t0 = time.time()
        data = sc.load_cached(sc.path_for("calib", clip_id, sc.CALIB_T0))
        seed = sc.clip_seed(args.seed, clip_id)
        nll, coc_len, prompt_len = coc_backward(
            model, processor, data, seed, args.max_gen, args.checkpoint)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        recs.append({"clip_id": clip_id, "nll": nll, "coc_len": coc_len,
                     "prompt_len": prompt_len, "peak_gb": peak})
        print(f"[{ci + 1}/{len(clips)}] {clip_id[:8]} nll={nll:.4f} coc_len={coc_len} "
              f"peak={peak:.1f}GB ({time.time() - t0:.0f}s)", flush=True)

        if ci == 0:
            # a tensor whose backward never fired scores all-zero and would be demoted to
            # the cheapest bit-width by any allocator -- catch it here, not in the results
            dead = [n for n, s in slots.items()
                    if not any(bool((s["acc"][b] > 0).any()) for b in SCORE_BITS)]
            if dead:
                raise SystemExit(
                    f"{len(dead)}/{len(slots)} pool tensors scored exactly zero on clip 1, "
                    f"i.e. autograd never reached them: {dead[:4]} ... "
                    "check that their weights require grad and are on the CoC graph.")
            print(f"  graph check: all {len(slots)} pool tensors scored", flush=True)

        if not args.probe and ((ci + 1) % 10 == 0 or ci + 1 == len(clips)):
            np.savez(out_dir / "qsens.npz",
                     **{f"{n}|{b}": s["acc"][b].cpu().numpy()
                        for n, s in slots.items() for b in SCORE_BITS},
                     _names=np.array(list(slots), dtype=object),
                     _bits=np.array(SCORE_BITS),
                     _shapes=np.array([shapes[n] for n in slots]))

    if args.probe:
        # reset_peak_memory_stats seeds the peak at the resident weights, so peak_gb
        # already contains them; the activations/temporaries are the difference
        pk = max(r["peak_gb"] for r in recs)
        print(f"\nPROBE OK: {len(recs)} clips | weights resident {resident:.1f} GB | "
              f"peak {pk:.1f} GB (activations {pk - resident:.1f} GB) | "
              f"max coc_len {max(r['coc_len'] for r in recs)} | "
              f"{(time.time() - t_start) / len(recs):.0f}s per clip", flush=True)
        return

    (out_dir / "metrics.json").write_text(json.dumps(
        {"n_clips": len(recs), "per_clip": recs}, indent=2))
    nz = {b: sum(int((s["acc"][b] > 0).sum()) for s in slots.values()) for b in SCORE_BITS}
    line = (f"{args.exp_id}: {len(recs)} clips, {len(slots)} tensors, {n_rows} rows, "
            f"mean CoC NLL {np.mean([r['nll'] for r in recs]):.4f}, "
            f"nonzero rows per bit {nz}, {(time.time() - t_start) / 60:.1f} min")
    (out_dir / "summary.txt").write_text(line + "\n")
    print("\n" + line, flush=True)


if __name__ == "__main__":
    main()
