"""Wanda scores for the ACTION EXPERT tower (run_wanda.py covers the VLM only).

Same official WrappedGPT accumulation as `run_wanda.py` -- per input dim j of
o_proj / down_proj, a running mean over calibration clips of sum_tokens x_{t,j}^2 in
fp32 -- but the activations are collected from the expert's own forward instead of the
VLM's, i.e. during the ten-step Euler denoise that reads the VLM's KV cache as prefill.
Scores follow the same formulas:

  MLP channel c : sqrt(scaler_mlp[c]) * ||W_down[:, c]||_2
  Q head h      : sqrt( sum_{j in head} scaler_o[j] * ||W_o[:, j]||_2^2 )

Why summing over denoising steps is right here, unlike for Taylor: the shipped expert
Taylor score `|sum_s dL_s/dg|` was a defect because per-step gradients cancel in sign
(plans/2026-08-21_denoise-step-importance.md, fixed by znorm). Wanda's statistic is a
second moment, so steps cannot cancel -- summing over the ten steps is the same operation
as summing over tokens, which is what Wanda does by construction.

Exists to answer: on the axis where the expert is cheap to cut (MLP width, free to at
least 87.5% -- reports/evaluation/2026-08-28_expert-axis.html and
plans/2026-08-31_dualrwl-expert-mlp.md), does a gradient-free criterion pick the same
channels as the flow-matching Taylor score, or is the axis so slack that any criterion
works? Selection is measured on calib_100 only, never on an evaluation set.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_wanda_expert.py --exp-id wanda_expert_v1 [--gpu 0,1]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


class ExpertWandaStats:
    """WrappedGPT accumulation on the expert tower, one clip = one sample."""

    def __init__(self, layers):
        self.n = 0
        d_o = layers[0].self_attn.o_proj.in_features
        d_m = layers[0].mlp.down_proj.in_features
        self.scaler_o = [torch.zeros(d_o, dtype=torch.float32, device="cuda")
                         for _ in layers]
        self.scaler_m = [torch.zeros(d_m, dtype=torch.float32, device="cuda")
                         for _ in layers]
        self.handles = []
        for i, layer in enumerate(layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                self._hook(self.scaler_o, i)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                self._hook(self.scaler_m, i)))

    def _hook(self, store, i):
        def fn(_module, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()  # (steps*64, D)
            store[i] += (x * x).sum(0) / (self.n + 1)
        return fn

    def step(self):
        for s in (self.scaler_o, self.scaler_m):
            for t in s:
                t *= self.n / (self.n + 1)

    def done(self):
        self.n += 1

    def remove(self):
        for h in self.handles:
            h.remove()


def build_cache(model, processor, data, max_gen, seed):
    """Rollout + teacher-forced VLM forward -- the prefill the expert denoises against."""
    inputs = lib.build_inputs(model, processor, data, "cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
        seq_tf = roll["sequences"][:, : roll["eos_pos"] + 1]
        del roll
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
    return out.past_key_values, out.rope_deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, default="wanda_expert_v1")
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--k", type=int, default=1, help="denoise draws per clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None,
                    help="comma-separated card ids to restrict the scan")
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

    layers = model.expert.layers
    ec = model.expert.config
    stats = ExpertWandaStats(layers)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "Wanda (|W| * ||X||_2) structured unit scores for the ACTION EXPERT, "
                   "activations from the 10-step Euler denoise, no labels",
        "tower": "expert", "num_clips": len(calib),
        "clip_ids": [c for c, _ in calib],
        "calib_manifest": args.calib_manifest, "cache": args.cache,
        "k_draws": args.k, "max_gen": args.max_gen, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "tokens_desc": "all 64 diffusion tokens at each of the 10 denoising steps; "
                       "second moments cannot cancel across steps, unlike Taylor grads",
        "aggregation": {"primary": "L2 over unit columns", "secondary": "L1"},
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    records = []
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        seed = sc.clip_seed(args.seed, clip_id)
        cache, rope_deltas = build_cache(model, processor, data, args.max_gen, seed)
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        stats.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k in range(args.k):
                lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask,
                                       seed=seed + k)
        stats.done()
        del cache
        records.append({"clip_id": clip_id, "prefill": int(prefill)})
        print(f"[{ci + 1}/{len(calib)}] {clip_id} prefill={prefill} "
              f"({time.time() - t0:.0f}s)", flush=True)
    stats.remove()

    L, H, hd = ec.num_hidden_layers, ec.num_attention_heads, ec.head_dim
    inter = ec.intermediate_size
    q_w = np.zeros((L, H)); q_w_l1 = np.zeros((L, H))
    mlp_w = np.zeros((L, inter)); mlp_w_l1 = np.zeros((L, inter))
    sc_o = np.zeros((L, H * hd), dtype=np.float32)
    sc_m = np.zeros((L, inter), dtype=np.float32)
    for i, layer in enumerate(layers):
        so = stats.scaler_o[i]                                  # (H*hd,)
        sm = stats.scaler_m[i]                                  # (inter,)
        wo = layer.self_attn.o_proj.weight.float()              # (hidden, H*hd)
        wd = layer.mlp.down_proj.weight.float()                 # (hidden, inter)
        q_w[i] = (so * (wo * wo).sum(0)).view(H, hd).sum(1).sqrt().cpu().numpy()
        q_w_l1[i] = (so.sqrt() * wo.abs().sum(0)).view(H, hd).sum(1).cpu().numpy()
        mlp_w[i] = (sm.sqrt() * (wd * wd).sum(0).sqrt()).cpu().numpy()
        mlp_w_l1[i] = (sm.sqrt() * wd.abs().sum(0)).cpu().numpy()
        sc_o[i] = so.cpu().numpy()
        sc_m[i] = sm.cpu().numpy()

    np.savez(out_dir / "wanda.npz", exp_q_w=q_w, exp_mlp_w=mlp_w,
             exp_q_w_l1=q_w_l1, exp_mlp_w_l1=mlp_w_l1,
             scaler_o=sc_o, scaler_m=sc_m)
    (out_dir / "records.json").write_text(json.dumps(records, indent=1))
    finite = all(np.isfinite(a).all() for a in (q_w, mlp_w, q_w_l1, mlp_w_l1))
    (out_dir / "summary.txt").write_text(
        f"expert wanda scores over {len(calib)} clips, k={args.k} denoise draws\n"
        f"finite: {finite}\n"
        f"exp_q_w   range [{q_w.min():.4g}, {q_w.max():.4g}]\n"
        f"exp_mlp_w range [{mlp_w.min():.4g}, {mlp_w.max():.4g}]\n")
    print(f"saved -> {out_dir} | finite={finite}", flush=True)
    assert finite, "non-finite wanda scores"


if __name__ == "__main__":
    main()
