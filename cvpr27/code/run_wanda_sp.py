"""Wanda-sp scores, layer-sequential as in FLAP's prune_wanda_sp (cvpr27/plans/2026-11-02_wandasp-closedloop-s600.md).

Official procedure (CASIA-IVA-Lab/FLAP lib/prune.py, prune_wanda_sp), adapted to Alpamayo's VLM text tower:
  - per input column j of o_proj / down_proj:  W_metric = |W| * sqrt(scaler_row[j]), averaged over output rows
    (scaler_row = WrappedGPT running mean over samples of sum_tokens x_j^2, fp32 -- same as run_wanda.py)
  - Q head score = sum of its 128 columns; MLP channel score = its column
  - every layer pruned at the same ratio (here MoOP's per-layer counts: 13 of 32 Q heads, 4,898 of 12,288 channels,
    2,657,452,032 params in all), no bias compensation
  - layer-sequential: layer L's statistics are taken with layers 0..L-1 already pruned (FLAP re-runs each pruned
    layer to feed the next). Here every clip is re-run through layers 0..L with the chosen units of 0..L-1 masked
    (o_proj / down_proj inputs zeroed, which is exact removal) and the forward stops after layer L.
Departures, as for the other baselines: calib_100 fused prompts (prefill tokens, label-free) instead of C4, and k/v
are never cut (the expert reads the VLM's 8 KV heads).

Writes outputs/<exp-id>/wanda.npz with q_w / mlp_w (the sequential scores, read by make_slim's `wanda` stem via
--wanda <exp-id>) and keep_q / keep_mlp (the units kept during collection, to check the build against).

Usage:
  bash experiments/head_analysis/run_retry_host.sh 2 cvpr27/code/run_wanda_sp.py --gpu 0 [--num-clips 100]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

CUT_Q, CUT_MLP = 13, 4898   # MoOP u40_v2 per-layer counts (defaults; u35 11/4301, u45 14/5530, u50 16/6144)


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, default="cvpr27/wandasp_v1")
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None, help="comma-separated card ids to restrict the scan")
    ap.add_argument("--cut-q", type=int, default=CUT_Q, help="Q heads cut per layer (MoOP u{N} count)")
    ap.add_argument("--cut-mlp", type=int, default=CUT_MLP, help="MLP channels cut per layer (MoOP u{N} count)")
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
    layers = model.vlm.model.language_model.layers
    tc = model.vlm.config.text_config
    L, H, hd, inter = tc.num_hidden_layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "Wanda-sp (FLAP prune_wanda_sp), layer-sequential, structured unit scores",
        "metric": "|W| * sqrt(scaler_row), mean over output rows; head = sum of its 128 columns",
        "sequential": "layer L scored with layers 0..L-1 pruned; forward stopped after layer L",
        "cut_per_layer": {"q_heads": args.cut_q, "mlp_channels": args.cut_mlp}, "layers": "all 36 (0..35)",
        "num_clips": len(calib), "clip_ids": [c for c, _ in calib],
        "calib_manifest": args.calib_manifest, "cache": args.cache, "tokens": "full fused prompt, prefill only",
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    # gates on o_proj / down_proj inputs: 1 = kept, 0 = removed (exact removal of a Q head / MLP channel)
    gate_o = [torch.ones(H * hd, device="cuda", dtype=torch.bfloat16) for _ in range(L)]
    gate_m = [torch.ones(inter, device="cuda", dtype=torch.bfloat16) for _ in range(L)]
    for i, layer in enumerate(layers):
        layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: (a[0] * gate_o[i],))
        layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: (a[0] * gate_m[i],))

    inputs = []
    for clip_id, clip_t0 in calib:
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        inputs.append(data)

    q_w = np.zeros((L, H)); mlp_w = np.zeros((L, inter))
    keep_q = np.ones((L, H), dtype=bool); keep_m = np.ones((L, inter), dtype=bool)
    sc_o = np.zeros((L, H * hd), dtype=np.float32); sc_m = np.zeros((L, inter), dtype=np.float32)
    t_all = time.time()
    for li in range(L):
        t0 = time.time()
        acc = {"o": torch.zeros(H * hd, dtype=torch.float32, device="cuda"),
               "m": torch.zeros(inter, dtype=torch.float32, device="cuda"), "n": 0}

        def stat(key, acc=acc):
            def fn(_m, a):
                x = a[0].reshape(-1, a[0].shape[-1]).float()               # (T, D), before this layer's gate
                acc[key] += (x * x).sum(0) / (acc["n"] + 1)
            return fn

        def stop(_m, _a, _o):
            raise Stop

        hs = [layers[li].self_attn.o_proj.register_forward_pre_hook(stat("o"), prepend=True),
              layers[li].mlp.down_proj.register_forward_pre_hook(stat("m"), prepend=True),
              layers[li].register_forward_hook(stop)]
        for data in inputs:
            for k in ("o", "m"):
                acc[k] *= acc["n"] / (acc["n"] + 1)                        # running mean, WrappedGPT style
            batch = lib.build_inputs(model, processor, data, "cuda")
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.vlm(input_ids=batch["input_ids"], use_cache=False, **batch["tokenized_data"])
            except Stop:
                pass
            acc["n"] += 1
        for h in hs:
            h.remove()

        layer = layers[li]
        wo = layer.self_attn.o_proj.weight.float()                         # (hidden, H*hd)
        wd = layer.mlp.down_proj.weight.float()                            # (hidden, inter)
        mo = (wo.abs() * acc["o"].sqrt()).mean(0)                          # (H*hd,)
        md = (wd.abs() * acc["m"].sqrt()).mean(0)                          # (inter,)
        q_w[li] = mo.view(H, hd).sum(1).cpu().numpy()
        mlp_w[li] = md.cpu().numpy()
        sc_o[li], sc_m[li] = acc["o"].cpu().numpy(), acc["m"].cpu().numpy()
        cut_q = np.argsort(q_w[li], kind="stable")[:args.cut_q]
        cut_m = np.argsort(mlp_w[li], kind="stable")[:args.cut_mlp]
        keep_q[li, cut_q] = False
        keep_m[li, cut_m] = False
        gq = torch.ones(H, device="cuda", dtype=torch.bfloat16)
        gq[torch.as_tensor(cut_q, device="cuda")] = 0
        gate_o[li] = gq.repeat_interleave(hd)
        gate_m[li][torch.as_tensor(cut_m, device="cuda")] = 0
        print(f"layer {li:2d}: cut Q {sorted(cut_q.tolist())} + {args.cut_mlp} MLP ch "
              f"({time.time() - t0:.0f}s, total {(time.time() - t_all) / 60:.1f} min)", flush=True)

    np.savez(out_dir / "wanda.npz", q_w=q_w, mlp_w=mlp_w, keep_q=keep_q, keep_mlp=keep_m,
             scaler_o=sc_o, scaler_m=sc_m)
    finite = all(np.isfinite(a).all() for a in (q_w, mlp_w))
    (out_dir / "summary.txt").write_text(
        f"Wanda-sp (sequential) over {len(calib)} clips, all {L} layers, cut {args.cut_q} Q heads + {args.cut_mlp} MLP ch "
        f"per layer\nfinite: {finite}\nkept Q {int(keep_q.sum())}, kept MLP {int(keep_m.sum())}\n"
        f"time {(time.time() - t_all) / 60:.1f} min\n")
    print(f"saved -> {out_dir} | finite={finite}", flush=True)
    assert finite, "non-finite wanda-sp scores"


if __name__ == "__main__":
    main()
