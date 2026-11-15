"""Structured Wanda on Alpamayo 2 Edge (plans/2026-11-15_a2-edge-baselines-u50.md). Edge venv.

The ICLR paper's Wanda row (`experiments/head_analysis/run_wanda.py --tokens all` -> make_slim `wanda_u*_v2`) ported to Edge:
  statistics  run_wanda.WandaStats: running mean over the 100 calib_100 clips of each clip's token sum of x^2 at the inputs of
              o_proj (Q heads) and mlp.fc2 (Edge's relu^2 MLP; A1.5: down_proj); prompt prefill only (no CoC, every position)
  scores      Q head h:      sqrt( sum_{j in h} scaler_o[j] * ||W_o[:, j]||^2 )
              MLP channel c: sqrt(scaler_m[c]) * ||W_fc2[:, c]||
  selection   make_slim's single-criterion path: mask_lib.select_mask_ratios with the same ratio in all 28 layers
              (u50 = literal 0.5: 8/16 heads, 4,608/9,216 channels, = MoOP u50's 645,922,816 parameters)
Inputs: calib_edge cache, Edge's no-nav prompt. Writes outputs/iclr27-rebuttal/edge_wanda_v1/{wanda.npz,config.json,summary.txt}
and outputs/iclr27-rebuttal/edge_moop/masks_wanda_u50.npz.

Gate GW: for the first clip, the hooked statistics of layer 14 equal a direct recomputation from captured activations.

Usage: CUDA_VISIBLE_DEVICES=4 /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/edge_wanda.py
"""

import os
import sys

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to one empty card")

import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import edge_grad_lib as gl
from mask_lib import select_mask_ratios


class WandaStats:
    """run_wanda.WandaStats with Edge's MLP input (mlp.fc2)."""

    def __init__(self, layers):
        self.n = 0
        d_o, d_m = layers[0].self_attn.o_proj.in_features, layers[0].mlp.fc2.in_features
        self.scaler_o = [torch.zeros(d_o, dtype=torch.float32, device="cuda") for _ in layers]
        self.scaler_m = [torch.zeros(d_m, dtype=torch.float32, device="cuda") for _ in layers]
        self.handles = []
        for i, layer in enumerate(layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook(self.scaler_o, i)))
            self.handles.append(layer.mlp.fc2.register_forward_pre_hook(self._hook(self.scaler_m, i)))

    def _hook(self, store, i):
        def fn(_module, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()  # (T, D)
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


def prefill(model, td):
    """Prompt-only VLM forward (run_wanda --tokens all), with Edge's mm_token_type_ids."""
    return model.vlm.model(input_ids=td["input_ids"], attention_mask=torch.ones_like(td["input_ids"]),
                           pixel_values=td["pixel_values"], image_grid_thw=td["image_grid_thw"],
                           mm_token_type_ids=td["mm_token_type_ids"], use_cache=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="iclr27-rebuttal/edge_wanda_v1")
    ap.add_argument("--pct", type=int, default=50)
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    from alpamayo2_edge import helper

    model, _ = ec.load_model("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer, model.config)
    layers = model.vlm.model.language_model.layers
    man = pd.read_parquet(ec.CALIB_MANIFEST)
    stats = WandaStats(layers)
    gate, n_tok, t0 = None, [], time.time()
    for i, r in enumerate(man.itertuples()):
        data = ec.load_cached(ec.path_for(r.clip_id, int(r.t0_us)))
        inputs, _ = ec.prepare_inputs_no_nav(data, model.config, model.tokenizer, processor)
        inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
        inputs["tokenized_data"] = {k: v.to("cuda") if torch.is_tensor(v) else v
                                    for k, v in inputs["tokenized_data"].items()}
        td = gl.fused_tokenized(model, inputs)
        cap = {}
        if i == 0:  # GW: capture layer 14's raw inputs alongside the hooks
            h1 = layers[14].self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, cap=cap: cap.__setitem__("o", a[0].reshape(-1, a[0].shape[-1]).float().clone()))
            h2 = layers[14].mlp.fc2.register_forward_pre_hook(
                lambda m, a, cap=cap: cap.__setitem__("m", a[0].reshape(-1, a[0].shape[-1]).float().clone()))
        stats.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prefill(model, td)
        stats.done()
        if i == 0:
            h1.remove()
            h2.remove()
            gate = {k: float(((cap[k] ** 2).sum(0) - s[14]).abs().max() / (cap[k] ** 2).sum(0).abs().max())
                    for k, s in (("o", stats.scaler_o), ("m", stats.scaler_m))}
            del cap
        n_tok.append(int(td["input_ids"].shape[1]))
        print(f"[{i + 1}/{len(man)}] {r.clip_id[:8]} T={n_tok[-1]} ({time.time() - t0:.0f}s)", flush=True)
    stats.remove()

    tc = model.config.vlm_config.text_config
    L, H, hd, inter = tc.num_hidden_layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size
    q_w, mlp_w = np.zeros((L, H)), np.zeros((L, inter))
    q_w_l1, mlp_w_l1 = np.zeros((L, H)), np.zeros((L, inter))
    for i, layer in enumerate(layers):
        so, sm = stats.scaler_o[i], stats.scaler_m[i]  # (H*hd,), (inter,)
        wo, wd = layer.self_attn.o_proj.weight.float(), layer.mlp.fc2.weight.float()  # (hidden, H*hd), (hidden, inter)
        q_w[i] = (so * (wo * wo).sum(0)).view(H, hd).sum(1).sqrt().cpu().numpy()
        q_w_l1[i] = (so.sqrt() * wo.abs().sum(0)).view(H, hd).sum(1).cpu().numpy()
        mlp_w[i] = (sm.sqrt() * (wd * wd).sum(0).sqrt()).cpu().numpy()
        mlp_w_l1[i] = (sm.sqrt() * wd.abs().sum(0)).cpu().numpy()
    np.savez(out / "wanda.npz", q_w=q_w, mlp_w=mlp_w, q_w_l1=q_w_l1, mlp_w_l1=mlp_w_l1,
             scaler_o=np.stack([s.cpu().numpy() for s in stats.scaler_o]),
             scaler_m=np.stack([s.cpu().numpy() for s in stats.scaler_m]))
    ratio = args.pct / 100
    q = select_mask_ratios(q_w, np.full(L, ratio))
    m = select_mask_ratios(mlp_w, np.full(L, ratio))
    kq, km = round(ratio * H), round(ratio * inter)
    assert ((q == 0).sum(1) == kq).all() and ((m == 0).sum(1) == km).all()
    removed = int((q == 0).sum() * 2 * hd * tc.hidden_size + (m == 0).sum() * 2 * tc.hidden_size)
    mask_path = REPO / "outputs" / "iclr27-rebuttal" / "edge_moop" / f"masks_wanda_u{args.pct}.npz"
    np.savez(mask_path, q_mask=q.astype(np.float32), mlp_mask=m.astype(np.float32))
    cfg = {"method": "structured Wanda (run_wanda.py --tokens all) ported to Edge", "model": ec.MODEL_ID,
           "model_revision": ec.MODEL_REV, "calib": str(ec.CALIB_MANIFEST), "cache": str(ec.CALIB_CACHE),
           "prompt": "Edge driving prompt without the navigation line, prefill only", "n_clips": len(man),
           "tokens_per_clip_mean": float(np.mean(n_tok)), "tokens_total": int(np.sum(n_tok)), "pct": args.pct,
           "cut_per_layer": {"q_heads": kq, "mlp_channels": km}, "removed_params": removed,
           "gate_GW_rel_err": gate, "masks": str(mask_path), "gpu": torch.cuda.get_device_name(0)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "summary.txt").write_text(f"Edge Wanda u{args.pct}: {kq} heads + {km} channels per layer, removed {removed:,}; "
                                     f"GW {gate}; {len(man)} clips, {np.mean(n_tok):.0f} tokens/clip\n")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
