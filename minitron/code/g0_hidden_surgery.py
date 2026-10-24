"""G0a-c: is slim_lib.slice_hidden() the faithful physical form of a residual-channel cut?

Plan minitron/plans/2026-10-23_plan.md section 6. Run before any importance or training
result is trusted. Probe = for two clips, the CoC-position logits of a teacher-forced
[prompt + CoC] forward, the K/V the expert reads at three cache layers, and the expert
velocity at one fixed (noise, t) -- the three quantities the KD loss and the action path
touch.

  G0a identity: slice_hidden(range(4096)) on the dense model -> every probe tensor
      bit-identical to the dense model's.
  G0b equivalence: a random kept set of 3,328 channels, sliced physically, against a masked
      reference built in place on the dense model: removed channels zeroed in the
      embedding, every RMSNorm weight, o/down output rows and the vision merger / DeepStack
      merger outputs, so they stay exactly zero in the residual; RMSNorm then averages over
      4096 instead of 3328 channels, which the reference undoes with weight * sqrt(r) and
      eps * r (r = 3328/4096) -- mathematically identical to the sliced model. The two
      differ only in bf16 accumulation order (dot products of length 4096 with zeros vs
      3328), so the gate is a numerical tolerance, fixed here before the run:
      PASS iff CoC-position argmax agreement >= 0.99, and the relative L2 error of the
      logits, of the K/V and of the expert velocity are each < 1e-2.
      (The plan said < 1e-3 before this was written; bf16 over 36 layers does not support
      that, and a slicing bug -- a wrong axis or a forgotten merger -- shows up as O(1)
      error or a shape failure, so 1e-2 still separates the two.)
  G0c accounting + round trip: hidden 3328 + MLP 9216 per layer (random kept sets) removes
      exactly 2,659,847,424 parameters, and save_slim(meta only) -> load_slim rebuilds a
      model whose probe tensors are bit-identical to the in-memory one.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \\
      minitron/code/g0_hidden_surgery.py --gpu 5
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "recovery"))

import analysis_lib as lib
import recover_lib as rl
import sample_cache as sc
import slim_lib as sl
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import MODEL_REV, set_determinism

H, H_KEEP, I_KEEP = 4096, 3328, 9216
EXPECTED_REMOVED = 2_659_847_424
KV_LAYERS = (0, 17, 35)
T_PROBE = 0.5


def load_dense():
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def clips():
    """One distillation-train window (with its teacher CoC) and one official-val window."""
    import mt_data
    r1 = mt_data.r1_windows()
    r1 = r1[[sc.path_for("train", c, int(t)).exists() for c, t in zip(r1.clip_id, r1.t0_us)]]
    tr = r1.iloc[0]
    val = pd.read_parquet(REPO / "outputs" / "recovery_sets" / "val_official_238.parquet").iloc[0]
    # the val window has no stored teacher CoC; reuse the train CoC text -- the probe only
    # needs the same input on both sides of each comparison
    return [("train", tr.clip_id, int(tr.t0_us), tr.coc),
            ("eval", val.clip_id, int(val.t0_us), tr.coc)]


def expert_velocity(model, cache, rope_deltas, x_t, t_val, prefill):
    """fm_loss_insulated's forward, returning the velocity instead of the loss. (1, 64, 2)"""
    device = x_t.device
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask)
    kw = {"is_causal": False} if model.config.expert_non_causal_attention else {}
    t = torch.full((1, 1, 1), t_val, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeds = model.action_in_proj(x_t.to(torch.bfloat16), t)  # (1, 64, 2048)
        if embeds.dim() == 2:
            embeds = embeds.view(1, n_tok, -1)
        out = model.expert(inputs_embeds=embeds, position_ids=position_ids,
                           past_key_values=cache, attention_mask=attention_mask,
                           use_cache=True, **kw)
        cache.crop(prefill)
        return model.action_out_proj(out.last_hidden_state[:, -n_tok:]).float()


@torch.no_grad()
def probe(model, processor, tok, items):
    res = []
    for ns, clip_id, t0_us, coc_text in items:
        data = sc.load_cached(sc.path_for(ns, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        p = inputs["input_ids"].shape[1]
        seq = torch.cat([inputs["input_ids"],
                         torch.tensor([rl.coc_ids(tok, coc_text)], device="cuda")], dim=1)
        ce = seq.shape[1]
        out, cache, rope_deltas = rl.vlm_forward(model, inputs, seq)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.vlm.lm_head(out.last_hidden_state[:, p - 1 : ce - 1])  # (1, L, V)
        kv = torch.stack([torch.stack(lib.cache_layer_kv(cache, i)) for i in KV_LAYERS])
        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
        noise = torch.randn(x1.shape, generator=torch.Generator().manual_seed(0)).to("cuda")
        x_t = (1.0 - T_PROBE) * noise + T_PROBE * x1
        v = expert_velocity(model, cache, rope_deltas, x_t, T_PROBE, ce)  # (1, 64, 2)
        res.append({"logits": logits[0].float().cpu(), "kv": kv.float().cpu(),
                    "v": v[0].cpu()})
        del out, cache
    return res


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def compare(a, b):
    return {
        "argmax_agree": float(np.mean([(x["logits"].argmax(-1) == y["logits"].argmax(-1)).float().mean()
                                       for x, y in zip(a, b)])),
        "rel_logits": max(rel(x["logits"], y["logits"]) for x, y in zip(a, b)),
        "rel_kv": max(rel(x["kv"], y["kv"]) for x, y in zip(a, b)),
        "rel_v": max(rel(x["v"], y["v"]) for x, y in zip(a, b)),
        "bit_identical": all(torch.equal(x[k], y[k]) for x, y in zip(a, b)
                             for k in ("logits", "kv", "v")),
    }


@torch.no_grad()
def mask_reference(model, removed, r):
    """Zero the removed residual channels everywhere they enter, rescale RMSNorm."""
    lm = model.vlm.model.language_model
    idx = torch.as_tensor(removed, dtype=torch.long, device=lm.embed_tokens.weight.device)
    s = math.sqrt(r)

    def norm_(norm):
        norm.weight[idx] = 0
        norm.weight.mul_(s)
        norm.variance_epsilon *= r

    lm.embed_tokens.weight[:, idx] = 0
    for layer in lm.layers:
        norm_(layer.input_layernorm)
        norm_(layer.post_attention_layernorm)
        layer.self_attn.o_proj.weight[idx] = 0
        layer.mlp.down_proj.weight[idx] = 0
    norm_(lm.norm)
    vis = model.vlm.model.visual
    for merger in (vis.merger, *vis.deepstack_merger_list):
        merger.linear_fc2.weight[idx] = 0
        merger.linear_fc2.bias[idx] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="g0_hidden_surgery")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = REPO / "outputs" / "minitron" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    rng = np.random.default_rng(args.seed)
    kept_h = np.sort(rng.choice(H, H_KEEP, replace=False))
    removed_h = np.setdiff1d(np.arange(H), kept_h)
    items = clips()
    res = {"gpu": gpu_name, "kept_h_seed": args.seed}

    model = load_dense()
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    n_dense = sl.n_params(model)
    ref = probe(model, processor, tok, items)

    # G0a: identity slice
    sl.slice_hidden(model, np.arange(H))
    res["G0a"] = compare(probe(model, processor, tok, items), ref)
    res["G0a"]["pass"] = res["G0a"]["bit_identical"]
    del model
    torch.cuda.empty_cache()

    # G0b: masked reference vs physical slice
    model = load_dense()
    mask_reference(model, removed_h, H_KEEP / H)
    masked = probe(model, processor, tok, items)
    del model
    torch.cuda.empty_cache()
    model = load_dense()
    sl.slice_hidden(model, kept_h)
    sliced = probe(model, processor, tok, items)
    res["G0b"] = compare(sliced, masked)
    res["G0b"]["vs_dense"] = compare(sliced, ref)  # how far the cut itself moves things
    g = res["G0b"]
    g["pass"] = (g["argmax_agree"] >= 0.99 and g["rel_logits"] < 1e-2 and g["rel_kv"] < 1e-2
                 and g["rel_v"] < 1e-2)
    del model
    torch.cuda.empty_cache()

    # G0c: accounting + meta-only round trip, through apply_surgery as every arm is built
    model = load_dense()
    tc = model.vlm.config.text_config
    ec = model.expert.config
    vm = np.zeros((tc.num_hidden_layers, tc.intermediate_size))
    for li in range(tc.num_hidden_layers):
        vm[li, rng.choice(tc.intermediate_size, I_KEEP, replace=False)] = 1.0
    meta = sl.apply_surgery(
        model, np.ones((tc.num_hidden_layers, tc.num_attention_heads)), vm,
        np.ones((ec.num_hidden_layers, ec.num_attention_heads)),
        np.ones((ec.num_hidden_layers, ec.intermediate_size)), vlm_hidden=kept_h)
    sl.check_slim(model)
    removed = n_dense - sl.n_params(model)
    built = probe(model, processor, tok, items)
    ckpt = out_dir / "roundtrip_ckpt"
    sl.save_slim(model, meta, ckpt, write_state=False)
    del model
    torch.cuda.empty_cache()
    model = sl.load_slim(ckpt, device="cuda")
    res["G0c"] = {"removed": int(removed), "expected": EXPECTED_REMOVED,
                  "roundtrip": compare(probe(model, processor, tok, items), built)}
    res["G0c"]["pass"] = (removed == EXPECTED_REMOVED
                          and res["G0c"]["roundtrip"]["bit_identical"])

    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "H_keep": H_KEEP, "I_keep": I_KEEP, "kv_layers": KV_LAYERS,
        "t_probe": T_PROBE, "clips": [i[:3] for i in items], "gpu": gpu_name}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    lines = [f"G0 hidden surgery ({gpu_name})"]
    for k in ("G0a", "G0b"):
        g = res[k]
        lines.append(f"{k}: {'PASS' if g['pass'] else 'FAIL'} | bit-identical {g['bit_identical']} "
                     f"argmax {g['argmax_agree']:.4f} rel logits {g['rel_logits']:.2e} "
                     f"kv {g['rel_kv']:.2e} v {g['rel_v']:.2e}")
    vd = res["G0b"]["vs_dense"]
    lines.append(f"     (sliced vs dense: argmax {vd['argmax_agree']:.4f} rel logits "
                 f"{vd['rel_logits']:.2e} kv {vd['rel_kv']:.2e} v {vd['rel_v']:.2e})")
    g = res["G0c"]
    lines.append(f"G0c: {'PASS' if g['pass'] else 'FAIL'} | removed {g['removed']:,} "
                 f"(expected {g['expected']:,}) | round trip bit-identical "
                 f"{g['roundtrip']['bit_identical']}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
