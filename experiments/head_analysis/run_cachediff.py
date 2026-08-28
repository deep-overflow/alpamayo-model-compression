"""Does the dual-pruned VLM hand the expert a different KV cache, and does that explain
the composition cost?  (plans/2026-08-26_dual-plus-znorm.md follow-up)

`dualexp_u40_e25` costs +0.0688 over dual, five times the proportional prediction. The
standing explanation is an interface shift: expert znorm r25 picked its kept set from
gradients measured on the DENSE VLM's cache, but at inference it reads the PRUNED VLM's
cache. This runner measures that shift and tests whether it is the cause.

Instrument -- one dense model, two mask sets, two caches. `mask_lib.PruneMasks` attaches
to either tower (run_integrated.py, run_kvfix.py do the same), and masking is functionally
removal for these units, so both configurations are reachable by toggling hooks instead of
loading two checkpoints. Per clip:

  1. rollout with everything unmasked -> seq_tf frozen, so every cell is scored against
     the same CoC text and cache positions align (run_kviso / run_pathway convention)
  2. VLM masks off -> teacher-forced forward -> cache_D
  3. VLM masks on  -> same seq_tf         -> cache_P
  4. 2x2 denoise over {cache_D, cache_P} x {expert off, expert znorm r25}, seeds shared

  A00 = dense cache + dense expert      A01 = dense cache + pruned expert
  A10 = pruned cache + dense expert     A11 = pruned cache + pruned expert   (= combined)

  interaction I = (A11 - A10) - (A01 - A00)  -- "how much MORE the expert cut costs once
  the cache it reads has shifted". I ~ 0 means the penalty is additive and the interface
  story is not what is going on; the dualexp report's median analysis predicts exactly
  that, so the run is designed to be informative either way.

Stages B and C ride along on the same two forwards, at no extra cost:
  B  per (layer, KV group) divergence -- cosine for K (k_norm makes it directional),
     relative L2 for V -- split by token span, plus expert attention mass/entropy under
     each cache (--attn-stats, eager attention).
  C  second moments A = sum k_P k_P^T, B = sum k_P k_D^T, C = sum k_D k_D^T per
     (layer, group), which give the optimal linear map M = (A + lambda I)^-1 B and its
     residual in closed form, with no stored activations. Accumulated per fold (clip
     parity) so the fit and the residual can be read on disjoint clips.

Gates (plan): A0 integrity -- layer-0 cache must be bit-identical (the masks sit on
o_proj/down_proj inputs, so layer 0's k/v cannot move) and A00/A10 must land on the
published baseline / dual within the masked-vs-slim floor; A1 -- paired CI on I.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 60 experiments/head_analysis/run_cachediff.py \
      --gpu 4 --num-clips 200
  # shard: --clip-offset 50 --num-clips 50 --exp-id cachediff_v1_s50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from make_slim import build_masks  # noqa: E402
from slim_lib import MODEL_REV  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CELLS = ["A00_denseC_denseE", "A01_denseC_prunE", "A10_prunC_denseE", "A11_prunC_prunE"]
SPANS = ["all", "vision", "text", "hist", "sink", "coc"]


def span_index(spans, prompt_len, coc_start, coc_end, prefill, device):
    """Cache-position boolean masks. compute_spans covers the prompt; CoC is appended."""
    out = {}
    for k in ("vision", "text", "hist", "sink"):
        m = torch.zeros(prefill, dtype=torch.bool, device=device)
        m[:prompt_len] = spans[k].to(device)
        out[k] = m
    coc = torch.zeros(prefill, dtype=torch.bool, device=device)
    coc[coc_start:min(coc_end, prefill)] = True
    out["coc"] = coc
    out["all"] = torch.ones(prefill, dtype=torch.bool, device=device)
    return out


@torch.no_grad()
def cache_divergence(cache_d, cache_p, n_layers, sidx, div, moments, fold):
    """Per (layer, group) divergence + the second moments Stage C solves from.

    K is compared by direction because Qwen3's k_norm normalises it per head; V by
    relative magnitude because nothing normalises it.
    """
    for li in range(n_layers):
        kd, vd = lib.cache_layer_kv(cache_d, li)
        kp, vp = lib.cache_layer_kv(cache_p, li)
        kd, vd = kd[0].float(), vd[0].float()      # (G, T, D)
        kp, vp = kp[0].float(), vp[0].float()
        cos_k = F.cosine_similarity(kd, kp, dim=-1)   # (G, T)
        cos_v = F.cosine_similarity(vd, vp, dim=-1)
        for name, m in sidx.items():
            if not bool(m.any()):
                continue
            div[name]["cos_k"][li] += cos_k[:, m].mean(1).cpu().numpy()
            div[name]["cos_v"][li] += cos_v[:, m].mean(1).cpu().numpy()
            for tag, a, b in (("k", kd, kp), ("v", vd, vp)):
                da = (a[:, m] - b[:, m]).norm(dim=(1, 2))
                na = a[:, m].norm(dim=(1, 2)).clamp_min(1e-12)
                div[name][f"rel_{tag}"][li] += (da / na).cpu().numpy()
        # Stage C: X = pruned (the input we would correct), Y = dense (the target)
        m = sidx["all"]
        for tag, y, x in (("k", kd, kp), ("v", vd, vp)):
            X = x[:, m].double()                      # (G, n, D)
            Y = y[:, m].double()
            moments[tag]["A"][fold, li] += torch.einsum("gnd,gne->gde", X, X).cpu().numpy()
            moments[tag]["B"][fold, li] += torch.einsum("gnd,gne->gde", X, Y).cpu().numpy()
            moments[tag]["C"][fold, li] += torch.einsum("gnd,gne->gde", Y, Y).cpu().numpy()


@torch.no_grad()
def denoise_minade(model, inputs, cache, rope_deltas, prefill, gt_xy, seeds):
    device = inputs["input_ids"].device
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    preds = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask, seed=s)
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    return el.min_metrics(np.stack(preds), gt_xy)


@torch.no_grad()
def tf_forward(model, seq_tf, inputs, coc_start, coc_end):
    out = model.vlm.model(
        input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
        pixel_values=inputs["tokenized_data"]["pixel_values"],
        image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
    )
    logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1: coc_end - 1]).float()
    nll = float(F.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end]))
    return out.past_key_values, out.rope_deltas, nll


def save(out_dir, res, div, moments, meta, n):
    (out_dir / "metrics.json").write_text(json.dumps({**meta, "n_clips": n, "cells": res},
                                                     indent=2))
    arrays = {}
    for name, d in div.items():
        for k, v in d.items():
            arrays[f"div_{name}_{k}"] = v / max(n, 1)
    for tag, d in moments.items():
        for k, v in d.items():
            arrays[f"mom_{tag}_{k}"] = v
    np.savez(out_dir / "cachediff.npz", **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cachediff_v1")
    ap.add_argument("--num-clips", type=int, default=200)
    ap.add_argument("--clip-offset", type=int, default=0)
    ap.add_argument("--manifest", default="indist_500")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--importance", default="importance_stepexp_bw_znorm",
                    help="supplies both mask halves: VLM keys == importance_v2, expert "
                         "keys == znorm; verified to reproduce slim_dualexp_u40_e25")
    ap.add_argument("--config", default="dualexp_u40_e25",
                    help="build_masks config whose VLM half is dual_u40_v2 and whose "
                         "expert half is the arm under test")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--attn-stats", action="store_true",
                    help="also collect expert attention mass/entropy under each cache "
                         "(switches the expert to eager attention, ~2 extra denoises/clip)")
    ap.add_argument("--reserve-gb", type=float, default=34.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO / "outputs" / args.sets_id / f"{args.manifest}.parquet")
    rows = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us)} for r in df.itertuples()]
    rows = rows[args.clip_offset: args.clip_offset + args.num_clips]

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

    tc, ec = model.vlm.config.text_config, model.expert.config
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    vq, vm, eq, em, kvonly = build_masks(args.config, imp, model)
    assert not kvonly, "this diagnostic assumes no KV drop"
    vmasks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                           tc.head_dim, tc.intermediate_size, "cuda")
    emasks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                           ec.intermediate_size, "cuda")
    n_layers, n_kv = tc.num_hidden_layers, tc.num_key_value_heads
    print(f"VLM keep q={vq.mean():.4f} mlp={vm.mean():.4f} | "
          f"expert keep q={eq.mean():.4f} mlp={em.mean():.4f}", flush=True)

    res = {c: {"ade": [], "fde": []} for c in CELLS}
    res["nll"] = {"dense": [], "pruned": []}
    res["layer0_max_dk"] = []
    div = {s: {k: np.zeros((n_layers, n_kv)) for k in ("cos_k", "cos_v", "rel_k", "rel_v")}
           for s in SPANS}
    moments = {t: {k: np.zeros((2, n_layers, n_kv, tc.head_dim, tc.head_dim))
                   for k in ("A", "B", "C")} for t in ("k", "v")}
    attn = {} if args.attn_stats else None
    clip_ids, buckets = [], []

    meta = {
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "KV-cache shift between dense and dual-pruned VLM, and whether it "
                   "explains the dualexp composition cost",
        "plan": "plans/2026-08-26_dual-plus-znorm.md follow-up (cache-shift analysis)",
        "manifest": args.manifest, "cache": args.cache, "clip_offset": args.clip_offset,
        "k_samples": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "config": args.config, "importance": args.importance,
        "protocol": ("one unmasked rollout per clip fixes seq_tf; two teacher-forced "
                     "forwards (VLM masks off/on) give the two caches; 2x2 denoise shares "
                     "seeds, so all four cells are clip-paired"),
        "cells": {"A00": "dense cache + dense expert", "A01": "dense cache + pruned expert",
                  "A10": "pruned cache + dense expert", "A11": "pruned cache + pruned expert"},
        "spans": SPANS, "gpu": torch.cuda.get_device_name(device),
    }
    (out_dir / "config.json").write_text(json.dumps({**meta, "clip_ids": [r["clip_id"] for r in rows]},
                                                    indent=2))

    for ci, r in enumerate(rows):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, r["clip_id"], r["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        spans = lib.compute_spans(model, inputs["input_ids"])
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
        base = sc.clip_seed(args.seed, r["clip_id"])
        seeds = [base + k for k in range(args.k)]

        vmasks.reset()
        emasks.reset()
        torch.manual_seed(base)
        torch.cuda.manual_seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end].clone()
        del roll

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache_d, rope_d, nll_d = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
            vmasks.set(q=vq, mlp=vm)
            cache_p, rope_p, nll_p = tf_forward(model, seq_tf, inputs, coc_start, coc_end)
            vmasks.reset()
        prefill = cache_d.get_seq_length()
        assert cache_p.get_seq_length() == prefill, "cache lengths diverged"

        # A0(i): the masks sit downstream of layer 0's k/v, so layer 0 must be identical
        k0d, _ = lib.cache_layer_kv(cache_d, 0)
        k0p, _ = lib.cache_layer_kv(cache_p, 0)
        res["layer0_max_dk"].append(float((k0d.float() - k0p.float()).abs().max()))

        sidx = span_index(spans, prompt_len, coc_start, coc_end, prefill, k0d.device)
        cache_divergence(cache_d, cache_p, n_layers, sidx, div, moments, ci % 2)

        for cell, cache, rope, use_expert_mask in (
            ("A00_denseC_denseE", cache_d, rope_d, False),
            ("A01_denseC_prunE", cache_d, rope_d, True),
            ("A10_prunC_denseE", cache_p, rope_p, False),
            ("A11_prunC_prunE", cache_p, rope_p, True),
        ):
            emasks.set(q=eq, mlp=em) if use_expert_mask else emasks.reset()
            ade, fde = denoise_minade(model, inputs, cache, rope, prefill, gt_xy, seeds)
            res[cell]["ade"].append(ade)
            res[cell]["fde"].append(fde)
        emasks.reset()

        if args.attn_stats:
            lib.set_expert_attn_impl(model, "eager")
            emasks.set(q=eq, mlp=em)
            for tag, cache, rope in (("dense", cache_d, rope_d), ("pruned", cache_p, rope_p)):
                col = lib.ExpertStatsCollector(ec.num_hidden_layers, ec.num_attention_heads,
                                               spans, prompt_len, coc_start, coc_end, prefill)
                col.register(model)
                denoise_minade(model, inputs, cache, rope, prefill, gt_xy, seeds[:1])
                col.remove()
                calls = np.maximum(col.calls, 1)[:, None]
                for k, v in col.sums.items():
                    attn.setdefault(tag, {}).setdefault(k, np.zeros_like(v))
                    attn[tag][k] += v / calls
                attn.setdefault(tag, {}).setdefault("headnorm", np.zeros_like(col.headnorm))
                attn[tag]["headnorm"] += col.headnorm / calls
            emasks.reset()
            lib.set_expert_attn_impl(model, "sdpa")

        res["nll"]["dense"].append(nll_d)
        res["nll"]["pruned"].append(nll_p)
        clip_ids.append(r["clip_id"])
        buckets.append(el.bucket(gt_xy))
        del cache_d, cache_p
        torch.cuda.empty_cache()

        i_val = ((res["A11_prunC_prunE"]["ade"][-1] - res["A10_prunC_denseE"]["ade"][-1])
                 - (res["A01_denseC_prunE"]["ade"][-1] - res["A00_denseC_denseE"]["ade"][-1]))
        print(f"[{ci + 1}/{len(rows)}] {r['clip_id']} {buckets[-1]:10s} "
              f"A00={res['A00_denseC_denseE']['ade'][-1]:.3f} "
              f"A10={res['A10_prunC_denseE']['ade'][-1]:.3f} "
              f"A11={res['A11_prunC_prunE']['ade'][-1]:.3f} I={i_val:+.3f} "
              f"L0dk={res['layer0_max_dk'][-1]:.1e} ({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 5 == 0 or ci + 1 == len(rows):
            m = {**meta, "clip_ids": clip_ids, "buckets": buckets}
            if attn:
                m["attn"] = {t: {k: (v / (ci + 1)).tolist() for k, v in d.items()}
                             for t, d in attn.items()}
            save(out_dir, res, div, moments, m, ci + 1)

    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
