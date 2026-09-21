"""Port map: through WHICH cache layer, at which positions, does the FM loss reach a VLM unit?

Follow-up to run_gradient_anatomy.py (plans/2026-09-20_gradient-anatomy.md, section
"Follow-up: port map"). That pass split the FM seed into six cache-layer bands and found the
band 22-24 -- three layers -- carrying more of the mid-layer units' FM gradient than any
other, and closing exactly where I_traj steps down: a unit in layer l can only write into
cache layers above l. Six bands cannot say whether that is one port or a smooth profile cut
at an unlucky boundary, so this pass repeats the split at single-layer resolution, and
crosses it with the position group of the cache entries (vision vs everything else).

Per clip: the shipped expert pass, one full-seed VLM backward (the reference), then
36 cache layers x 2 position groups = 72 partial-seed backwards on the retained graph. A
backward seeded at cache layer m only traverses layers below m, so they are cheap on
average. No CE objective here: it has no cache port.

Two reductions per (cache layer m, position group p), both per clip before averaging:
  mass    sum_u |G_mp,u|                      -- how much gradient arrives through the port
  signed  sum_u G_mp,u * sign(G_full,u)       -- the port's ADDITIVE share of the shipped
                                                 score: summed over all ports it returns
                                                 sum_u |G_full,u| = the layer's I_traj
each also split by the token type the unit acts on (TypedUnitGates). The MLP axis is
reduced over units on the fly (72 x (5, 36, 12288) per clip would not fit anywhere); the
Q-head axis keeps every unit.

Usage (three free cards, round-robin shards; analyze_port_map.py merges):
  for s in 0 1 2; do ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 2 \
      experiments/head_analysis/run_port_map.py --gpu $((5+s)) --exp-id portmap_v1_s$s \
      --shard $s --n-shards 3 & done
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
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_gradient_anatomy import MODEL_REV, TYPES, token_types  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GROUPS = ("vision", "rest")  # position group of the cache entries the seed is kept on


def process_clip(model, processor, data, args, seed):
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    coc_end = roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
    del roll

    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
    type_idx = token_types(model, seq_tf, prompt_len)  # (T,)

    tc = model.vlm.config.text_config
    layers = model.vlm.model.language_model.layers
    n_vlm = len(layers)
    gates = pl.TypedUnitGates(layers, tc.num_attention_heads, tc.head_dim,
                              tc.intermediate_size, len(TYPES), "cuda")
    gates.set_types(type_idx)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=True
        )
    cache_t = [lib.cache_layer_kv(cache, i) for i in range(n_vlm)]
    prefill = cache.get_seq_length()

    fm_loss, grads, leaves = pl.expert_fm_grads(
        model, cache, rope_deltas, x1, args.fm_steps, seed, prefill
    )

    def read():
        q, mlp = gates.q_signed(), gates.mlp_signed()  # (5, L, H), (5, L, I)
        gates.zero_grads()
        return q, mlp

    pl.vlm_backward_from_cache(cache_t, grads, retain=True)
    full_q, full_mlp = read()
    sgn_mlp = np.sign(full_mlp.sum(0))  # (L, I)

    vis = (type_idx == TYPES.index("vision")).view(1, 1, -1, 1)
    keep = (vis, ~vis)
    nG, nT = len(GROUPS), len(TYPES)
    port_q = np.zeros((n_vlm, nG, nT, n_vlm, tc.num_attention_heads), np.float32)  # signed, per unit
    mass = np.zeros((n_vlm, nG, nT, n_vlm))  # sum_u |G| by unit-side type
    signed = np.zeros((n_vlm, nG, nT, n_vlm))  # sum_u G * sign(G_full) by unit-side type
    tot = np.zeros((n_vlm, nG, n_vlm))  # sum_u |sum_type G|
    sum_q = np.zeros_like(full_q.sum(0))
    for m in range(n_vlm):
        for p in range(nG):
            seeds = [(None, None)] * n_vlm
            seeds[m] = (grads[m][0] * keep[p], grads[m][1] * keep[p])  # (1, KV, T, D) each
            pl.vlm_backward_from_cache(cache_t, seeds,
                                       retain=not (m == n_vlm - 1 and p == nG - 1))
            q, mlp = read()
            port_q[m, p] = q
            sum_q += q.sum(0)
            mass[m, p] = np.abs(mlp).sum(-1)
            signed[m, p] = (mlp * sgn_mlp[None]).sum(-1)
            tot[m, p] = np.abs(mlp.sum(0)).sum(-1)
            del seeds

    tq = full_q.sum(0)  # (L, H)
    relerr = (np.linalg.norm(sum_q - tq, axis=1) / np.maximum(np.linalg.norm(tq, axis=1), 1e-300))[:35]
    peak = torch.cuda.max_memory_allocated() / 1024**3
    gates.remove()
    out = {"port_q": port_q, "full_q": full_q.astype(np.float32), "mlp_mass": mass,
           "mlp_signed": signed, "mlp_tot": tot,
           "mlp_full": np.abs(full_mlp.sum(0)).sum(-1),  # (L,) the shipped layer sum
           "mlp_full_type_signed": (full_mlp * sgn_mlp[None]).sum(-1)}  # (5, L)
    rec = {"coc_len": int(coc_end - prompt_len), "fm_loss": float(fm_loss),
           "peak_gb": round(peak, 2), "ports_vs_full_relerr_by_layer": relerr.tolist(),
           # the additive identity on the MLP axis: ports' signed shares vs the shipped sum
           "mlp_signed_vs_full": float(signed.sum((0, 1, 2))[:35].sum() / out["mlp_full"][:35].sum())}
    del cache, cache_t, leaves, grads, inputs
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--shard", type=int, default=0,
                    help="round-robin shard index; seeds come from the clip id, so a clip's "
                         "numbers do not depend on which shard measured it")
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    if args.n_shards > 1:
        calib = calib[args.shard::args.n_shards]
        print(f"shard {args.shard}/{args.n_shards}: {len(calib)} clips", flush=True)

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
    model.vlm.enable_input_require_grads()

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "FM gradient to VLM units by single cache layer x position group",
        "plan": "plans/2026-09-20_gradient-anatomy.md (follow-up: port map)",
        "types": TYPES, "groups": GROUPS, "num_clips": len(calib),
        "clip_ids": [c for c, _ in calib], "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]", "calib_manifest": args.calib_manifest,
        "cache": args.cache, "fm_steps": args.fm_steps, "max_gen": args.max_gen,
        "shard": args.shard, "n_shards": args.n_shards,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
    }, indent=2))

    acc, records = {}, []
    for ci, (clip_id, clip_t0) in enumerate(calib):
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, clip_t0))
        torch.cuda.reset_peak_memory_stats()
        g, rec = process_clip(model, processor, data, args, sc.clip_seed(args.seed, clip_id))
        sgn = np.sign(g["full_q"].sum(0))  # (L, H)
        red = {"q_mass": np.abs(g["port_q"]).sum(-1),  # (36, 2, 5, L)
               "q_signed": (g["port_q"] * sgn[None, None, None]).sum(-1),  # (36, 2, 5, L)
               "q_tot": np.abs(g["port_q"].sum(2)).sum(-1),  # (36, 2, L)
               "q_full": np.abs(g["full_q"].sum(0)).sum(-1),  # (L,)
               "q_port_abs": np.abs(g["port_q"].sum(2)),  # (36, 2, L, H) per unit, types pooled
               **{k: v for k, v in g.items() if k.startswith("mlp_")}}
        for k, v in red.items():
            acc[k] = acc.get(k, 0) + v.astype(np.float64)
        rec["clip_id"] = clip_id
        records.append(rec)
        del g
        print(f"[{ci + 1}/{len(calib)}] {clip_id} coc={rec['coc_len']} fm={rec['fm_loss']:.4f} "
              f"ports-vs-full {max(rec['ports_vs_full_relerr_by_layer']):.1e} (median layer "
              f"{np.median(rec['ports_vs_full_relerr_by_layer']):.1e}) mlp signed/full "
              f"{rec['mlp_signed_vs_full']:.4f} peak={rec['peak_gb']:.1f}GB "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ci + 1) % 10 == 0 or ci + 1 == len(calib):
            n = ci + 1
            np.savez(out_dir / "port_map.npz", **{k: v / n for k, v in acc.items()})
            (out_dir / "metrics.json").write_text(
                json.dumps({"n_clips": n, "per_clip": records}, indent=2))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
