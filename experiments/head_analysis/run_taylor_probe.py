"""Stage 1: is the first-order Taylor score comparable ACROSS the two pruning axes?

`importance.npz` scores a Q head and an MLP channel with the same construction -- a
multiplicative gate at 1.0, the same loss, the same clips, the same aggregation -- so the
two numbers carry the same units and "head h matters 500x more per parameter than channel
c" is a well-posed statement. What is not guaranteed is that the FIRST-ORDER TERM is
equally accurate on both axes: removing a head is a 128-dimensional perturbation and
removing a channel is a scalar one, so the second-order remainder need not scale the same
way. If it does not, a cross-axis ratio read off the raw scores is biased by exactly the
ratio of the two axes' prediction slopes.

This measures that slope. For each calibration clip and each probe set S of expert units:

    predicted   dL(S) = -(1/S_steps) sum_s sum_{u in S} dL_s/dg_u      (signed, one backward)
    measured    dL(S) = Lbar(mask S off) - Lbar(nothing off)           (one forward per probe)

with the SAME flow-matching loss on both sides: the same prefill cache, the same t grid
(s+0.5)/10, and ONE shared noise draw held fixed across every probe of a clip
(`noise_mode="shared"`), so probes are paired to the parameter and carry no draw variance.
The regression of measured on predicted, fit per axis, is the answer: equal slopes mean the
raw cross-axis ratios can be read quantitatively, unequal slopes give the correction factor.

Probes per layer (36 layers): every one of the 16 Q heads; eight contiguous 85-channel MLP
blocks spread over the raw-score rank order (85 channels = 522,240 params = 99.6% of one
head, the parameter-matched perturbation); and the five shipped expert-axis arm cuts
restricted to that layer. Plus the five arm masks applied to all 36 layers at once, scored
by loss and by minADE over 6 seeds, which is what ties the probe scale back to the reported
metric and tests additivity across layers.

Gates carry float32 while the masks stay in the activation dtype, so the baseline loss is
always taken from the mask path (all ones) -- the difference is then within one path.

Pre-registered gates (S0-S4) live in plans/2026-08-30_axis-taylor-comparability.md;
analyze_taylor_probe.py judges them.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 60 \
      experiments/head_analysis/run_taylor_probe.py --gpu 7 --exp-id taylor_probe_expert \
      --num-clips 32
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
import prune_lib as pl  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
BLOCK = 85  # channels whose parameter cost matches one expert Q head (99.6%)
BLOCK_POS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)  # rank position of the block
# single channels at the same rank positions: a head probe is one unit, so without these
# the MLP side of the slope test would confound "the axis" with "85 units cut at once"
SINGLE_POS = (0.0, 0.25, 0.50, 0.75, 0.90, 0.99)
# the shipped expert-axis arms, as (axis, units cut per layer); selection by the znorm score
ARMS = (("q", 4), ("q", 8), ("mlp", 341), ("mlp", 2064), ("mlp", 4128))


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_probes(raw_q, raw_m, zn_q, zn_m, n_layers, n_heads, intermediate):
    """(name, layers, axis, per-layer unit indices, n_units, kind) for every probe.

    Blocks are contiguous in the RAW rank order so a block's position has a stated
    importance; arm cuts use the znorm order because that is what built the checkpoints.
    """
    probes = []
    for li in range(n_layers):
        order_m = np.argsort(raw_m[li])  # ascending raw importance
        for h in range(n_heads):
            probes.append((f"head_l{li}_h{h}", [li], "q", {li: [h]}, 1, "unit"))
        for p in SINGLE_POS:
            c = int(order_m[min(round(p * (intermediate - 1)), intermediate - 1)])
            probes.append((f"chan_l{li}_p{int(p * 100):02d}", [li], "mlp", {li: [c]}, 1,
                           "unit"))
        for p in BLOCK_POS:
            start = round(p * (intermediate - BLOCK))
            idx = sorted(int(x) for x in order_m[start : start + BLOCK])
            probes.append((f"block_l{li}_p{int(p * 100):02d}", [li], "mlp", {li: idx},
                           BLOCK, "block"))
        for axis, k in ARMS:
            sc_ = zn_q if axis == "q" else zn_m
            idx = sorted(int(x) for x in np.argsort(sc_[li])[:k])
            probes.append((f"arm_{axis}{k}_l{li}", [li], axis, {li: idx}, k, "arm_layer"))
    # the same five cuts applied to every layer at once -- the actual shipped masks
    for axis, k in ARMS:
        sc_ = zn_q if axis == "q" else zn_m
        sel = {li: sorted(int(x) for x in np.argsort(sc_[li])[:k]) for li in range(n_layers)}
        probes.append((f"arm_{axis}{k}_full", list(range(n_layers)), axis, sel,
                       k * n_layers, "arm_full"))
    return probes


def apply_probe(masks, axis, sel):
    masks.reset()
    tgt = masks.q_mask if axis == "q" else masks.mlp_mask
    for li, idx in sel.items():
        tgt[li, torch.as_tensor(idx, device=tgt.device)] = 0.0


@torch.no_grad()
def denoise_paths(model, inputs, cache, rope_deltas, offset, prefix_mask, seeds):
    """K denoisings on a prebuilt cache -> (K, 64, 2) predicted xy (run_expert_agg's twin)."""
    preds = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask,
                                            seed=s)
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    return np.stack(preds)


@torch.no_grad()
def fm_loss(model, cache, position_ids, attention_mask, leaves, prefill, x1, noise,
            fm_steps, forward_kwargs):
    """Mean flow-matching loss over the t grid, one shared noise draw, masks as installed.

    Same construction as prune_lib.expert_fm_grads (same t values, same x_t, same target),
    so the gradient taken there and the loss measured here are of one and the same scalar.
    """
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    losses = []
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        x_t = (1.0 - t_val) * noise + t_val * x1  # (1, 64, 2)
        v_target = x1 - noise
        t = torch.full((1, 1, 1), t_val, device=x1.device)
        for i, (k, v) in enumerate(leaves):
            lib.set_cache_layer_kv(cache, i, k, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.action_in_proj(x_t.to(torch.bfloat16), t)  # (1, 64, 2048)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert(
                inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                attention_mask=attention_mask, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)
            pred = model.action_out_proj(out.last_hidden_state[:, -n_tok:])  # (1, 64, 2)
        losses.append(F.mse_loss(pred.float(), v_target).item())
    return float(np.mean(losses)), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, default="taylor_probe_expert")
    ap.add_argument("--num-clips", type=int, default=32)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--raw-importance", default="importance_stepexp_sum",
                    help="raw mean_clips|sum_s dL/dg|; the scale the comparison is read in")
    ap.add_argument("--znorm-importance", default="importance_stepexp_znorm",
                    help="the score the shipped expert-axis arms were selected with")
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=6, help="denoise seeds for the arm_full probes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--layers", type=int, default=None,
                    help="probe only the first N layers (smoke timing)")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    calib = calib[args.shard :: args.n_shards]

    raw = dict(np.load(REPO / "outputs" / args.raw_importance / "importance.npz"))
    zn = dict(np.load(REPO / "outputs" / args.znorm_importance / "importance.npz"))

    set_determinism()
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

    ec = model.expert.config
    n_layers, n_heads = ec.num_hidden_layers, ec.num_attention_heads
    inter, head_dim, hidden = ec.intermediate_size, ec.head_dim, ec.hidden_size
    p_head, p_chan = 2 * hidden * head_dim, 3 * hidden

    probes = build_probes(raw["traj_exp_q"], raw["traj_exp_mlp"],
                          zn["traj_exp_q"], zn["traj_exp_mlp"],
                          args.layers or n_layers, n_heads, inter)
    print(f"{len(probes)} probes per clip, {len(calib)} clips", flush=True)

    # installed once and left in place: an all-ones mask is an exact no-op, so the gate
    # pass and every probe run through the identical hook chain
    masks = ml.PruneMasks(model.expert.layers, n_heads, head_dim, inter, "cuda")

    rows_path = out_dir / f"rows_s{args.shard}of{args.n_shards}.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}

    for ci, (clip_id, t0_us) in enumerate(calib):
        if clip_id in done:
            continue
        t_start = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        base = sc.clip_seed(args.seed, clip_id)

        seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_end = roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        prefill = cache.get_seq_length()
        del out

        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
        gen = torch.Generator(device="cpu").manual_seed(base)
        noise = torch.randn(x1.shape, generator=gen).to(x1.device)  # (1, 64, 2), fixed

        # ---- one backward per step: signed dL_s/dg for every expert unit ----
        gates = pl.UnitGates(model.expert.layers, n_heads, head_dim, inter,
                             "cuda", torch.float32)
        grad_losses, q_sig, m_sig, _, _ = pl.expert_fm_grads_stepwise(
            model, cache, rope_deltas, x1, args.fm_steps, base, prefill, gates,
            noise_mode="shared")  # (S, L, H), (S, L, I)
        gates.remove()
        # the probe measures Lbar = mean_s L_s, so the prediction divides by S as well
        g_q = q_sig.sum(0) / args.fm_steps  # (L, H)  dLbar/dg
        g_m = m_sig.sum(0) / args.fm_steps  # (L, I)
        a_q = np.abs(q_sig).sum(0) / args.fm_steps  # sum_s |.| variant, for S4
        a_m = np.abs(m_sig).sum(0) / args.fm_steps
        del q_sig, m_sig
        torch.cuda.empty_cache()

        # ---- forward-only measurement on the same loss ----
        leaves = []
        for i in range(n_layers):
            k, v = lib.cache_layer_kv(cache, i)
            leaves.append((k.detach(), v.detach()))
        n_tok = model.action_space.get_action_space_dims()[0]
        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        position_ids, attn_mask = model._build_expert_pos_ids_and_attn_mask(
            offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
            n_diffusion_tokens=n_tok, b_star=1, device="cuda", prefix_mask=prefix_mask)
        fkw = {"is_causal": False} if model.config.expert_non_causal_attention else {}

        masks.reset()
        base_loss, base_steps = fm_loss(model, cache, position_ids, attn_mask, leaves,
                                        prefill, x1, noise, args.fm_steps, fkw)
        seeds = [base + j for j in range(args.k)]
        base_pred = denoise_paths(model, inputs, cache, rope_deltas, offset, prefix_mask,
                                  seeds)  # (K, 64, 2)
        base_ade, _ = el.ade_fde(base_pred, gt_xy)

        rec = {"clip_id": clip_id, "seed": base, "prefill": int(prefill),
               "coc_len": int(coc_end - prompt_len), "base_loss": base_loss,
               "base_loss_steps": [round(x, 10) for x in base_steps],
               "base_minADE": float(base_ade.min()),
               # S0: the gate pass's own losses must reproduce the mask path's baseline;
               # they differ only by the gates' float32 upcast of the o_proj/down_proj input
               "grad_loss": float(np.mean(grad_losses)),
               "grad_loss_steps": [round(float(x), 10) for x in grad_losses],
               "probes": []}

        for name, _lys, axis, sel, n_units, kind in probes:
            apply_probe(masks, axis, sel)
            loss, _ = fm_loss(model, cache, position_ids, attn_mask, leaves, prefill, x1,
                              noise, args.fm_steps, fkw)
            g, a = (g_q, a_q) if axis == "q" else (g_m, a_m)
            pred = -float(sum(g[li, idx].sum() for li, idx in sel.items()))
            pred_abs = float(sum(a[li, idx].sum() for li, idx in sel.items()))
            r_ = raw["traj_exp_q"] if axis == "q" else raw["traj_exp_mlp"]
            entry = {"name": name, "axis": axis, "kind": kind, "n_units": n_units,
                     "params": n_units * (p_head if axis == "q" else p_chan),
                     "pred": pred, "pred_abs": pred_abs, "loss": loss,
                     "dloss": loss - base_loss,
                     # the clip-averaged raw score of the same set -- what the shipped
                     # cross-axis comparison reads, carried alongside this clip's own term
                     "raw_I": float(sum(r_[li, idx].sum() for li, idx in sel.items()))}
            if kind == "arm_full":
                pk = denoise_paths(model, inputs, cache, rope_deltas, offset, prefix_mask,
                                   seeds)
                ade, _ = el.ade_fde(pk, gt_xy)
                entry["minADE"] = float(ade.min())
                entry["dminADE"] = float(ade.min() - base_ade.min())
            rec["probes"].append(entry)
        rows.append(rec)
        del cache, leaves, inputs

        print(f"[{ci + 1}/{len(calib)}] {clip_id} base_loss={base_loss:.5f} "
              f"minADE={rec['base_minADE']:.3f} ({time.time() - t_start:.0f}s)", flush=True)
        rows_path.write_text(json.dumps(rows))

    cfg = {"exp_id": args.exp_id, "num_clips": len(calib),
           "clip_ids": [c for c, _ in calib], "seed": args.seed,
           "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]", "fm_steps": args.fm_steps,
           "k": args.k, "noise_mode": "shared", "block": BLOCK, "block_pos": BLOCK_POS,
           "arms": [list(a) for a in ARMS], "n_probes": len(probes),
           "raw_importance": args.raw_importance, "znorm_importance": args.znorm_importance,
           "model_revision": MODEL_REV, "p_head": p_head, "p_chan": p_chan,
           "shard": [args.shard, args.n_shards], "gpu": torch.cuda.get_device_name(device),
           "single_pos": SINGLE_POS, "probe_layers": args.layers or n_layers}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
