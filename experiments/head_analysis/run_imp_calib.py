"""Calibrate the first-order Taylor importance against realized ablation loss.

The Taylor score |dL/dg| is a *magnitude* prediction: with the gate going 1 -> 0 the
first-order predicted damage of removing unit u is exactly the (signed) gradient. The
pruning criteria only ever use its per-layer rank, so this run asks what that discards:

  per clip  1. intact rollout               -> the CoC text every measurement teacher-forces
            2. one dual-objective backward  -> this clip's predicted dL_CoC/dg, dL_traj/dg
               (run_importance's exact path, so no dependence on importance_v2's text/GPU)
            3. reference forward-only pass  -> L_CoC, L_traj of the intact model
            4. per sampled unit/group: mask via mask_lib, same text + same FM noise,
               forward-only                 -> realized dL_CoC(u), dL_traj(u), paired

The reference is re-measured after the sweep and must match the first bitwise, which is
the determinism claim the pairing rests on (recorded per clip, warned if violated).

Gates (pre-registered in plans/2026-08-15_importance-calibration.md):
  G-CAL-A  pooled within-layer Spearman(pred, realized) >= 0.7 -> ordering is reliable
  G-CAL-B  log-log Pearson r >= 0.7 on realized dL > 0 -> magnitude carries signal
           (then a loss-unit knapsack combination is worth building; otherwise rank_norm
           stands and no normalisation experiment is run)
  G-CAL-C  realized group dL vs the sum of its members' singles on the actual u40 Q cut:
           median |log2 ratio| > 1 -> first-order additivity is broken at the operating
           point, which makes iterative re-calibration the priority follow-up

Sampled: layers 1/8/15/22/29/35; all 32 Q heads per layer; ~48 stratified MLP channels
(24 even in traj rank + 12 even in coc rank + 12 inside +-5% of the dual cut); groups are
the realized u40 cuts per criterion (Q bottom-13, MLP bottom-4898 for traj/coc/dual/jtraj)
plus small subsets of the measured singles (bottom/random/cut-region at k=8 and 32).

Usage:
  bash experiments/head_analysis/run_retry_host.sh 30 \
      experiments/head_analysis/run_imp_calib.py --gpu 4 --shard 0 --n-shards 2
  ... --smoke   # 1 clip, 1 layer, a few units, for a fast end-to-end check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib
import mask_lib as ml
import prune_lib as pl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_cocsafe import rank_norm

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"

LAYERS = (1, 8, 15, 22, 29, 35)
# the realized u40_v2 cut: every layer prunes 13/32 Q heads and 4898/12288 MLP channels
CUT_Q = 13
CUT_MLP = 4898


def criterion_scores(imp, jl):
    """The four within-layer scores exactly as make_slim.build_masks composes them."""
    crit_q = {
        "traj": imp["traj_vlm_q"],
        "coc": imp["coc_vlm_q"],
        "dual": np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),
        "jtraj": np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(jl["q_j"])),
    }
    crit_m = {
        "traj": imp["traj_vlm_mlp"],
        "coc": imp["coc_vlm_mlp"],
        "dual": np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"])),
        "jtraj": np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(jl["mlp_j"])),
    }
    return crit_q, crit_m


def sample_units(imp, jl, layers):
    """Deterministic per-layer unit/group sample, drawn from the shipped v2 score frame.

    The sampling frame is importance_v2 + jlens_v2 (what the actual configs used); the
    calibration comparison itself uses the per-clip scores this run measures, so the
    frame only decides *which* units get measured, never how they are judged.
    """
    crit_q, crit_m = criterion_scores(imp, jl)
    n_mlp = imp["traj_vlm_mlp"].shape[1]
    w = round(0.05 * n_mlp)
    rng = np.random.default_rng(2026)
    units, groups = {}, {}
    for li in layers:
        order_traj = np.argsort(imp["traj_vlm_mlp"][li])
        order_coc = np.argsort(imp["coc_vlm_mlp"][li])
        order_dual = np.argsort(crit_m["dual"][li])
        rank_dual = np.argsort(order_dual)
        pick = [order_traj[r] for r in np.round(np.linspace(0, n_mlp - 1, 24)).astype(int)]
        pick += [order_coc[r] for r in np.round(np.linspace(0, n_mlp - 1, 12)).astype(int)]
        pick += [order_dual[r] for r in
                 np.round(np.linspace(CUT_MLP - w, CUT_MLP + w - 1, 12)).astype(int)]
        mlp_ids = sorted({int(c) for c in pick})
        units[li] = {"q": list(range(imp["traj_vlm_q"].shape[1])), "mlp": mlp_ids}

        g = []
        for name in ("traj", "coc", "dual", "jtraj"):
            g.append((f"qcut_{name}", "q",
                      sorted(np.argsort(crit_q[name][li])[:CUT_Q].tolist())))
            g.append((f"mcut_{name}", "mlp",
                      sorted(np.argsort(crit_m[name][li])[:CUT_MLP].tolist())))
        # small groups are subsets of the measured singles, so the realized single-unit
        # sum exists for them and G-CAL-C can compare it to the realized group loss
        by_traj = sorted(mlp_ids, key=lambda c: imp["traj_vlm_mlp"][li][c])
        cut_region = [c for c in mlp_ids if abs(int(rank_dual[c]) - CUT_MLP) <= w]
        g.append(("msub_trajbot8", "mlp", by_traj[:8]))
        g.append(("msub_trajbot32", "mlp", by_traj[:32]))
        g.append(("msub_rand8", "mlp",
                  sorted(rng.choice(np.array(mlp_ids), 8, replace=False).tolist())))
        g.append(("msub_rand32", "mlp",
                  sorted(rng.choice(np.array(mlp_ids), 32, replace=False).tolist())))
        g.append(("msub_cut8", "mlp", cut_region[:8]))
        groups[li] = g
    return units, groups


def signed_grads(gates):
    """Signed gate gradients; predicted removal damage is -grad (gate 1 -> 0)."""
    q = np.stack([g.grad.float().cpu().numpy() if g.grad is not None
                  else np.zeros(gates.n_heads) for g in gates.q_gates])
    m = np.stack([g.grad.float().cpu().numpy() if g.grad is not None
                  else np.zeros(gates.intermediate) for g in gates.mlp_gates])
    return q, m


def fm_forward(model, cache, rope_deltas, x1, fm_steps, seed, prefill):
    """Forward-only twin of prune_lib.expert_fm_grads: same t grid, same seeded noise."""
    device = x1.device
    offset = torch.tensor([prefill], device=device)
    prefix_mask = torch.ones(1, prefill, device=device, dtype=torch.long)
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = torch.randn(x1.shape, generator=gen).to(device)  # (1, 64, 2)
        x_t = (1.0 - t_val) * noise + t_val * x1
        v_target = x1 - noise
        t = torch.full((1, 1, 1), t_val, device=device)
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
    return float(np.mean(losses))


def eval_losses(model, seq_tf, tokenized_data, coc_start, coc_end, x1, fm_steps, seed):
    """One paired forward-only measurement under the currently set masks."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, tokenized_data, use_cache=True)
        nll = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end).item()
        prefill = cache.get_seq_length()
        fm = fm_forward(model, cache, rope_deltas, x1, fm_steps, seed, prefill)
    del hidden, cache
    return nll, fm


def process_clip(model, processor, masks, data, clip_id, args, layers, units, groups):
    tc = model.vlm.config.text_config
    seed = sc.clip_seed(args.seed, clip_id)
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
    del roll
    if coc_end - coc_start < 2:
        return None, {"clip_id": clip_id, "skipped": "degenerate rollout"}
    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)

    # ---- predicted scores: one dual-objective backward, run_importance's exact path ----
    vlm_gates = pl.UnitGates(model.vlm.model.language_model.layers, tc.num_attention_heads,
                             tc.head_dim, tc.intermediate_size, "cuda", torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rope_deltas = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=True)
    cache_t = pl.retain_cache_grads(cache, tc.num_hidden_layers)
    prefill = cache.get_seq_length()

    nll_graph = pl.coc_nll(model, hidden, seq_tf, coc_start, coc_end)
    nll_graph_val = float(nll_graph.item())
    nll_graph.backward(retain_graph=True)
    grad_coc_q, grad_coc_m = signed_grads(vlm_gates)
    vlm_gates.zero_grads()
    for k, v in cache_t:
        k.grad = None
        v.grad = None

    fm_graph, grads, leaves = pl.expert_fm_grads(
        model, cache, rope_deltas, x1, args.fm_steps, seed, prefill)
    ts, gs = [], []
    for (k, v), (gk, gv) in zip(cache_t, grads):
        for t_, g_ in ((k, gk), (v, gv)):
            if g_ is not None:
                ts.append(t_)
                gs.append(g_.to(t_.dtype))
    torch.autograd.backward(ts, gs)
    grad_traj_q, grad_traj_m = signed_grads(vlm_gates)
    vlm_gates.remove()
    del hidden, cache, cache_t, leaves, grads, ts, gs, nll_graph
    torch.cuda.empty_cache()

    # ---- realized losses: forward-only sweep under masks, paired to one reference ----
    def measure():
        return eval_losses(model, seq_tf, inputs["tokenized_data"], coc_start, coc_end,
                           x1, args.fm_steps, seed)

    def pred(gq, gm, axis, li, ids):
        g = gq if axis == "q" else gm
        sel = g[li, ids]
        return float(-sel.sum()), float(np.abs(sel).sum())

    masks.reset()
    ref_nll, ref_fm = measure()
    rows = []
    for li in layers:
        for axis, id_list in (("q", units[li]["q"]), ("mlp", units[li]["mlp"])):
            for u in id_list:
                masks.reset()
                (masks.q_mask if axis == "q" else masks.mlp_mask)[li, u] = 0.0
                nll, fm = measure()
                pc, pca = pred(grad_coc_q, grad_coc_m, axis, li, [u])
                pt, pta = pred(grad_traj_q, grad_traj_m, axis, li, [u])
                rows.append({"clip_id": clip_id, "layer": li, "axis": axis, "kind": "unit",
                             "name": f"{axis}{u}", "n_ids": 1, "ids": [int(u)],
                             "dnll": nll - ref_nll, "dfm": fm - ref_fm,
                             "pred_coc": pc, "pred_coc_abs": pca,
                             "pred_traj": pt, "pred_traj_abs": pta})
        for name, axis, ids in groups[li]:
            masks.reset()
            (masks.q_mask if axis == "q" else masks.mlp_mask)[li, ids] = 0.0
            nll, fm = measure()
            pc, pca = pred(grad_coc_q, grad_coc_m, axis, li, ids)
            pt, pta = pred(grad_traj_q, grad_traj_m, axis, li, ids)
            rows.append({"clip_id": clip_id, "layer": li, "axis": axis, "kind": "group",
                         "name": name, "n_ids": len(ids),
                         "ids": [int(i) for i in ids] if len(ids) <= 40 else None,
                         "dnll": nll - ref_nll, "dfm": fm - ref_fm,
                         "pred_coc": pc, "pred_coc_abs": pca,
                         "pred_traj": pt, "pred_traj_abs": pta})
    masks.reset()
    ref2_nll, ref2_fm = measure()
    deterministic = (ref2_nll == ref_nll) and (ref2_fm == ref_fm)
    if not deterministic:
        print(f"WARNING {clip_id}: reference drifted "
              f"nll {ref_nll!r}->{ref2_nll!r} fm {ref_fm!r}->{ref2_fm!r}", flush=True)
    meta = {"clip_id": clip_id, "coc_len": coc_end - coc_start, "prompt_len": prompt_len,
            "ref_nll": ref_nll, "ref_fm": ref_fm, "nll_graph": float(nll_graph_val),
            "fm_graph": fm_graph, "deterministic": deterministic,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3}
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=25)
    ap.add_argument("--exp-id", type=str, default="imp_calib_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--importance", type=str, default="importance_v2")
    ap.add_argument("--jlens", type=str, default="jlens_v2")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 clip, 1 layer, 4+4 units, 2 groups -- end-to-end check")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    jl = dict(np.load(REPO / "outputs" / args.jlens / "jlens.npz"))
    layers = (15,) if args.smoke else LAYERS
    units, groups = sample_units(imp, jl, layers)
    if args.smoke:
        li = layers[0]
        units[li] = {"q": units[li]["q"][:4], "mlp": units[li]["mlp"][:4]}
        groups[li] = [g for g in groups[li] if g[0] in ("qcut_dual", "msub_trajbot8")]

    clips = sc.calib_clips(REPO, "calib_100")[: args.num_clips]
    my_clips = clips[args.shard :: args.n_shards]
    if args.smoke:
        my_clips = my_clips[:1]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    # frozen weights + gates downstream of k/v: without this the early layers' cache
    # tensors carry no graph and the traj backward cannot reach the gates
    model.vlm.enable_input_require_grads()

    tc = model.vlm.config.text_config
    masks = ml.PruneMasks(model.vlm.model.language_model.layers, tc.num_attention_heads,
                          tc.head_dim, tc.intermediate_size, "cuda")

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "calibration of Taylor importance vs realized ablation loss",
        "plan": "plans/2026-08-15_importance-calibration.md",
        "layers": list(layers), "cut_q": CUT_Q, "cut_mlp": CUT_MLP,
        "units_per_layer": {str(li): {k: len(v) for k, v in units[li].items()}
                            for li in layers},
        "groups_per_layer": {str(li): [(n, a, len(i)) for n, a, i in groups[li]]
                             for li in layers},
        "num_clips": len(clips), "clip_ids": clips, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "importance_frame": args.importance, "jlens_frame": args.jlens,
        "fm_steps": args.fm_steps, "max_gen": args.max_gen, "smoke": args.smoke,
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    tag = f"s{args.shard}of{args.n_shards}"
    all_rows, metas = [], []
    for ci, clip_id in enumerate(my_clips):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        data = sc.load_cached(sc.path_for("calib", clip_id, sc.CALIB_T0))
        rows, meta = process_clip(model, processor, masks, data, clip_id, args,
                                  layers, units, groups)
        metas.append(meta)
        if rows is not None:
            all_rows.extend(rows)
        (out_dir / f"records_{tag}.json").write_text(json.dumps(all_rows))
        (out_dir / f"meta_{tag}.json").write_text(json.dumps(metas, indent=2))
        if rows is None:
            print(f"[{ci + 1}/{len(my_clips)}] {clip_id} SKIPPED", flush=True)
            continue
        print(f"[{ci + 1}/{len(my_clips)}] {clip_id} rows={len(rows)} "
              f"det={meta['deterministic']} peak={meta['peak_gb']:.1f}GB "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
