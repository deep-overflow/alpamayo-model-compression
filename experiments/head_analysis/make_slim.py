"""Build a physically-slimmed Alpamayo-1.5 checkpoint from a validated mask config.

Configs:
  integrated_mag   -- VLM graded 30/50 + layer35(kv-only) + KV1 (trajectory Taylor),
                      expert early40/late10 (magnitude). Expected -3.25B.
  cocsafe_full_r20 -- VLM dual max(rank_traj, rank_coc) 20% + KV1(dual),
                      expert early40/late10 (magnitude). Expected -2.01B.
  *_u40_v2         -- the one-factor family, all at the grid's dual_uniform cell
                      (uniform matched budget, expert untouched, no KV drop); only the
                      within-layer score differs. Combined: dual = max(rank_traj,
                      rank_coc), j_traj = max(rank_traj, rank_J). Single-criterion
                      controls: traj, coc, j. Operator ablation: dualsum / dualprod
                      keep dual's halves but combine by rank-sum / rank-product.
                      Expected -2.66B each.

The mask recipes are imported from run_integrated / run_cocsafe -- no duplicated math.
Writes slim_state.pt + slim_meta.json to --out, then smoke-tests one val clip
(short rollout, teacher-forced NLL, one denoise, cache-shape assert).

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/make_slim.py \
      --config integrated_mag --gpu 4 --out outputs/slim_integrated_mag
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
import slim_lib as sl  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_cocsafe import rank_norm  # noqa: E402
from run_eval import eval_config  # noqa: E402
from run_grid import allocations, grid_configs  # noqa: E402
from run_integrated import expert_masks, vlm_combined_masks  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# pinned to the blobs every result in this track was produced with, matching run_baseline
MODEL_REV = sl.MODEL_REV  # single source, so a build and a load can never drift apart


def build_masks(cfg_name, imp, model, jlens="jlens_v2", vqa_imp="importance_vqa"):
    tc = model.vlm.config.text_config
    ec = model.expert.config
    emag = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                               ec.intermediate_size)
    eq, em = expert_masks(imp, emag, ec.num_hidden_layers, "magnitude")
    it = re.match(r"^(.+)_u40_it(\d+)$", cfg_name)
    uni = re.match(r"^(.+)_u(\d+)_v2$", cfg_name)
    if it:
        # Staged re-calibration masks from run_iter_prune.py: same budget, allocation
        # and axes as the *_u40_v2 family (verified by its R0 gate), only the score
        # measurement schedule differs. The masks are precomputed, so this branch just
        # loads them; plans/2026-08-16_iterative-recalibration.md.
        z = np.load(REPO / "outputs" / f"iter_{it.group(1)}_u40" / "final_masks.npz")
        vq, vm = z["vq"], z["vm"]
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif uni:
        # The one-factor family. Everything is held at the grid's dual_uniform cell --
        # uniform allocation, expert untouched, no KV drop -- so the only things that
        # vary are the within-layer score and, across the ratio sweep, the budget.
        # `dual`/`j_traj` are the combined criteria max(rank I_traj, rank X);
        # `traj`/`coc`/`j` are the single-criterion controls that say what each half of
        # that max() does on its own.
        stem, pct = uni.group(1), int(uni.group(2))
        if pct == 40:
            # u40 is NOT 0.40: it is the matched target 0.3985632694 that
            # run_grid.allocations() derives from slim_integrated_mag's realized budget.
            # Rounding it to 0.40 moves 17 MLP channels per layer, so the shipped
            # checkpoints would no longer regenerate bit-identically.
            ref_meta = json.loads(
                (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
            allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                    tc.num_attention_heads, tc.intermediate_size, 0.5)
            rq, rm = allocs["uniform"]
        else:
            # the sweep points mean exactly what their name says
            rq = np.full(tc.num_hidden_layers, pct / 100)
            rm = np.full(tc.num_hidden_layers, pct / 100)

        def half(name):
            if name == "traj":
                return imp["traj_vlm_q"], imp["traj_vlm_mlp"]
            if name == "coc":
                return imp["coc_vlm_q"], imp["coc_vlm_mlp"]
            if name in ("vqa", "coclingo"):
                # VQA-context importance and its same-images CoC control come from their
                # own run: they are measured on LingoQA train, which ships no ego
                # trajectory, so that npz holds only the two VLM arrays per objective and
                # cannot supply traj / KV / expert scores. Those still come from `imp`.
                z = dict(np.load(REPO / "outputs" / vqa_imp / "importance.npz"))
                pre = "vqa" if name == "vqa" else "coc"
                return z[f"{pre}_vlm_q"], z[f"{pre}_vlm_mlp"]
            jl = dict(np.load(REPO / "outputs" / jlens / "jlens.npz"))
            return jl["q_j"], jl["mlp_j"]

        parts = {"dual": ("traj", "coc"), "j_traj": ("traj", "j"),
                 "trajvqa": ("traj", "vqa"), "dualsum": ("traj", "coc"),
                 "dualprod": ("traj", "coc")}.get(stem, (stem,))
        # dualsum/dualprod are the operator ablation: same halves as dual, only the
        # combination differs (plans/2026-08-20_combination-operator-ablation.md)
        op = {"dualsum": np.add, "dualprod": np.multiply}.get(stem, np.maximum)
        # select_mask_ratios ranks within a layer, so rank_norm is a no-op for a single
        # criterion -- it only matters when two scores have to share one scale
        sq, sm = half(parts[0])
        for p in parts[1:]:
            oq, om = half(p)
            sq = op(rank_norm(sq), rank_norm(oq))
            sm = op(rank_norm(sm), rank_norm(om))
        vq = ml.select_mask_ratios(sq, rq)
        vm = ml.select_mask_ratios(sm, rm)
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name == "dual_uniform":
        # the grid's practical winner: dual criterion, uniform layerwise budget, VLM only.
        # Masks come straight from grid_configs so the closed-loop checkpoint is bit-identical
        # to the cell that was evaluated open-loop; expert and KV are deliberately untouched
        # so this tests exactly one factor combination and nothing else.
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        cfgs, _ = grid_configs(imp, ref_meta, tc.num_hidden_layers, tc.num_attention_heads,
                               tc.intermediate_size, 0.5)
        vq, vm = next((q, m) for n, _, q, m in cfgs if n == "dual_uniform")
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name == "integrated_mag":
        vq, vm = vlm_combined_masks(imp, tc.num_hidden_layers, tc.num_attention_heads,
                                    tc.intermediate_size)
        kvonly = (tc.num_hidden_layers - 1,)
    elif cfg_name.startswith("j_traj_full"):
        # Label-free twin of cocsafe_full: identical structure, identical ratio, identical
        # expert/KV axes -- only the reasoning half of the criterion changes, from the CoC
        # NLL Taylor score to the J-lens score. That makes the closed-loop comparison a
        # one-factor test of "do we still need CoC reference text?".
        ratio = 0.30 if cfg_name.endswith("r30") else 0.20
        all_l = list(range(tc.num_hidden_layers))
        jl = dict(np.load(REPO / "outputs" / "jlens_coc" / "jlens.npz"))
        n_per_group = tc.num_attention_heads // tc.num_key_value_heads  # GQA: 4
        # the J-lens scores Q heads and MLP channels but not KV groups, so a group's
        # J-mass is the summed squared J-score of the Q heads it feeds (group h ->
        # VLM Q heads [4h, 4h+4)) -- the same coupling mask_lib uses to remove a group
        j_kv = (jl["q_j"] ** 2).reshape(tc.num_hidden_layers, tc.num_key_value_heads,
                                        n_per_group).sum(-1)
        dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(jl["q_j"]))
        dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(jl["mlp_j"]))
        dual_kv = np.maximum(rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"]),
                             rank_norm(j_kv))
        vq = ml.select_mask(dual_q, ratio, all_l) * ml.kv_group_mask(
            dual_kv, 1, tc.num_attention_heads, all_l)
        vm = ml.select_mask(dual_m, ratio, all_l)
        kvonly = ()
    else:  # cocsafe_full_r20 / cocsafe_full_r30 -- dual width r% + KV1(dual) + expert magnitude
        ratio = 0.30 if cfg_name == "cocsafe_full_r30" else 0.20
        all_l = list(range(tc.num_hidden_layers))
        dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
        dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
        dual_kv = np.maximum(rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"]),
                             rank_norm(imp["coc_kv_k"] + imp["coc_kv_v"]))
        vq = ml.select_mask(dual_q, ratio, all_l) * ml.kv_group_mask(
            dual_kv, 1, tc.num_attention_heads, all_l)
        vm = ml.select_mask(dual_m, ratio, all_l)
        kvonly = ()
    return vq, vm, eq, em, kvonly


def expected_removed(model, vq, vm, eq, em, kvonly):
    tc = model.vlm.config.text_config
    ec = model.expert.config
    p_vq = 2 * tc.hidden_size * tc.head_dim
    p_vm = 3 * tc.hidden_size
    p_eq = 2 * ec.hidden_size * ec.head_dim
    p_em = 3 * ec.hidden_size
    removed = (int((vq == 0).sum()) * p_vq + int((vm == 0).sum()) * p_vm
               + int((eq == 0).sum()) * p_eq + int((em == 0).sum()) * p_em)
    # kv-only layers also lose q_norm (head_dim) and post_attention_layernorm (hidden)
    removed += len(kvonly) * (tc.head_dim + tc.hidden_size)
    return removed


def smoke(model, processor, clip_id, t0_us, seed=42):
    """Short rollout + TF eval + cache-shape assert on the slim model.

    Reads the per-clip cache rather than the camera chunk zip: the val chunks were
    deleted in 2026-08, so `load_physical_aiavdataset` would re-download here.
    """
    data = sc.load_cached(sc.path_for("eval", clip_id, t0_us))
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=64)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end].clone()
    for i in range(len(model.vlm.model.language_model.layers)):
        k, _ = lib.cache_layer_kv(roll["past_key_values"], i)
        assert k.shape[1] == 8, f"layer {i}: cache heads {k.shape[1]}"
    del roll
    ade, fde, nll = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, [seed])
    return {"clip_id": clip_id, "coc_len": coc_end - coc_start,
            "minADE_k1": ade, "minFDE_k1": fde, "nll": nll}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="a named config, or <criterion>_u<pct>_v2 for the uniform family "
                         "(criterion in traj|coc|j|dual|j_traj; pct 40 means the matched "
                         "0.3985632694, any other pct means exactly pct/100)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--jlens", type=str, default="jlens_v2",
                    help="J-lens run supplying q_j/mlp_j for the j_traj configs")
    ap.add_argument("--vqa-importance", type=str, default="importance_vqa",
                    help="run supplying vqa_vlm_* / coc_vlm_* for the vqa, coclingo and "
                         "trajvqa configs (measured on LingoQA train)")
    ap.add_argument("--sets-id", type=str, default="eval_sets")
    ap.add_argument("--no-state", action="store_true",
                    help="write only slim_meta.json, skipping the 16.8 GB slim_state.pt. "
                         "load_slim() reconstructs the identical model from the meta "
                         "(base weights are sliced in place, verified tensor-by-tensor), "
                         "so this is enough for evaluation; the state file is only needed "
                         "to hand the checkpoint to a machine without the base weights.")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / args.out
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    vq, vm, eq, em, kvonly = build_masks(args.config, imp, model, args.jlens,
                                         args.vqa_importance)

    full_total = sl.n_params(model)
    t0 = time.time()
    meta = sl.apply_surgery(model, vq, vm, eq, em, kvonly_layers=kvonly)
    meta["config"] = args.config
    meta["importance_from"] = args.importance
    slim_total = sl.n_params(model)
    removed = expected_removed(model, vq, vm, eq, em, kvonly)
    print(f"surgery {time.time() - t0:.0f}s: {full_total:,} -> {slim_total:,} "
          f"(-{full_total - slim_total:,}, {(full_total - slim_total) / full_total * 100:.1f}%)",
          flush=True)
    assert slim_total == full_total - removed, (slim_total, full_total - removed)
    sl.check_slim(model)
    meta["params"] = {"full": full_total, "slim": slim_total, "removed": removed}

    t0 = time.time()
    sl.save_slim(model, meta, out_dir, write_state=not args.no_state)
    kind = "recipe" if args.no_state else "checkpoint"
    print(f"saved {kind} in {time.time() - t0:.0f}s -> {out_dir}", flush=True)

    man = pd.read_parquet(REPO / "outputs" / args.sets_id / "indist_500.parquet")
    result = smoke(model, processor, man.clip_id.iloc[0], int(man.t0_us.iloc[0]))
    print(f"smoke: {result}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "config": args.config,
        "importance_from": args.importance, "jlens_from": args.jlens,
        "params": meta["params"],
        "kvonly_layers": list(kvonly),
        "kept_q_per_layer": {"vlm": [len(m["q"]) for m in meta["vlm"]],
                             "expert": [len(m["q"]) for m in meta["expert"]]},
        "smoke": result, "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    (out_dir / "summary.txt").write_text(
        f"slim checkpoint {args.config}\n"
        f"params {full_total:,} -> {slim_total:,} "
        f"(-{removed:,}, {removed / full_total * 100:.1f}%)\n"
        f"smoke clip {result['clip_id']}: NLL {result['nll']:.4f} "
        f"minADE@1 {result['minADE_k1']:.3f}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
